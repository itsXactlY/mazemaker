"""
dae.py — Dream-Augmented Embeddings (Pro feature, scaffolding stage).

DAE gives every memory a *second* embedding produced during the NREM
phase of the dream engine.  Where the *primary* embedding represents
the memory's own surface content (the embedder's view of the text),
the *DAE* embedding represents the memory's <em>neighbourhood</em> —
a graph-PPR-weighted average of the embeddings of memories the engine
consolidated it with during sleep.

The thesis is that recall over the DAE column should win on
context-dependent queries that the primary embedding misses (the
classic "what did Alice say about the migration" where the question
matches a turn semantically adjacent to the answer rather than the
answer itself).  The thesis is unproven on our 195 k-corpus.  Until a
LongMemEval-S run shows DAE produces a measurable lift over hybrid +
ColBERT@1.5 (current verified ceiling: R@5 = 0.9787, MRR = 0.9114),
this module ships scaffolding only:

  * The license gate (``has_feature("dae")``) gates module init —
    community / free / payg builds short-circuit cleanly here so
    the rest of the engine never sees a half-loaded DAE state.
  * The schema is bootstrapped per-backend: SQLiteStore.ensure_dae_schema()
    runs the legacy CREATE TABLE; PostgresStore._ensure_schema() creates
    `memory_dae_embeddings` on first connect.
  * The compute path is wired into the dream cycle as `_phase_dae()` —
    runs every MM_DAE_RECOMPUTE_EVERY cycles (default 5) AFTER NREM /
    REM / Insight / AFE so the neighbour graph it averages over is
    the freshly-consolidated one.
  * The recall path is wired in `Mazemaker._dae_score_candidates` and
    dispatches via `store.fetch_dae_vectors()` — both SQLiteStore and
    PostgresStore expose the method, so DAE works on either backend.

Until 2026-05-14 the compute path was SQLite-only and the recall path
silently disabled on PG via `engine_config.py`. Every bench number prior
to that date ran without the DAE channel — see
[[bug-dae-disabled-on-pg]] for the full root-cause writeup.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from license import has_feature

logger = logging.getLogger(__name__)


# Schema version for the dae_embeddings table.  Bump on column changes;
# the migrator picks up the new version and runs the ALTER.
DAE_SCHEMA_VERSION = 1

# Default PPR mixing weight: how much of the DAE embedding comes from
# the memory itself vs. the PPR-weighted neighbour average.  0.0 =
# pure neighbourhood (zero self-content), 1.0 = identical to the
# primary embedding (no signal).  0.4 lands the embedding meaningfully
# in the neighbour cluster while keeping the memory's own surface
# semantics readable.  Tuneable per-deployment via ``compute.toml``;
# the bench will sweep this knob.
DEFAULT_SELF_WEIGHT = 0.4

# Top-k neighbours used for the PPR-weighted average.  Larger k
# dilutes the DAE signal toward the corpus mean; smaller k makes the
# embedding brittle to the specific neighbours sampled.  20 is a
# reasonable starting point for a 100 k-corpus; the bench will sweep.
DEFAULT_NEIGHBOUR_K = 20


@dataclass(frozen=True)
class DaeEmbedding:
    memory_id: int
    vector: Sequence[float]
    self_weight: float
    neighbour_k: int
    schema_version: int
    computed_at: float


def is_enabled() -> bool:
    """True when the runtime license grants the DAE feature.

    Cheap — calls into the cached license singleton.  Safe to call
    from hot paths because the license is parsed once at startup.
    """
    return has_feature("dae")


def ensure_schema(conn) -> bool:
    """Create the ``memory_dae_embeddings`` table if missing.

    Idempotent.  Returns True if the table now exists, False if the
    DAE feature is gated off (community / free / payg) — in which
    case the table is intentionally not created.

    The schema mirrors the primary embedding table 1:1 so the bench
    harness can A/B by swapping the join target without touching
    the recall path.
    """
    if not is_enabled():
        return False

    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memory_dae_embeddings (
            memory_id        INTEGER PRIMARY KEY,
            vector           BLOB NOT NULL,
            self_weight      REAL NOT NULL,
            neighbour_k      INTEGER NOT NULL,
            schema_version   INTEGER NOT NULL,
            computed_at      REAL NOT NULL,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_dae_computed_at
        ON memory_dae_embeddings(computed_at)
    """)
    conn.commit()
    return True


def compute_dae_embedding(
    memory_id: int,
    self_vec: Sequence[float],
    neighbour_vecs: Sequence[Sequence[float]],
    neighbour_weights: Sequence[float],
    self_weight: float = DEFAULT_SELF_WEIGHT,
) -> Optional[DaeEmbedding]:
    """Mix self-vector + PPR-weighted neighbour mean.

    Vectorised with numpy. The previous Python-loop form computed
    `for i in range(dim): neighbour_mean[i] += vec[i] * w` for every
    seed × every neighbour, on a 1024-dim vector, which dominated
    dae_bulk_compute on 100k+ corpora (CPU pegged for minutes). The
    numpy matmul drops the per-seed math from ~5 ms to ~0.05 ms.

    Returns None if the feature is gated off — the caller must
    handle that case rather than silently writing zero vectors.
    """
    if not is_enabled():
        return None
    if len(neighbour_vecs) != len(neighbour_weights):
        raise ValueError(
            f"neighbour_vecs ({len(neighbour_vecs)}) and "
            f"neighbour_weights ({len(neighbour_weights)}) length mismatch"
        )
    now = time.time()
    if not neighbour_vecs:
        return DaeEmbedding(
            memory_id=memory_id,
            vector=tuple(float(x) for x in self_vec),
            self_weight=1.0,
            neighbour_k=0,
            schema_version=DAE_SCHEMA_VERSION,
            computed_at=now,
        )

    import numpy as _np
    weight_sum = float(sum(neighbour_weights))
    if weight_sum <= 0:
        # Defensive: zero-weight neighbours collapse to self.
        return DaeEmbedding(
            memory_id=memory_id,
            vector=tuple(float(x) for x in self_vec),
            self_weight=1.0,
            neighbour_k=0,
            schema_version=DAE_SCHEMA_VERSION,
            computed_at=now,
        )

    self_arr = _np.asarray(self_vec, dtype=_np.float32)
    neigh = _np.asarray(neighbour_vecs, dtype=_np.float32)  # (K, D)
    if neigh.shape[1] != self_arr.shape[0]:
        raise ValueError(
            f"neighbour vector dim mismatch: {neigh.shape[1]} vs {self_arr.shape[0]}"
        )
    w = _np.asarray(neighbour_weights, dtype=_np.float32) / weight_sum  # (K,)
    neighbour_mean = neigh.T @ w  # (D,)
    mixed = self_weight * self_arr + (1.0 - self_weight) * neighbour_mean
    norm = float(_np.linalg.norm(mixed))
    if norm > 0:
        mixed = mixed / norm

    return DaeEmbedding(
        memory_id=memory_id,
        vector=tuple(float(x) for x in mixed),
        self_weight=self_weight,
        neighbour_k=len(neighbour_vecs),
        schema_version=DAE_SCHEMA_VERSION,
        computed_at=now,
    )


def dae_bulk_compute(
    nm,
    self_weight: float = DEFAULT_SELF_WEIGHT,
    neighbour_k: int = DEFAULT_NEIGHBOUR_K,
) -> dict:
    """Compute and persist DAE embeddings for every memory in ``nm``.

    Backend-agnostic: dispatches through store-level methods
    (`ensure_dae_schema`, `get_all`, `get_connections`,
    `upsert_dae_vectors`) so it runs identically against SQLiteStore
    and PostgresStore. Previously SQLite-only — the inline
    `conn.execute("... ? ...")` calls and `INSERT OR REPLACE` SQL
    crashed under PG, which is why DAE compute was never wired into
    production dream cycles.

    Returns a stats dict:
      {ok: bool, written: int, skipped: int, errors: int,
       schema_version: int, self_weight: float, neighbour_k: int}

    Short-circuits cleanly when the DAE feature is gated off (community
    / free / payg).  In that case ``ok`` is False and ``skipped`` equals
    the corpus size — the caller can branch on ok without checking
    has_feature directly.
    """
    if not is_enabled():
        return {"ok": False, "reason": "license_gate", "written": 0,
                "skipped": 0, "errors": 0}

    store = getattr(nm, "store", None)
    if store is None:
        return {"ok": False, "reason": "no_store", "written": 0,
                "skipped": 0, "errors": 0}

    # Schema bootstrap — store decides how (SQLite calls the legacy
    # ensure_schema(conn) helper; PG creates the table inside
    # _ensure_schema on first connect). Both return True on success.
    if not getattr(store, "ensure_dae_schema", lambda: False)():
        return {"ok": False, "reason": "no_dae_schema", "written": 0,
                "skipped": 0, "errors": 0}

    import struct

    # Pull every (id, embedding) via the backend-agnostic get_all().
    # On large production corpora this materialises the whole vector
    # set in memory; bench-scale (≤500 rows) is the common case. A
    # paged variant lives in the TODO list for ≥1M-row backfills.
    vectors: "dict[int, list[float]]" = {}
    for m in store.get_all():
        emb = m.get("embedding")
        if emb:
            vectors[int(m["id"])] = list(emb)
    if not vectors:
        return {"ok": True, "written": 0, "skipped": 0, "errors": 0,
                "schema_version": DAE_SCHEMA_VERSION,
                "self_weight": self_weight, "neighbour_k": neighbour_k}

    def _neighbours_via_ppr(seed_id: int) -> "tuple[list[int], list[float]]":
        try:
            ids = nm._ppr_top_ids_gpu(int(seed_id), k=neighbour_k)
        except Exception:
            ids = []
        if not ids:
            return [], []
        # Equal weights for the top-k — _ppr_top_ids_gpu doesn't return
        # the scalar PPR values, just the rank-ordered ids. Equal-weight
        # mean is a reasonable approximation; the rank-ordered top-k
        # captures the structure.  If a lift is observed and we want to
        # squeeze more, switch to the full _ppr_scores call.
        return list(ids), [1.0] * len(ids)

    def _neighbours_via_graph(seed_id: int) -> "tuple[list[int], list[float]]":
        """Backend-agnostic top-k outgoing edges by weight.

        store.get_connections returns dicts with at least `target` /
        `weight` keys on both SQLiteStore and PostgresStore. We filter
        for positive-weight, sort descending, and take the top-k.
        """
        try:
            edges = store.get_connections(int(seed_id)) or []
        except Exception:
            return [], []
        scored: "list[tuple[int, float]]" = []
        for e in edges:
            try:
                src = int(e.get("source") if "source" in e else e.get("source_id"))
                tgt = int(e.get("target") if "target" in e else e.get("target_id"))
                w = float(e.get("weight") or 0.0)
            except Exception:
                continue
            if w <= 0:
                continue
            other = tgt if src == int(seed_id) else src
            if other == int(seed_id):
                continue
            scored.append((other, w))
        scored.sort(key=lambda kv: -kv[1])
        scored = scored[: int(neighbour_k)]
        return [n for n, _ in scored], [w for _, w in scored]

    # Neighbour source — default to direct graph edges, which on a
    # sparse corpus (avg degree ~3) are already cheap and capture the
    # signal DAE needs. Per-seed GPU PPR costs ~150 ms/call (full 20-
    # iter sparse_mm on the entire 89k-node adjacency), which on a
    # 100k seeds × 167ms = 4.6 h dominated Stage 1 wall-clock. PPR
    # adds value when the corpus is densely connected; opt in via
    # MAZEMAKER_DAE_NEIGHBOURS=ppr.
    import os as _os
    _neigh_mode = (_os.environ.get("MAZEMAKER_DAE_NEIGHBOURS") or "graph").strip().lower()
    use_ppr = (_neigh_mode == "ppr") and hasattr(nm, "_ppr_top_ids_gpu")
    written = 0
    skipped = 0
    errors = 0
    insert_rows: "list[tuple]" = []
    for mid, self_vec in vectors.items():
        nids, nweights = (
            _neighbours_via_ppr(mid) if use_ppr else _neighbours_via_graph(mid)
        )
        if not nids:
            nids, nweights = _neighbours_via_graph(mid)
        nvecs = [vectors[n] for n in nids if n in vectors]
        # Filter weights to match the surviving vectors
        nweights = [w for n, w in zip(nids, nweights) if n in vectors]
        try:
            emb = compute_dae_embedding(
                memory_id=mid,
                self_vec=self_vec,
                neighbour_vecs=nvecs,
                neighbour_weights=nweights,
                self_weight=self_weight,
            )
        except Exception:
            errors += 1
            continue
        if emb is None:
            skipped += 1
            continue
        try:
            # numpy.tobytes() is ~10x faster than struct.pack(*list) on
            # 1024-d floats — same wire format (little-endian float32).
            import numpy as _np
            blob = _np.asarray(emb.vector, dtype=_np.float32).tobytes()
        except Exception:
            errors += 1
            continue
        insert_rows.append(
            (emb.memory_id, blob, emb.self_weight, emb.neighbour_k,
             emb.schema_version, emb.computed_at)
        )
        written += 1

    if insert_rows:
        try:
            store.upsert_dae_vectors(insert_rows)
        except Exception as exc:
            logger.warning("DAE upsert failed: %s", exc)
            errors += len(insert_rows)
            written = 0

    return {
        "ok": True,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "schema_version": DAE_SCHEMA_VERSION,
        "self_weight": self_weight,
        "neighbour_k": neighbour_k,
    }


def fetch_dae_vectors(conn, ids: "list[int]") -> "dict[int, list[float]]":
    """Look up DAE vectors for a list of memory ids.

    Returns {memory_id: vector}; ids without a row are absent (caller
    treats absent ids as no-DAE-signal).  Cheap: one SQL roundtrip, the
    caller's ``cap`` already limits ids to the RRF head.
    """
    if not ids:
        return {}
    import struct
    placeholders = ",".join("?" * len(ids))
    try:
        rows = conn.execute(
            f"SELECT memory_id, vector FROM memory_dae_embeddings "
            f"WHERE memory_id IN ({placeholders})",
            tuple(int(i) for i in ids),
        ).fetchall()
    except Exception:
        return {}
    out: "dict[int, list[float]]" = {}
    for mid, blob in rows:
        if not blob:
            continue
        try:
            n = len(blob) // 4
            out[int(mid)] = list(struct.unpack(f"{n}f", blob))
        except Exception:
            continue
    return out


def channel_status() -> dict:
    """Operator-facing snapshot of DAE state.

    Returned by the engine's stats endpoint when the operator queries
    the gating health.  Distinguishes:

      * "off"        — community / free / payg.  Module not loaded
                       beyond the import.
      * "loaded"     — Pro tier, schema staged, no compute wired.
                       Current ship state.
      * "writing"    — Pro tier, NREM is writing DAE rows.  Reached
                       when the bench validates and the channel
                       integration PR lands.
      * "channel-on" — Pro tier, recall path consults the DAE column.
                       Reached when the bench shows a measurable
                       lift over hybrid + ColBERT.
    """
    if not is_enabled():
        return {"status": "off", "reason": "license_gate"}
    return {
        "status": "loaded",
        "schema_version": DAE_SCHEMA_VERSION,
        "self_weight_default": DEFAULT_SELF_WEIGHT,
        "neighbour_k_default": DEFAULT_NEIGHBOUR_K,
        "wired_to_nrem": False,
        "wired_to_recall": False,
    }


# Single startup log so operators see whether the gate is on.  Repeats
# are not interesting — the license is cached and the answer doesn't
# change at runtime.
if is_enabled():
    logger.info(
        "DAE feature gate enabled (Pro tier). Module loaded; recall "
        "channel intentionally unwired pending bench validation."
    )
else:
    logger.debug("DAE feature gate off — community / free / payg tier.")
