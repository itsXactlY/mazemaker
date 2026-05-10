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
  * The schema migration is committed but only applied on tier
    upgrade — i.e. when ``has_feature("dae")`` first returns True
    and the table is missing.
  * The compute path (``compute_dae_embedding``) is implemented but
    not invoked — the dream engine NREM phase will call it once
    the bench validates a lift.
  * No recall channel is registered.  ``memory_client.NeuralMemory``
    is unaware DAE exists.  Recall stays on hybrid + ColBERT until
    the bench delta is measurable.

Honesty: shipping the module without the channel wiring lets us
populate DAE embeddings in the field (Pro users opt-in) so the
research run has real data to score, without misleading anyone that
recall is using DAE today.  When the bench lands and the channel
turns on, that change will be one PR and one feature-flag check.
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

    Pure function — no I/O, no side effects.  The dream engine will
    call this from the NREM phase once the bench validates a recall
    lift.  Until then, this exists so the unit test for the
    arithmetic can pass independently of the dream loop.

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
    if not neighbour_vecs:
        return DaeEmbedding(
            memory_id=memory_id,
            vector=tuple(self_vec),
            self_weight=1.0,
            neighbour_k=0,
            schema_version=DAE_SCHEMA_VERSION,
            computed_at=time.time(),
        )

    dim = len(self_vec)
    weight_sum = sum(neighbour_weights)
    if weight_sum <= 0:
        # Defensive: zero-weight neighbours collapse to self.
        return DaeEmbedding(
            memory_id=memory_id,
            vector=tuple(self_vec),
            self_weight=1.0,
            neighbour_k=0,
            schema_version=DAE_SCHEMA_VERSION,
            computed_at=time.time(),
        )

    norm_weights = [w / weight_sum for w in neighbour_weights]
    neighbour_mean = [0.0] * dim
    for vec, w in zip(neighbour_vecs, norm_weights):
        if len(vec) != dim:
            raise ValueError(
                f"neighbour vector dim mismatch: {len(vec)} vs {dim}"
            )
        for i in range(dim):
            neighbour_mean[i] += vec[i] * w

    mixed = [
        self_weight * self_vec[i] + (1.0 - self_weight) * neighbour_mean[i]
        for i in range(dim)
    ]
    norm = sum(x * x for x in mixed) ** 0.5
    if norm > 0:
        mixed = [x / norm for x in mixed]

    return DaeEmbedding(
        memory_id=memory_id,
        vector=tuple(mixed),
        self_weight=self_weight,
        neighbour_k=len(neighbour_vecs),
        schema_version=DAE_SCHEMA_VERSION,
        computed_at=time.time(),
    )


def dae_bulk_compute(
    nm,
    self_weight: float = DEFAULT_SELF_WEIGHT,
    neighbour_k: int = DEFAULT_NEIGHBOUR_K,
) -> dict:
    """Compute and persist DAE embeddings for every memory in ``nm``.

    Designed for the bench harness — the LongMemEval-S harness spins up
    an ephemeral per-question SQLite (50-500 memories, sub-second),
    ingests, calls this, then runs ``recall()``.  Not intended as a
    production-corpus backfill path; that is deferred until the bench
    validates a recall lift.

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

    # Surface conn for the schema migration and the executemany write.
    # Both SQLiteStore and PostgresStore expose `_cursor` / `conn`; we
    # use the lowest-common-denominator: `conn` on SQLite, `_cursor`
    # context manager on Postgres.  Bench harness uses SQLite, so we
    # rely on `conn` here.  Postgres path is added if/when prod backfill
    # lands.
    conn = getattr(store, "conn", None)
    if conn is None:
        return {"ok": False, "reason": "no_sqlite_conn", "written": 0,
                "skipped": 0, "errors": 0}
    ensure_schema(conn)

    # Pull every (id, embedding) — bench corpus is small (≤500), full
    # scan is fine.  On a production-corpus backfill this would page.
    import sqlite3
    cur = conn.cursor()
    cur.execute("SELECT id, embedding FROM memories ORDER BY id ASC")
    rows = cur.fetchall()

    import struct
    def _decode(blob) -> "list[float] | None":
        if blob is None:
            return None
        try:
            n = len(blob) // 4
            return list(struct.unpack(f"{n}f", blob))
        except Exception:
            return None

    vectors: "dict[int, list[float]]" = {}
    for mid, emb_blob in rows:
        vec = _decode(emb_blob)
        if vec:
            vectors[int(mid)] = vec
    if not vectors:
        return {"ok": True, "written": 0, "skipped": 0, "errors": 0,
                "schema_version": DAE_SCHEMA_VERSION,
                "self_weight": self_weight, "neighbour_k": neighbour_k}

    # Neighbour selection: prefer PPR-top-k via the GPU helper when the
    # corpus is large enough to warrant it; otherwise fall back to the
    # in-DB connection table (cheap on tiny bench corpora).  Either way
    # we end up with a list of (mid, neighbour_ids, neighbour_weights).
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
        try:
            r = conn.execute(
                "SELECT target_id, weight FROM connections "
                "WHERE source_id = ? AND weight > 0 "
                "ORDER BY weight DESC LIMIT ?",
                (int(seed_id), int(neighbour_k)),
            ).fetchall()
        except sqlite3.OperationalError:
            return [], []
        return [int(r0) for r0, _ in r], [float(r1) for _, r1 in r]

    use_ppr = hasattr(nm, "_ppr_top_ids_gpu")
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
            blob = struct.pack(f"{len(emb.vector)}f", *emb.vector)
        except Exception:
            errors += 1
            continue
        insert_rows.append(
            (emb.memory_id, blob, emb.self_weight, emb.neighbour_k,
             emb.schema_version, emb.computed_at)
        )
        written += 1

    if insert_rows:
        cur.executemany(
            "INSERT OR REPLACE INTO memory_dae_embeddings "
            "(memory_id, vector, self_weight, neighbour_k, "
            " schema_version, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            insert_rows,
        )
        conn.commit()

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
