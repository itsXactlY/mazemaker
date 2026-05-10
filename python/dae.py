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
