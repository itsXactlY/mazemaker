"""Unified continuous-salience scoring law.

Per Sprint 2 Phase 7 Commit 8 / handoff Section 7.3. Establishes the
non-negotiable: RRF and rank-only fusion are CANDIDATE FEATURES, never
the final ranking authority. The final law is salience-weighted continuous
scoring over feature-vector channels.

Bad:    final_score = RRF(...)
Good:   final_score = salience * confidence * weighted_continuous_features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Per-channel feature weights. Defaults tuned per addendum guidance; can be
# overridden per query via intent_edge_weights or per-config.
DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic":   0.30,
    "sparse":     0.15,
    "graph":      0.20,
    "temporal":   0.10,
    "entity":     0.10,
    "procedural": 0.05,
    "locus":      0.03,
    "rrf":        0.07,  # bounded contribution; RRF as feature, never authority
}


@dataclass(frozen=True)
class ScoringConfig:
    """Frozen scoring config surfaced to callers (and tests).

    `final_authority` is the contract-level claim: the law that decides ranks.
    Tests verify it equals 'continuous_salience_score' (not 'rrf').

    `features` enumerates which signals can influence rank. RRF appears here
    as 'rrf_feature' to make explicit that it is one input among many, not
    the rank law.
    """
    final_authority: str = "continuous_salience_score"
    features: tuple[str, ...] = (
        "semantic",
        "sparse",
        "graph",
        "temporal",
        "entity",
        "procedural",
        "locus",
        "salience_multiplier",
        "confidence",
        "contradiction_penalty",
        "stale_penalty",
        "rrf_feature",
    )
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


@dataclass
class CandidateFeatures:
    """Per-candidate scoring inputs. Channels populate the fields they own;
    unset fields default to 0 / 1 / None as appropriate."""
    memory_id: int
    semantic_score: float = 0.0
    sparse_score: float = 0.0
    graph_score: float = 0.0
    temporal_score: float = 0.0
    entity_score: float = 0.0
    procedural_score: float = 0.0
    locus_score: float = 0.0
    rrf_feature: float = 0.0
    salience: float = 1.0
    confidence: float = 1.0
    contradiction_penalty: float = 0.0
    stale_penalty: float = 0.0


def score_candidate(
    f: CandidateFeatures,
    weights: Optional[dict[str, float]] = None,
    *,
    cross_encoder_score: Optional[float] = None,
    beta: float = 0.0,
) -> float:
    """Compute the final continuous score for a candidate.

    Formula:
        base = w.semantic * semantic + w.sparse * sparse + w.graph * graph
             + w.temporal * temporal + w.entity * entity
             + w.procedural * procedural + w.locus * locus
             + w.rrf * rrf_feature
        memory_score = salience * confidence * base
        memory_score -= contradiction_penalty
        memory_score -= stale_penalty
        if cross_encoder_score and beta > 0:
            return (1 - beta) * memory_score + beta * cross_encoder_score
        return memory_score
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    base = (
        w.get("semantic", 0.0) * f.semantic_score
        + w.get("sparse", 0.0) * f.sparse_score
        + w.get("graph", 0.0) * f.graph_score
        + w.get("temporal", 0.0) * f.temporal_score
        + w.get("entity", 0.0) * f.entity_score
        + w.get("procedural", 0.0) * f.procedural_score
        + w.get("locus", 0.0) * f.locus_score
        + w.get("rrf", 0.0) * f.rrf_feature
    )
    memory_score = f.salience * f.confidence * base
    memory_score -= f.contradiction_penalty
    memory_score -= f.stale_penalty

    if cross_encoder_score is None or beta <= 0.0:
        return memory_score
    return (1.0 - beta) * memory_score + beta * cross_encoder_score
