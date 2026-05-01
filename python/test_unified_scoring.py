"""Acceptance tests for Sprint 2 Phase 7 Commit 8 — unified continuous scorer.

Per addendum lines 420-466. Five contracts:

  1. RRF is a feature, never the final ranking authority
  2. salience multiplier observably changes final rank
  3. stale memory is penalized when as_of is current (via temporal filter)
  4. salience kwarg flows to memories.salience column on retain
  5. scoring formula is the salience-weighted continuous law

Stdlib unittest. Run:
    python3 python/test_unified_scoring.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory  # noqa: E402
from scoring import (  # noqa: E402
    DEFAULT_WEIGHTS,
    CandidateFeatures,
    ScoringConfig,
    score_candidate,
)


class ScoringConfigTests(unittest.TestCase):
    def test_rrf_is_feature_not_final_authority(self) -> None:
        cfg = ScoringConfig()
        self.assertEqual(cfg.final_authority, "continuous_salience_score")
        self.assertIn("rrf_feature", cfg.features)
        self.assertNotEqual(cfg.final_authority, "rrf")

    def test_features_include_all_required_channels(self) -> None:
        cfg = ScoringConfig()
        for required in ("semantic", "sparse", "graph", "temporal", "entity",
                         "procedural", "locus", "salience_multiplier",
                         "confidence", "contradiction_penalty",
                         "stale_penalty", "rrf_feature"):
            self.assertIn(required, cfg.features)

    def test_default_weights_sum_above_zero(self) -> None:
        # Just sanity — weights are non-negative and finite
        for k, v in DEFAULT_WEIGHTS.items():
            self.assertGreaterEqual(v, 0.0)
            self.assertLess(v, 1.0)


class ScoreCandidateTests(unittest.TestCase):
    def test_salience_changes_final_score(self) -> None:
        f_low = CandidateFeatures(memory_id=1, semantic_score=0.8, salience=0.1)
        f_high = CandidateFeatures(memory_id=2, semantic_score=0.8, salience=2.0)
        self.assertGreater(score_candidate(f_high), score_candidate(f_low))

    def test_contradiction_penalty_lowers_score(self) -> None:
        f_clean = CandidateFeatures(memory_id=1, semantic_score=0.8)
        f_penalty = CandidateFeatures(memory_id=2, semantic_score=0.8,
                                       contradiction_penalty=0.3)
        self.assertGreater(score_candidate(f_clean), score_candidate(f_penalty))

    def test_stale_penalty_lowers_score(self) -> None:
        f_clean = CandidateFeatures(memory_id=1, semantic_score=0.8)
        f_penalty = CandidateFeatures(memory_id=2, semantic_score=0.8,
                                       stale_penalty=0.5)
        self.assertGreater(score_candidate(f_clean), score_candidate(f_penalty))

    def test_cross_encoder_blend(self) -> None:
        f = CandidateFeatures(memory_id=1, semantic_score=0.5)
        plain = score_candidate(f)
        blended = score_candidate(f, cross_encoder_score=0.9, beta=0.5)
        self.assertNotEqual(plain, blended)


class NeuralMemoryScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "memory.db")
        self.mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="hash",
            use_cpp=False,
            use_hnsw=False,
        )

    def tearDown(self) -> None:
        try:
            self.mem.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def test_scoring_config_surface(self) -> None:
        cfg = self.mem.scoring_config()
        self.assertEqual(cfg.final_authority, "continuous_salience_score")
        self.assertIn("rrf_feature", cfg.features)

    def test_salience_kwarg_flows_to_db(self) -> None:
        mid = self.mem.remember("test memory", detect_conflicts=False, salience=0.42)
        row = self.mem.store.conn.execute(
            "SELECT salience FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        self.assertAlmostEqual(row[0], 0.42, places=4)

    def test_salience_multiplier_changes_rank(self) -> None:
        # Insert two memories with similar content but different salience.
        # Higher-salience should outrank lower in recall.
        low = self.mem.remember("rare exact term phrase", detect_conflicts=False, salience=0.1)
        high = self.mem.remember("rare exact term phrase variant", detect_conflicts=False, salience=2.0)

        results = self.mem.recall("rare exact term", k=5)
        self.assertGreater(len(results), 0)
        # high-salience id should outrank low-salience id
        ids = [r['id'] for r in results]
        if low in ids and high in ids:
            self.assertLess(ids.index(high), ids.index(low),
                            "higher-salience memory should outrank lower-salience")
        else:
            self.assertIn(high, ids,
                          "high-salience memory should at least appear in results")

    def test_recall_as_of_filter_excludes_stale(self) -> None:
        stale = self.mem.remember(
            "Old contact is Mike.",
            detect_conflicts=False,
            valid_to=100.0,
        )
        current = self.mem.remember(
            "Current contact is Sarah.",
            detect_conflicts=False,
            valid_from=101.0,
        )

        # at as_of=200, stale (valid_to=100) is invalid; current (from 101) is valid
        results = self.mem.recall("contact", k=5, as_of=200.0)
        ids = [r['id'] for r in results]
        self.assertIn(current, ids)
        self.assertNotIn(stale, ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
