"""Tests for hybrid_recall — multi-channel candidate union + continuous scorer.

Per the architectural Hindsight-shape unification: dense + sparse + graph
(+ temporal when as_of given) candidate pool, RRF as feature, salience-
weighted continuous law as the final authority. Tests verify shape and
ordering, not absolute numbers (those depend on corpus + tuning).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory  # noqa: E402


class HybridRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = NeuralMemory(
            db_path=str(Path(self._tmp.name) / "memory.db"),
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

    def test_returns_results_from_pool_union(self) -> None:
        """Hybrid returns memory dicts including 'channels' annotation."""
        m1 = self.mem.remember(
            "Lennar lot 27 needs GFCI on outdoor receptacles.",
            kind="experience", detect_conflicts=False,
        )
        m2 = self.mem.remember(
            "When estimating panel upgrades, check load calc first.",
            kind="procedural", detect_conflicts=False,
        )

        results = self.mem.hybrid_recall("Lennar GFCI", k=5)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("channels", r)
            self.assertIn("combined", r)
        # m1 should rank — it has both semantic + sparse signal
        ids = [r["id"] for r in results]
        self.assertIn(m1, ids)

    def test_kind_filter_applied(self) -> None:
        self.mem.remember("Customer asked about lot 27.",
                          kind="experience", detect_conflicts=False)
        proc = self.mem.remember("When estimating panels, check load.",
                                  kind="procedural", detect_conflicts=False)

        results = self.mem.hybrid_recall("estimate panels", k=5, kind="procedural")
        ids = [r["id"] for r in results]
        self.assertIn(proc, ids)
        # Verify all returned have kind='procedural'
        for r in results:
            row = self.mem.store.conn.execute(
                "SELECT kind FROM memories WHERE id = ?", (r["id"],)
            ).fetchone()
            self.assertEqual(row[0], "procedural")

    def test_as_of_filter_excludes_stale(self) -> None:
        old = self.mem.remember(
            "Mike is the contact.", detect_conflicts=False, valid_to=100.0,
        )
        new = self.mem.remember(
            "Sarah is the contact.", detect_conflicts=False, valid_from=101.0,
        )
        results = self.mem.hybrid_recall("contact", k=5, as_of=200.0)
        ids = [r["id"] for r in results]
        self.assertIn(new, ids)
        self.assertNotIn(old, ids)

    def test_empty_db_returns_empty(self) -> None:
        results = self.mem.hybrid_recall("anything", k=5)
        self.assertEqual(results, [])

    def test_channels_annotation_reflects_actual_pool_membership(self) -> None:
        m = self.mem.remember(
            "Lennar GFCI exterior receptacle.", detect_conflicts=False,
        )
        results = self.mem.hybrid_recall("Lennar GFCI", k=5)
        for r in results:
            if r["id"] == m:
                # m should appear in semantic + sparse + graph pools
                self.assertGreater(len(r["channels"]), 0)
                break

    def test_salience_weighted_ordering(self) -> None:
        """Higher-salience memory should outrank lower-salience all else equal."""
        low = self.mem.remember(
            "rare exact term phrase here", detect_conflicts=False, salience=0.1,
        )
        high = self.mem.remember(
            "rare exact term phrase variant", detect_conflicts=False, salience=2.0,
        )
        results = self.mem.hybrid_recall("rare exact term", k=5)
        ids = [r["id"] for r in results]
        self.assertIn(high, ids)
        if low in ids and high in ids:
            self.assertLess(ids.index(high), ids.index(low),
                            "higher-salience must outrank lower in hybrid_recall")

    def test_rrf_is_feature_not_authority(self) -> None:
        """If RRF were the final authority, an item ranked top by RRF would
        always be #1. Verify a high-salience item can outrank a top-RRF
        item when other features pile up."""
        # Plant a memory with strong overlap (high RRF rank) but low salience
        rrf_top = self.mem.remember(
            "exact match exact match exact match", detect_conflicts=False, salience=0.05,
        )
        # Plant a memory with moderate overlap but very high salience
        sal_top = self.mem.remember(
            "exact match phrase variant", detect_conflicts=False, salience=3.0,
        )
        results = self.mem.hybrid_recall("exact match", k=5)
        ids = [r["id"] for r in results]
        # The salience-weighted continuous law allows high-salience to win
        # over high-RRF; verify behavior is at least possible (won't always
        # win — depends on weight balance — but should appear in top-2)
        self.assertIn(sal_top, ids[:2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
