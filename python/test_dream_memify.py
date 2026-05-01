"""Acceptance tests for Sprint 2 Phase 7 Commit 9 — dream Memify hygiene.

Per addendum lines 468-509. Five contracts:

  1. run_memify_once downweights exact-content duplicates
  2. create_insight_from_cluster creates dream_insight node with evidence edges
  3. run_contradiction_detection_once creates contradicts edge for conflicting
     time-validity pairs
  4. Memify does NOT hard-delete (preserves audit history per H19/H6 invariant)
  5. Insights without source ids are no-op

Stdlib unittest. Run:
    python3 python/test_dream_memify.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory  # noqa: E402


class DreamMemifyTests(unittest.TestCase):
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

    # ----- Memify duplicates -------------------------------------------------

    def test_memify_downweights_exact_duplicates(self) -> None:
        a = self.mem.remember("Customer asked for panel upgrade.",
                              detect_conflicts=False, salience=1.0)
        b = self.mem.remember("Customer asked for panel upgrade.",
                              detect_conflicts=False, salience=0.8)

        stats = self.mem.run_memify_once(decay_factor=0.5)
        self.assertEqual(stats["duplicates_downweighted"], 1)

        ma = self.mem.get_memory(a)
        mb = self.mem.get_memory(b)
        # higher original salience (a) survives; lower (b) gets downweighted
        self.assertEqual(ma["salience"], 1.0)
        self.assertLess(mb["salience"], 0.8)
        self.assertGreaterEqual(ma["salience"], mb["salience"])

    def test_memify_does_not_delete_records(self) -> None:
        before_count = self.mem.store.conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        self.mem.remember("dup content", detect_conflicts=False)
        self.mem.remember("dup content", detect_conflicts=False)
        self.mem.remember("dup content", detect_conflicts=False)
        self.mem.run_memify_once()
        after_count = self.mem.store.conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        self.assertEqual(after_count - before_count, 3,
                         "memify must not delete records — only downweight")

    def test_memify_no_op_when_no_duplicates(self) -> None:
        self.mem.remember("unique a", detect_conflicts=False)
        self.mem.remember("unique b", detect_conflicts=False)
        stats = self.mem.run_memify_once()
        self.assertEqual(stats["duplicates_downweighted"], 0)

    # ----- Dream insight creation -------------------------------------------

    def test_dream_insight_has_evidence_edges(self) -> None:
        e1 = self.mem.remember(
            "Customer delayed approval until itemized pricing.",
            detect_conflicts=False, kind="experience",
        )
        e2 = self.mem.remember(
            "Customer requested itemized quote.",
            detect_conflicts=False, kind="experience",
        )

        insight = self.mem.create_insight_from_cluster([e1, e2])
        self.assertGreater(insight, 0)

        edges = self.mem.get_edges(insight)
        edge_types = {e["edge_type"] for e in edges}
        self.assertTrue(
            edge_types & {"summarizes", "derived_from"},
            f"insight {insight} must have summarizes or derived_from edges; "
            f"got {edge_types}",
        )

    def test_dream_insight_kind_is_dream_insight(self) -> None:
        e = self.mem.remember("source experience", detect_conflicts=False)
        insight = self.mem.create_insight_from_cluster([e])
        row = self.mem.get_memory(insight)
        self.assertEqual(row["kind"], "dream_insight")
        self.assertEqual(row["origin_system"], "dream_engine")

    def test_create_insight_empty_cluster_is_no_op(self) -> None:
        insight = self.mem.create_insight_from_cluster([])
        self.assertEqual(insight, 0)

    # ----- Contradiction detection ------------------------------------------

    def test_contradiction_edge_for_conflicting_validity(self) -> None:
        old = self.mem.remember("Mike is current contact.",
                                detect_conflicts=False, valid_to=100.0)
        new = self.mem.remember("Sarah is current contact.",
                                detect_conflicts=False, valid_from=101.0)

        stats = self.mem.run_contradiction_detection_once()
        self.assertGreaterEqual(stats["contradiction_edges_added"], 1)
        self.assertTrue(self.mem.has_edge(old, new, edge_type="contradicts"))

    def test_contradiction_detection_skips_overlapping_validity(self) -> None:
        # Both valid simultaneously — should NOT be flagged as contradiction
        a = self.mem.remember("Some statement here.",
                              detect_conflicts=False, valid_from=100.0, valid_to=300.0)
        b = self.mem.remember("Some statement here too.",
                              detect_conflicts=False, valid_from=150.0, valid_to=350.0)
        self.mem.run_contradiction_detection_once()
        self.assertFalse(self.mem.has_edge(a, b, edge_type="contradicts"))

    def test_contradiction_detection_skips_unrelated_content(self) -> None:
        # Validity sequence holds but content totally disjoint
        a = self.mem.remember("Alpha bravo charlie.", detect_conflicts=False, valid_to=100.0)
        b = self.mem.remember("Tango foxtrot zulu.", detect_conflicts=False, valid_from=101.0)
        self.mem.run_contradiction_detection_once()
        self.assertFalse(self.mem.has_edge(a, b, edge_type="contradicts"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
