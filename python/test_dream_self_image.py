"""D5 contracts: dream-engine insights must land on retrievable memory surface.

The legacy dream_insights table is analytics-only. D5 closes the useful seam by
having dream_engine create kind='dream_insight' memory nodes with summarizes
evidence edges, including a self-image synthesis pass.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream_engine import DreamEngine  # noqa: E402
from memory_client import NeuralMemory  # noqa: E402


class DreamSelfImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "memory.db")
        self.mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="hash",
            use_cpp=False,
            use_hnsw=False,
        )
        self.engine = DreamEngine.sqlite(
            self.db_path,
            neural_memory=self.mem,
            max_memories_per_cycle=20,
        )

    def tearDown(self) -> None:
        try:
            self.mem.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def _dream_insight_rows(self) -> list[tuple[int, str]]:
        with self.mem.store._lock:
            return self.mem.store.conn.execute(
                "SELECT id, content FROM memories WHERE kind = 'dream_insight' ORDER BY id"
            ).fetchall()

    def test_phase_insights_creates_retrievable_dream_insight_node(self) -> None:
        source_ids = [
            self.mem.remember(
                "Valiendo verified QBO warning truth from dashboard source.",
                detect_conflicts=False,
                kind="experience",
            ),
            self.mem.remember(
                "Valiendo verified QBO warning truth from live estimates UI.",
                detect_conflicts=False,
                kind="experience",
            ),
            self.mem.remember(
                "Valiendo patched AE operator canon after QBO warning verification.",
                detect_conflicts=False,
                kind="experience",
            ),
        ]
        self.mem.store.add_connection(source_ids[0], source_ids[1], 0.9, edge_type="semantic_similar_to")
        self.mem.store.add_connection(source_ids[1], source_ids[2], 0.9, edge_type="semantic_similar_to")
        self.mem.store.add_connection(source_ids[0], source_ids[2], 0.8, edge_type="semantic_similar_to")

        stats = self.engine._phase_insights()

        rows = self._dream_insight_rows()
        self.assertGreaterEqual(stats["insights"], 1)
        summarized_seed_sets = []
        for insight_id, _content in rows:
            edge_targets = {
                e["target_id"] if e["source_id"] == insight_id else e["source_id"]
                for e in self.mem.get_edges(insight_id)
                if e["edge_type"] == "summarizes"
            }
            summarized_seed_sets.append(set(source_ids) & edge_targets)
        self.assertTrue(
            any(len(targets) >= 2 for targets in summarized_seed_sets),
            f"expected at least one dream insight to summarize seeded memories; got {summarized_seed_sets}",
        )
        recalled = self.mem.recall("QBO warning verification", k=5, kind="dream_insight")
        self.assertTrue(recalled, "kind-filtered recall must surface the new dream insight")

    def test_self_image_phase_creates_self_image_legacy_and_retrievable_node(self) -> None:
        source_ids = [
            self.mem.remember(
                "Valiendo verified WA direct RPC readiness without sending outbound WhatsApp messages.",
                detect_conflicts=False,
                kind="experience",
            ),
            self.mem.remember(
                "Hermes patched the WA poll launchd path and preserved no-send boundaries.",
                detect_conflicts=False,
                kind="experience",
            ),
            self.mem.remember(
                "I ACKed Claude Code after checking runtime proof instead of trusting claims.",
                detect_conflicts=False,
                kind="experience",
            ),
        ]

        stats = self.engine._phase_self_image()

        self.assertEqual(stats["insights"], 1)
        with self.mem.store._lock:
            legacy_count = self.mem.store.conn.execute(
                "SELECT COUNT(*) FROM dream_insights WHERE insight_type = 'self_image'"
            ).fetchone()[0]
        self.assertEqual(legacy_count, 1)
        rows = self._dream_insight_rows()
        self.assertEqual(len(rows), 1)
        insight_id = rows[0][0]
        edge_targets = {
            e["target_id"] if e["source_id"] == insight_id else e["source_id"]
            for e in self.mem.get_edges(insight_id)
            if e["edge_type"] == "summarizes"
        }
        self.assertTrue(set(source_ids).issubset(edge_targets))
        recalled = self.mem.recall("WA direct RPC readiness", k=5, kind="dream_insight")
        self.assertTrue(recalled, "self-image insights must be retrievable by kind-filtered recall")


if __name__ == "__main__":
    unittest.main(verbosity=2)
