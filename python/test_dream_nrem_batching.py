"""Regression coverage for D6 NREM SQLite batching.

The first real 03:45 launchd run proved a performance bug rather than a launchd
bug: NREM iterated the full active graph while SQLiteDreamBackend opened a fresh
connection and commit for each edge update + history row. This test locks the
fix to O(1) backend connections per NREM pass instead of O(N) with edge count.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream_engine import DreamEngine  # noqa: E402
from memory_client import NeuralMemory  # noqa: E402


class DreamNremBatchingTests(unittest.TestCase):
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

    def test_phase_nrem_uses_constant_backend_connections_for_edge_updates(self) -> None:
        memory_ids = [
            self.mem.remember(
                f"Valiendo D6 batching seed memory {idx}",
                detect_conflicts=False,
                kind="experience",
            )
            for idx in range(6)
        ]
        for idx, source_id in enumerate(memory_ids):
            for target_id in memory_ids[idx + 1 :]:
                self.mem.store.add_connection(source_id, target_id, 0.8, edge_type="similar")

        backend = self.engine._backend
        real_connect = backend._connect
        connect_calls = 0

        def counted_connect():
            nonlocal connect_calls
            connect_calls += 1
            return real_connect()

        backend._connect = counted_connect
        try:
            stats = self.engine._phase_nrem()
        finally:
            backend._connect = real_connect

        self.assertGreater(
            stats["strengthened"] + stats["weakened"],
            0,
            f"expected NREM to update at least one edge, got {stats}",
        )
        self.assertLessEqual(
            connect_calls,
            6,
            f"expected O(1) backend connections for NREM batching, got {connect_calls}",
        )
        with self.mem.store._lock:
            history_count = self.mem.store.conn.execute(
                "SELECT COUNT(*) FROM connection_history"
            ).fetchone()[0]
        self.assertGreater(
            history_count,
            0,
            "batched NREM updates must still write connection history rows",
        )


    def test_phase_nrem_skips_connection_history_when_disabled(self) -> None:
        """NM_DISABLE_CONN_HISTORY=1 must short-circuit history INSERTs.

        Sonnet investigation 2026-05-02 [verified-now]: zero production
        code reads connection_history. The gate prevents 7.7M+ INSERTs
        per cycle from accruing dead audit weight (61.5M pre-fix).
        """
        import os
        memory_ids = [
            self.mem.remember(
                f"D6 history-gate seed memory {idx}",
                detect_conflicts=False,
                kind="experience",
            )
            for idx in range(4)
        ]
        for idx, source_id in enumerate(memory_ids):
            for target_id in memory_ids[idx + 1 :]:
                self.mem.store.add_connection(source_id, target_id, 0.8, edge_type="similar")

        prior = os.environ.get("NM_DISABLE_CONN_HISTORY")
        os.environ["NM_DISABLE_CONN_HISTORY"] = "1"
        try:
            stats = self.engine._phase_nrem()
        finally:
            if prior is None:
                os.environ.pop("NM_DISABLE_CONN_HISTORY", None)
            else:
                os.environ["NM_DISABLE_CONN_HISTORY"] = prior

        self.assertGreater(
            stats["strengthened"] + stats["weakened"], 0,
            "NREM should still process edges with history gate on",
        )
        with self.mem.store._lock:
            history_count = self.mem.store.conn.execute(
                "SELECT COUNT(*) FROM connection_history"
            ).fetchone()[0]
        self.assertEqual(
            history_count, 0,
            "NM_DISABLE_CONN_HISTORY=1 must skip all connection_history writes",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
