"""Acceptance tests for Phase 7.5-α — procedural_score auto-population.

Caught 2026-05-01: scoring.py:108 weights `f.procedural_score` in the unified
formula, but no production write-path ever set procedural_score. Live DB had
0/142 procedural memories with the score populated, so the procedural channel
contributed nothing to ranking. This shipped a wire — kind='procedural' →
procedural_score=0.7 baseline.

Three contracts:
  1. remember(kind='procedural') auto-sets procedural_score=0.7
  2. remember(kind='experience') leaves procedural_score NULL
  3. Explicit procedural_score kwarg overrides the auto-default
  4. SQLiteStore.store accepts procedural_score kwarg and persists it

Stdlib unittest. Run:
    python3.11 python/test_procedural_score_wiring.py
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_client import NeuralMemory, SQLiteStore  # noqa: E402


class ProceduralScoreWiringTests(unittest.TestCase):
    """End-to-end: NeuralMemory.remember() populates procedural_score."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".db", delete=False
        )
        self._tmp.close()
        self.db_path = self._tmp.name
        self.mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="auto",
            use_cpp=False,
            use_hnsw=False,
        )

    def tearDown(self) -> None:
        try:
            self.mem.close()
        except Exception:
            pass
        Path(self.db_path).unlink(missing_ok=True)

    def _read_score(self, mem_id: int) -> object:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT procedural_score FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def test_procedural_kind_auto_default(self) -> None:
        mid = self.mem.remember(
            "To file an estimate: 1. Open form 2. Enter customer 3. Submit",
            kind="procedural",
        )
        self.assertEqual(self._read_score(mid), 0.7)

    def test_experience_kind_stays_null(self) -> None:
        mid = self.mem.remember(
            "I went to the Lennar site today",
            kind="experience",
        )
        self.assertIsNone(self._read_score(mid))

    def test_explicit_score_overrides_default(self) -> None:
        mid = self.mem.remember(
            "Custom-scored procedural",
            kind="procedural",
            procedural_score=0.95,
        )
        self.assertEqual(self._read_score(mid), 0.95)

    def test_explicit_score_works_for_non_procedural_kind(self) -> None:
        # Caller can hand-set a score even on non-procedural kinds — useful
        # for hybrid memories that have some procedural-ness.
        mid = self.mem.remember(
            "Mostly experience but partly how-to",
            kind="experience",
            procedural_score=0.3,
        )
        self.assertEqual(self._read_score(mid), 0.3)


class SQLiteStorePassThroughTests(unittest.TestCase):
    """Unit: SQLiteStore.store() persists procedural_score column."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".db", delete=False
        )
        self._tmp.close()
        self.db_path = self._tmp.name

    def tearDown(self) -> None:
        Path(self.db_path).unlink(missing_ok=True)

    def test_store_accepts_and_persists_procedural_score(self) -> None:
        store = SQLiteStore(self.db_path)
        # Use a tiny embedding to avoid pulling models
        mem_id = store.store(
            label="proc test",
            content="Test row",
            embedding=[0.1, 0.2, 0.3],
            kind="procedural",
            procedural_score=0.55,
        )
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT procedural_score FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
            self.assertEqual(row[0], 0.55)
        finally:
            conn.close()

    def test_store_omits_score_when_none(self) -> None:
        store = SQLiteStore(self.db_path)
        mem_id = store.store(
            label="no score",
            content="Test row 2",
            embedding=[0.1, 0.2, 0.3],
            kind="experience",
        )
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT procedural_score FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
            self.assertIsNone(row[0])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
