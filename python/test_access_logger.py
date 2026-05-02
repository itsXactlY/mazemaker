"""Tests for access_logger.py — JSONL recall-event observability.

Per round-4 test-coverage reviewer: AccessLogger had zero tests despite
being shipped wired into hybrid_recall. These contracts cover:
- Single-event log + read-back round-trip
- Best-effort guarantee (errors don't propagate)
- Disable-via-env env var
- Rotation at threshold
- Field schema completeness
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from access_logger import RecallAccessLogger, default_logger  # noqa: E402


class AccessLoggerBasicTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self._tmp.close()
        self.path = Path(self._tmp.name)
        self.logger = RecallAccessLogger(path=self.path)

    def tearDown(self) -> None:
        self.path.unlink(missing_ok=True)
        # Clean up any rotated logs
        for n in range(1, 1000):
            self.path.with_suffix(f".jsonl.{n}").unlink(missing_ok=True)

    def test_log_single_event_roundtrip(self) -> None:
        self.logger.log(
            query="find lennar lot 12",
            k=5,
            results=[{"id": 100}, {"id": 200}, {"id": 300}],
            latency_ms=42.5,
            channels=["semantic", "sparse"],
            method="hybrid_recall",
            kind_filter="experience",
            rerank=True,
        )
        content = self.path.read_text(encoding="utf-8").strip()
        entry = json.loads(content)
        self.assertEqual(entry["query"], "find lennar lot 12")
        self.assertEqual(entry["k"], 5)
        self.assertEqual(entry["n_results"], 3)
        self.assertEqual(entry["top_ids"], [100, 200, 300])
        self.assertEqual(entry["latency_ms"], 42.5)
        self.assertEqual(entry["channels"], ["semantic", "sparse"])
        self.assertEqual(entry["method"], "hybrid_recall")
        self.assertEqual(entry["kind"], "experience")
        self.assertTrue(entry["rerank"])

    def test_log_appends_not_overwrites(self) -> None:
        self.logger.log(query="q1", k=5, results=[], latency_ms=1.0)
        self.logger.log(query="q2", k=5, results=[], latency_ms=2.0)
        lines = self.path.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["query"], "q1")
        self.assertEqual(json.loads(lines[1])["query"], "q2")

    def test_query_truncated_at_500_chars(self) -> None:
        long_query = "x" * 1000
        self.logger.log(query=long_query, k=5, results=[], latency_ms=1.0)
        entry = json.loads(self.path.read_text(encoding="utf-8").strip())
        self.assertEqual(len(entry["query"]), 500)

    def test_top_ids_capped_at_10(self) -> None:
        results = [{"id": i} for i in range(50)]
        self.logger.log(query="q", k=50, results=results, latency_ms=1.0)
        entry = json.loads(self.path.read_text(encoding="utf-8").strip())
        self.assertEqual(len(entry["top_ids"]), 10)
        self.assertEqual(entry["n_results"], 50)

    def test_best_effort_swallows_errors(self) -> None:
        # Logger should never raise even if path is unwritable
        bad_logger = RecallAccessLogger(path=Path("/nonexistent/path/log.jsonl"))
        try:
            bad_logger.log(query="q", k=5, results=[], latency_ms=1.0)
        except Exception as e:
            self.fail(f"logger raised: {e}")

    def test_rotation_at_threshold(self) -> None:
        small_logger = RecallAccessLogger(
            path=self.path, rotate_at_bytes=200,
        )
        # Each entry is ~150 bytes; write 5 → triggers rotation after threshold
        for i in range(5):
            small_logger.log(
                query=f"query {i}", k=5, results=[], latency_ms=1.0,
            )
        # At least one rotated file should exist
        rotated = list(self.path.parent.glob(f"{self.path.stem}.jsonl.*"))
        self.assertGreater(len(rotated), 0,
                           "rotation should have produced .jsonl.<N> file")


class DefaultLoggerEnvDisableTests(unittest.TestCase):
    def test_env_disable_returns_none(self) -> None:
        # Force re-init by clearing module-level singleton
        import access_logger
        access_logger._default_logger = None
        os.environ["NM_DISABLE_ACCESS_LOG"] = "1"
        try:
            self.assertIsNone(default_logger())
        finally:
            del os.environ["NM_DISABLE_ACCESS_LOG"]
            access_logger._default_logger = None


if __name__ == "__main__":
    unittest.main()
