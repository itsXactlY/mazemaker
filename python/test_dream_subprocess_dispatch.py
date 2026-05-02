"""Regression coverage for DreamEngine subprocess dispatch.

D6 scheduler uses an external launchd wrapper, but the library subprocess path is
also part of the dream-cycle contract. It must reopen the SQLite DB with a real
DreamBackend, not accidentally pass NeuralMemory as the backend.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream_engine import DreamEngine  # noqa: E402
from memory_client import NeuralMemory  # noqa: E402


class DreamSubprocessDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "memory.db")
        self.mem = NeuralMemory(
            db_path=self.db_path,
            embedding_backend="hash",
            use_cpp=False,
            use_hnsw=False,
        )
        for text in [
            "Valiendo should preserve no-send boundaries on WhatsApp crew messages.",
            "Valiendo should use exact approval gates before QBO writes.",
            "Valiendo should checkpoint durable AE state after meaningful changes.",
        ]:
            self.mem.remember(text, detect_conflicts=False, kind="experience")
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

    def test_subprocess_dispatch_runs_cycle_without_backend_error(self) -> None:
        job = self.engine.dream_now(dispatch="subprocess")
        deadline = time.time() + 20
        status = {"status": "queued"}
        while time.time() < deadline:
            status = self.engine.dream_status(job["job_id"])
            if status.get("status") in {"complete", "error"}:
                break
            time.sleep(0.2)

        self.assertEqual(status.get("status"), "complete", status)
        # Reap the child process so unittest does not emit ResourceWarning even
        # when the status file is written a few milliseconds before process exit.
        reap_deadline = time.time() + 5
        while time.time() < reap_deadline:
            try:
                waited_pid, _status = os.waitpid(job["pid"], os.WNOHANG)
            except ChildProcessError:
                waited_pid = job["pid"]
            if waited_pid == job["pid"]:
                break
            time.sleep(0.1)
        result = status.get("result") or {}
        self.assertNotIn("error", result, result)
        self.assertIn("self_image", result, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
