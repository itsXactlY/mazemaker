"""Smoke tests for the nm CLI at tools/nm.py.

Each test invokes the CLI as a subprocess against a tmp DB, asserting
expected output shape. Subprocess isolation prevents test pollution and
exercises the actual command-line surface.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_NM = str(_ROOT / "tools" / "nm.py")


def _run(*args: str, db: str | None = None,
         input_: str | None = None) -> tuple[int, str, str]:
    cmd = [sys.executable, _NM]
    if db:
        cmd += ["--db", db]
    cmd += list(args)
    res = subprocess.run(cmd, capture_output=True, text=True, input=input_,
                         timeout=60)
    return res.returncode, res.stdout, res.stderr


class NmCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "memory.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_count_on_empty_db(self) -> None:
        rc, out, err = _run("count", db=self.db)
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["memories"], 0)
        self.assertEqual(data["entities"], 0)

    def test_remember_then_count(self) -> None:
        rc, _, _ = _run("remember", "First memory.",
                        "--kind=experience", db=self.db)
        self.assertEqual(rc, 0)
        rc, out, _ = _run("count", db=self.db)
        data = json.loads(out)
        self.assertEqual(data["memories"], 1)

    def test_remember_recall_roundtrip(self) -> None:
        _run("remember", "Lennar lot 27 needs panel labels.",
             "--kind=experience", "--source=dashboard", db=self.db)
        rc, out, _ = _run("recall", "Lennar panel", "--k=5",
                          "--format=json", db=self.db)
        self.assertEqual(rc, 0)
        results = json.loads(out)
        self.assertGreater(len(results), 0)

    def test_sparse_search(self) -> None:
        _run("remember", "GFCI exterior receptacles required.", db=self.db)
        rc, out, _ = _run("sparse", "GFCI exterior",
                          "--format=json", db=self.db)
        self.assertEqual(rc, 0)
        results = json.loads(out)
        self.assertGreaterEqual(len(results), 1)

    def test_entities_top(self) -> None:
        _run("remember", "Sarah from Lennar called.", db=self.db)
        _run("remember", "Lennar lot 27 update.", db=self.db)
        rc, out, _ = _run("entities", "--top=10",
                          "--format=json", db=self.db)
        self.assertEqual(rc, 0)
        ents = json.loads(out)
        labels = {e["label"] for e in ents}
        self.assertIn("Lennar", labels)

    def test_audit_runs_without_error(self) -> None:
        _run("remember", "First.", db=self.db)
        rc, out, _ = _run("audit", db=self.db)
        self.assertEqual(rc, 0)
        # Audit output is human-readable; just check key sections present
        self.assertIn("Memories by kind", out)
        self.assertIn("Phase 7 schema completeness", out)

    def test_explain_returns_features(self) -> None:
        _run("remember", "Sarah is the Lennar contact this week.",
             "--kind=experience", db=self.db)
        rc, out, _ = _run("explain", "Lennar contact", "--k=2",
                          "--format=json", db=self.db)
        self.assertEqual(rc, 0)
        results = json.loads(out)
        self.assertGreater(len(results), 0)
        self.assertIn("explanation", results[0])
        self.assertIn("salience", results[0]["explanation"]["features"])

    def test_forget_background_visibility(self) -> None:
        rc, out, _ = _run("remember", "Temp note.", db=self.db)
        # Get the id from count then assume it's 1 on fresh DB
        rc, _, _ = _run("forget", "1", "--mode=background", db=self.db)
        self.assertEqual(rc, 0)

    def test_memify_runs_without_error(self) -> None:
        _run("remember", "duplicate content here", db=self.db)
        _run("remember", "duplicate content here", db=self.db)
        rc, out, _ = _run("memify", "--decay=0.5", db=self.db)
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertIn("duplicates_downweighted", data)

    def test_help_is_not_an_error(self) -> None:
        rc, _, _ = _run("--help")
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
