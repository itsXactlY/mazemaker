"""Tests for the self-portrait cron plist + wrapper script.

Per packet S-PORTRAIT-3 acceptance criteria:
- plist XML is well-formed (plutil -lint)
- plist has FOUR StartCalendarInterval dicts (every 6h)
- those dicts fire at 06:00, 12:00, 18:00, 00:00
- wrapper script invokes self_portrait_cycle.py
- wrapper script iterates the v0 agent set (claude-code, hermes, codex)
- wrapper uses `set -uo pipefail`, NOT `set -e` (per acceptance #2 — we want
  to continue past per-agent failures and report at the end)
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PLIST_PATH = REPO_ROOT / "tools" / "launchd" / "com.ae.neural-self-portrait.plist"
WRAPPER_PATH = REPO_ROOT / "tools" / "self_portrait_cron.sh"


class TestSelfPortraitCronPlist(unittest.TestCase):
    """Plist + wrapper script structural sanity tests."""

    # ------------------------------------------------------------------
    # plist tests
    # ------------------------------------------------------------------

    def test_plist_file_exists(self):
        self.assertTrue(
            PLIST_PATH.exists(),
            f"plist not found at {PLIST_PATH}",
        )

    def test_plist_xml_is_valid(self):
        """plutil -lint must report OK on the plist."""
        plutil = shutil.which("plutil")
        if not plutil:
            self.skipTest("plutil not available (non-macOS environment)")
        result = subprocess.run(
            [plutil, "-lint", str(PLIST_PATH)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"plutil -lint failed: stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertIn("OK", result.stdout)

    def test_plist_has_four_calendar_intervals(self):
        """StartCalendarInterval must be an ARRAY of 4 dicts (every 6h)."""
        with open(PLIST_PATH, "rb") as fh:
            data = plistlib.load(fh)
        self.assertIn("StartCalendarInterval", data)
        intervals = data["StartCalendarInterval"]
        self.assertIsInstance(
            intervals,
            list,
            "StartCalendarInterval must be an array (list of dicts)",
        )
        self.assertEqual(
            len(intervals),
            4,
            f"expected 4 calendar intervals (every 6h), got {len(intervals)}",
        )
        for entry in intervals:
            self.assertIsInstance(entry, dict)

    def test_plist_calendar_intervals_are_06_12_18_00(self):
        """Hours must be exactly {0, 6, 12, 18}, all at minute 0."""
        with open(PLIST_PATH, "rb") as fh:
            data = plistlib.load(fh)
        intervals = data["StartCalendarInterval"]
        hours = sorted(entry["Hour"] for entry in intervals)
        self.assertEqual(
            hours,
            [0, 6, 12, 18],
            f"expected hours [0, 6, 12, 18], got {hours}",
        )
        for entry in intervals:
            self.assertEqual(
                entry.get("Minute", 0),
                0,
                f"all intervals must fire on the hour (minute=0), got {entry}",
            )

    def test_plist_keepalive_only_on_failure(self):
        """KeepAlive must be a dict with SuccessfulExit=False (only respawn on failure)."""
        with open(PLIST_PATH, "rb") as fh:
            data = plistlib.load(fh)
        ka = data.get("KeepAlive")
        self.assertIsInstance(
            ka,
            dict,
            "KeepAlive must be a dict (SuccessfulExit form), not bool",
        )
        self.assertIs(
            ka.get("SuccessfulExit"),
            False,
            f"KeepAlive.SuccessfulExit must be false, got {ka}",
        )

    def test_plist_run_at_load_true(self):
        """RunAtLoad=true so the job fires immediately on first load (testable)."""
        with open(PLIST_PATH, "rb") as fh:
            data = plistlib.load(fh)
        self.assertIs(data.get("RunAtLoad"), True)

    def test_plist_label_matches_filename(self):
        with open(PLIST_PATH, "rb") as fh:
            data = plistlib.load(fh)
        self.assertEqual(data.get("Label"), "com.ae.neural-self-portrait")

    def test_plist_program_arguments_invokes_wrapper(self):
        with open(PLIST_PATH, "rb") as fh:
            data = plistlib.load(fh)
        args = data.get("ProgramArguments", [])
        self.assertEqual(args[0], "/bin/bash")
        self.assertTrue(
            any("self_portrait_cron.sh" in a for a in args),
            f"plist must invoke self_portrait_cron.sh, got args={args}",
        )

    def test_plist_log_paths_under_library_logs_ae(self):
        with open(PLIST_PATH, "rb") as fh:
            data = plistlib.load(fh)
        self.assertEqual(
            data.get("StandardOutPath"),
            "/Users/tito/Library/Logs/ae/neural-self-portrait.stdout.log",
        )
        self.assertEqual(
            data.get("StandardErrorPath"),
            "/Users/tito/Library/Logs/ae/neural-self-portrait.stderr.log",
        )

    # ------------------------------------------------------------------
    # wrapper script tests
    # ------------------------------------------------------------------

    def test_wrapper_script_exists(self):
        self.assertTrue(
            WRAPPER_PATH.exists(),
            f"wrapper not found at {WRAPPER_PATH}",
        )

    def test_wrapper_script_invokes_self_portrait_cycle(self):
        """The wrapper must shell out to tools/self_portrait_cycle.py."""
        text = WRAPPER_PATH.read_text()
        self.assertIn("self_portrait_cycle.py", text)
        self.assertIn("python3", text)
        self.assertIn("--mode scaffold", text)

    def test_wrapper_script_iterates_v0_agent_set(self):
        """All three v0 agents (claude-code, hermes, codex) must appear."""
        text = WRAPPER_PATH.read_text()
        self.assertIn("claude-code", text)
        self.assertIn("hermes", text)
        self.assertIn("codex", text)

    def test_wrapper_uses_set_uo_pipefail_not_e(self):
        """`set -uo pipefail` (NOT `set -e`) — we want to continue past per-agent failures.

        This is the explicit acceptance-criteria #2 from the packet: with `set -e`,
        the first agent failure aborts the loop and the other agents never get
        their cycle. We deliberately drop the `-e` flag.
        """
        text = WRAPPER_PATH.read_text()
        self.assertIn(
            "set -uo pipefail",
            text,
            "wrapper must declare `set -uo pipefail`",
        )
        # Make sure no `set -e` (or `set -euo pipefail`) sneaked in.
        # We allow `set -uo pipefail` but reject any line that starts with
        # `set -` and contains a bare `e` flag (without the `u` first).
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("set -") and "pipefail" in stripped:
                # The flag block is between `set -` and the next space.
                flags = stripped.split()[1].lstrip("-")
                self.assertNotIn(
                    "e",
                    flags,
                    f"wrapper must NOT use `-e` in `set -<flags>`; got line: {stripped!r}",
                )

    def test_wrapper_uses_bash_shebang(self):
        """Per acceptance criteria #2 — bash, not zsh (more portable for launchd)."""
        first_line = WRAPPER_PATH.read_text().splitlines()[0]
        self.assertTrue(
            first_line.startswith("#!") and "bash" in first_line,
            f"wrapper must start with a bash shebang; got {first_line!r}",
        )
        self.assertNotIn("zsh", first_line)

    def test_wrapper_script_syntax_valid(self):
        """`bash -n` (parse-only) must succeed on the wrapper."""
        bash = shutil.which("bash") or "/bin/bash"
        result = subprocess.run(
            [bash, "-n", str(WRAPPER_PATH)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"bash -n failed: stderr={result.stderr!r}",
        )


if __name__ == "__main__":
    unittest.main()
