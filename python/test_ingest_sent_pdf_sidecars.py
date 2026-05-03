"""Tests for tools/ingest_sent_pdf_sidecars.py — NM-side tail of AE
sent-PDF sidecars.

Original surface: S-OptE packet (2026-05-03 03:23Z) — basic ingest, msg_id-only
identity, watermark, dry-run vs --live.

Patched surface: S3 packet (2026-05-03 10:41:47Z) — composite source_record_id
(msg_id:filename), filehash fallback, composite-key watermark, canonical --live
DB-guard refusal (requires evidence_ledger table OR user_version >=
EVIDENCE_LEDGER_TARGET_USER_VERSION), duplicate-group fixture proving N
sidecars sharing a msg_id produce N distinct ledger entries.

These tests are hermetic — they patch ae_workflow_helpers.record_evidence_artifact
and use synthetic sidecars in a tempdir. No substrate write, no real Gmail data.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make the tool importable as a module
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import ingest_sent_pdf_sidecars as ingest  # noqa: E402


def make_sidecar(
    sidecar_dir: Path,
    msg_id: str,
    *,
    text: str = "Sample PDF text",
    downloaded_at: str = "2026-05-02T05:13:53.556661+00:00",
    filename: str = "estimate.pdf",
    page_count: int = 1,
    extra: dict | None = None,
    json_name: str | None = None,
    write_pdf: bool = False,
    pdf_bytes: bytes = b"%PDF-1.4 stub",
) -> Path:
    payload = {
        "msg_id": msg_id,
        "thread_id": msg_id,
        "subject": "Test estimate",
        "from": "\"Angel's Electric\" <angelselectricservice@gmail.com>",
        "to": "customer@example.com",
        "date": "Wed, 15 Apr 2026 20:27:36 -0500",
        "filename": filename,
        "size_bytes": 12345,
        "text": text,
        "extraction": {"page_count": page_count, "method": "pdfplumber"},
        "dollar_total_guess": None,
        "downloaded_at": downloaded_at,
    }
    if extra:
        payload.update(extra)
    # Allow caller to override the json filename so duplicate-msg_id sidecars
    # can coexist on disk with distinct paths.
    safe_fname = (filename or "noname").replace(".pdf", "").replace("/", "_")
    name = json_name or f"{msg_id}_customer_{safe_fname}.json"
    p = sidecar_dir / name
    p.write_text(json.dumps(payload))
    if write_pdf:
        p.with_suffix(".pdf").write_bytes(pdf_bytes)
    return p


def make_guarded_db(path: Path, *, install_table: bool = True,
                    user_version: int = 2,
                    install_index: bool = True) -> None:
    """Create a SQLite DB at `path` with the v2 evidence_ledger guard installed.

    S4 hardening (2026-05-03): valid guard now requires user_version >= 2,
    the evidence_ledger table, AND idx_evidence_ledger_type_source_record.
    Default values match the canonical v2 shape from schema_upgrade.
    """
    conn = sqlite3.connect(path)
    if install_table:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence_ledger (
                evidence_id      TEXT PRIMARY KEY,
                memory_id        INTEGER,
                evidence_type    TEXT NOT NULL,
                source_system    TEXT NOT NULL,
                source_record_id TEXT NOT NULL
            )
        """)
        if install_index:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_evidence_ledger_type_source_record "
                "ON evidence_ledger (evidence_type, source_system, source_record_id)"
            )
    conn.execute(f"PRAGMA user_version = {user_version}")
    conn.commit()
    conn.close()


def make_unguarded_db(path: Path) -> None:
    """Create a SQLite DB with NO evidence_ledger and user_version=0 — the
    pre-flight guard MUST refuse --live against this file.
    """
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY)")
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()


class IngestSentPdfSidecarsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sidecar_dir = self.tmp / "sidecars"
        self.sidecar_dir.mkdir()
        self.watermark = self.tmp / "watermark.json"
        self.dryrun_dir = self.tmp / "dryrun"
        self.live_dir = self.tmp / "live"
        # Patch the module-level output dirs so tests don't pollute ~/.neural_memory
        self._dr_patcher = patch.object(ingest, "DRYRUN_DIR", self.dryrun_dir)
        self._lv_patcher = patch.object(ingest, "LIVE_DIR", self.live_dir)
        self._dr_patcher.start()
        self._lv_patcher.start()

    def tearDown(self) -> None:
        self._dr_patcher.stop()
        self._lv_patcher.stop()
        self._tmp.cleanup()

    def _run(self, *argv: str, mock_record=None) -> int:
        full_argv = [
            "ingest_sent_pdf_sidecars.py",
            "--sidecar-dir", str(self.sidecar_dir),
            "--watermark", str(self.watermark),
            *argv,
        ]
        with patch.object(sys, "argv", full_argv):
            if mock_record is None:
                # Default: fail if the tool tries to call record_evidence_artifact
                with patch.object(
                    ingest, "record_evidence_artifact",
                    side_effect=AssertionError(
                        "record_evidence_artifact must NOT be called in dry-run"
                    ),
                ):
                    return ingest.main()
            with patch.object(ingest, "record_evidence_artifact", mock_record):
                # --live also imports memory_client — stub it
                fake_mc_module = MagicMock()
                fake_mc_module.NeuralMemory = MagicMock(return_value=MagicMock())
                with patch.dict(sys.modules, {"memory_client": fake_mc_module}):
                    return ingest.main()

    # ============================================================
    # S3 PRIMARY CONTRACT — composite source_record_id
    # ============================================================

    def test_composite_source_record_id_when_filename_present(self) -> None:
        """source_record_id must be f"{msg_id}:{filename}" — the filename
        path. No filehash anywhere.
        """
        make_sidecar(
            self.sidecar_dir, "msg_a", filename="LOI-Bldg-117.pdf",
        )
        guarded = self.tmp / "guarded.db"
        make_guarded_db(guarded)
        mock = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        rc = self._run(
            "--backfill", "--live", "--db-path", str(guarded),
            mock_record=mock,
        )
        self.assertEqual(rc, 0)
        kwargs = mock.call_args.kwargs
        self.assertEqual(kwargs["source_record_id"], "msg_a:LOI-Bldg-117.pdf")
        self.assertEqual(kwargs["extra_metadata"]["msg_id"], "msg_a")
        self.assertEqual(
            kwargs["extra_metadata"]["source_record_key_strategy"], "filename",
        )

    def test_filehash_fallback_when_filename_missing(self) -> None:
        """Strategy switches to f"{msg_id}:{filehash[:16]}" only when
        filename is missing AND a sibling .pdf exists for hashing.
        """
        pdf_payload = b"%PDF-1.4 deterministic-bytes-for-hash"
        # Compute the expected hash16 manually
        import hashlib as _h
        expected_h16 = _h.sha256(pdf_payload).hexdigest()[:16]

        make_sidecar(
            self.sidecar_dir, "msg_no_fname",
            filename="",  # missing
            json_name="msg_no_fname_attachment.json",
            write_pdf=True,
            pdf_bytes=pdf_payload,
        )
        guarded = self.tmp / "guarded.db"
        make_guarded_db(guarded)
        mock = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        rc = self._run(
            "--backfill", "--live", "--db-path", str(guarded),
            mock_record=mock,
        )
        self.assertEqual(rc, 0)
        kwargs = mock.call_args.kwargs
        self.assertEqual(
            kwargs["source_record_id"], f"msg_no_fname:{expected_h16}",
        )
        self.assertEqual(
            kwargs["extra_metadata"]["source_record_key_strategy"], "filehash",
        )

    def test_filehash_fallback_fails_if_pdf_missing(self) -> None:
        """If filename is missing AND no sibling .pdf exists for hashing,
        the per-sidecar build path must error (not silently fall back to
        bare msg_id). Tool exits 3 (errors > 0) but doesn't crash.
        """
        make_sidecar(
            self.sidecar_dir, "msg_no_fname_no_pdf",
            filename="",
            json_name="msg_no_fname_no_pdf.json",
            write_pdf=False,
        )
        rc = self._run("--backfill")
        self.assertEqual(rc, 3)
        outs = list(self.dryrun_dir.glob("sent-pdf-*.dry.jsonl"))
        rows = [json.loads(l) for l in outs[0].read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 1)
        self.assertIn("error", rows[0])

    def test_duplicate_msg_id_distinct_filenames_yield_distinct_keys(self) -> None:
        """Critical S3 contract: 2 sidecars sharing msg_id but with
        different filenames must produce 2 DISTINCT ledger entries (different
        composite source_record_id), NOT 1 collapsed entry. This is the
        bug class the patch fixes.
        """
        # Mirror real corpus: msg_id 19707875871798bb has 5 different LOI PDFs.
        make_sidecar(
            self.sidecar_dir, "msg_shared",
            filename="LOI-Bldg-117.pdf",
            json_name="msg_shared_117.json",
        )
        make_sidecar(
            self.sidecar_dir, "msg_shared",
            filename="LOI-Lot-299.pdf",
            json_name="msg_shared_299.json",
        )

        guarded = self.tmp / "guarded.db"
        make_guarded_db(guarded)
        mock = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        rc = self._run(
            "--backfill", "--live", "--db-path", str(guarded),
            mock_record=mock,
        )
        self.assertEqual(rc, 0)
        # Two distinct calls, two distinct source_record_ids
        self.assertEqual(mock.call_count, 2)
        ids = sorted(
            call.kwargs["source_record_id"] for call in mock.call_args_list
        )
        self.assertEqual(
            ids,
            ["msg_shared:LOI-Bldg-117.pdf", "msg_shared:LOI-Lot-299.pdf"],
        )
        # Underlying msg_id preserved in metadata for both
        for call in mock.call_args_list:
            self.assertEqual(call.kwargs["extra_metadata"]["msg_id"], "msg_shared")

    def test_duplicate_msg_id_distinct_evidence_ids(self) -> None:
        """Pre-computed evidence_id (sha256 over the composite triple) must
        differ when source_record_id differs — guarantees the upsert path
        keys two distinct ledger rows.
        """
        from ae_workflow_helpers import _compute_evidence_id
        a = _compute_evidence_id("sent_pdf", "sent_estimate_pdf_miner",
                                 "msg_shared:LOI-Bldg-117.pdf")
        b = _compute_evidence_id("sent_pdf", "sent_estimate_pdf_miner",
                                 "msg_shared:LOI-Lot-299.pdf")
        self.assertNotEqual(a, b)

    # ============================================================
    # S3 PRIMARY CONTRACT — composite-key watermark
    # ============================================================

    def test_watermark_uses_composite_keys(self) -> None:
        """After --live, watermark file must list composite keys, not bare
        msg_ids. schema_version=2.
        """
        make_sidecar(
            self.sidecar_dir, "msg_a", filename="a.pdf",
            json_name="msg_a_a.json",
        )
        make_sidecar(
            self.sidecar_dir, "msg_a", filename="b.pdf",
            json_name="msg_a_b.json",
        )
        guarded = self.tmp / "guarded.db"
        make_guarded_db(guarded)
        mock = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        rc = self._run(
            "--backfill", "--live", "--db-path", str(guarded),
            mock_record=mock,
        )
        self.assertEqual(rc, 0)
        wm = json.loads(self.watermark.read_text())
        self.assertEqual(wm["schema_version"], 2)
        self.assertEqual(
            set(wm["processed_keys"]),
            {"msg_a:a.pdf", "msg_a:b.pdf"},
        )
        self.assertNotIn("processed_msg_ids", wm)

    def test_idempotent_rerun_via_composite_watermark(self) -> None:
        """Second --live run (no --backfill) over the same sidecars skips
        every entry via watermark — record_evidence_artifact called 0 times.
        """
        make_sidecar(
            self.sidecar_dir, "msg_a", filename="a.pdf",
            json_name="msg_a_a.json",
        )
        make_sidecar(
            self.sidecar_dir, "msg_a", filename="b.pdf",
            json_name="msg_a_b.json",
        )
        guarded = self.tmp / "guarded.db"
        make_guarded_db(guarded)

        # Round 1: backfill + live → fills watermark
        mock1 = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        rc1 = self._run(
            "--backfill", "--live", "--db-path", str(guarded),
            mock_record=mock1,
        )
        self.assertEqual(rc1, 0)
        self.assertEqual(mock1.call_count, 2)

        # Round 2: no --backfill, same sidecars → watermark skips all
        mock2 = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        rc2 = self._run(
            "--live", "--db-path", str(guarded),
            mock_record=mock2,
        )
        self.assertEqual(rc2, 0)
        self.assertEqual(mock2.call_count, 0)

    def test_legacy_watermark_schema_v1_migrated(self) -> None:
        """Legacy schema_version=1 watermark (bare msg_ids) must be
        recognised + discarded; the next save writes schema_version=2.
        """
        # Seed legacy v1 watermark
        self.watermark.parent.mkdir(parents=True, exist_ok=True)
        self.watermark.write_text(json.dumps({
            "processed_msg_ids": ["msg_a"],
            "last_run_ts": 1700000000,
        }))

        make_sidecar(
            self.sidecar_dir, "msg_a", filename="a.pdf",
            json_name="msg_a_a.json",
        )
        guarded = self.tmp / "guarded.db"
        make_guarded_db(guarded)

        mock = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        # NO --backfill: tool consults watermark; legacy data discarded so
        # msg_a:a.pdf is NOT skipped.
        rc = self._run(
            "--live", "--db-path", str(guarded),
            mock_record=mock,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(mock.call_count, 1)
        wm = json.loads(self.watermark.read_text())
        self.assertEqual(wm["schema_version"], 2)
        self.assertIn("processed_keys", wm)
        self.assertEqual(set(wm["processed_keys"]), {"msg_a:a.pdf"})

    # ============================================================
    # S3 PRIMARY CONTRACT — canonical --live DB guard refusal
    # ============================================================

    def test_canonical_live_refused_without_evidence_ledger(self) -> None:
        """--live must REFUSE (exit 5) when the target DB lacks the
        evidence_ledger table AND user_version is below the S2 target.
        """
        make_sidecar(self.sidecar_dir, "msg_a", filename="a.pdf")
        unguarded = self.tmp / "unguarded.db"
        make_unguarded_db(unguarded)

        # Use mock_record so the unhappy path isn't masked by import side-effects.
        mock = MagicMock()
        rc = self._run(
            "--backfill", "--live", "--db-path", str(unguarded),
            mock_record=mock,
        )
        self.assertEqual(rc, 5, "tool must refuse --live without DB guard")
        # record_evidence_artifact must NOT have been called
        self.assertEqual(mock.call_count, 0)

    def test_canonical_live_refused_when_db_missing(self) -> None:
        """Non-existent DB also fails the guard (no silent create)."""
        make_sidecar(self.sidecar_dir, "msg_a", filename="a.pdf")
        missing = self.tmp / "does-not-exist.db"
        mock = MagicMock()
        rc = self._run(
            "--backfill", "--live", "--db-path", str(missing),
            mock_record=mock,
        )
        self.assertEqual(rc, 5)
        self.assertEqual(mock.call_count, 0)

    def test_canonical_default_db_refused_when_unguarded(self) -> None:
        """When --db-path NOT supplied, the default canonical DB is
        probed. If unguarded → refuse. We patch CANONICAL_DB_PATH to a
        tempdir-controlled unguarded file.
        """
        make_sidecar(self.sidecar_dir, "msg_a", filename="a.pdf")
        unguarded = self.tmp / "fake-canonical.db"
        make_unguarded_db(unguarded)
        with patch.object(ingest, "CANONICAL_DB_PATH", unguarded):
            mock = MagicMock()
            rc = self._run("--backfill", "--live", mock_record=mock)
        self.assertEqual(rc, 5)
        self.assertEqual(mock.call_count, 0)

    def test_copy_db_live_allowed_when_guard_installed(self) -> None:
        """The 'copy proof' path: caller creates a copy DB AND installs
        the v2 evidence_ledger guard before --live. Tool must allow this.
        """
        make_sidecar(self.sidecar_dir, "msg_a", filename="a.pdf")
        copy_db = self.tmp / "copy.db"
        make_guarded_db(copy_db, install_table=True, user_version=2)

        mock = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        rc = self._run(
            "--backfill", "--live", "--db-path", str(copy_db),
            mock_record=mock,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(mock.call_count, 1)

    def test_guard_fails_via_user_version_alone(self) -> None:
        """S4 hardening: user_version-only (no table) must FAIL the guard.
        v2 guard requires user_version >= 2 AND the table AND the index.
        """
        make_sidecar(self.sidecar_dir, "msg_a", filename="a.pdf")
        db = self.tmp / "uv-only.db"
        make_guarded_db(db, install_table=False, user_version=2)
        mock = MagicMock()
        rc = self._run(
            "--backfill", "--live", "--db-path", str(db),
            mock_record=mock,
        )
        self.assertEqual(rc, 5, "uv-only DB must be refused by v2 guard")
        self.assertEqual(mock.call_count, 0)

    def test_guard_passes_v2_full_shape(self) -> None:
        """S4 hardening: user_version=2 + table + index must pass the guard."""
        make_sidecar(self.sidecar_dir, "msg_a", filename="a.pdf")
        db = self.tmp / "v2-full.db"
        make_guarded_db(db, install_table=True, user_version=2, install_index=True)
        mock = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        rc = self._run(
            "--backfill", "--live", "--db-path", str(db),
            mock_record=mock,
        )
        self.assertEqual(rc, 0, "v2 full-shape DB must be allowed by guard")

    def test_dry_run_never_probes_db_guard(self) -> None:
        """Dry-run is purely informational; the DB guard check must NOT
        run in dry-run, even when --db-path points at an unguarded DB.
        """
        make_sidecar(self.sidecar_dir, "msg_a", filename="a.pdf")
        unguarded = self.tmp / "unguarded.db"
        make_unguarded_db(unguarded)
        # Default mock_record (None) asserts record_evidence_artifact is NOT called
        rc = self._run("--backfill", "--db-path", str(unguarded))
        self.assertEqual(rc, 0)

    # ============================================================
    # PRESERVED CONTRACTS FROM S-OPTE
    # ============================================================

    def test_dry_run_default_no_substrate_write(self) -> None:
        make_sidecar(
            self.sidecar_dir, "msg_a", filename="a.pdf",
            json_name="msg_a_a.json",
        )
        make_sidecar(
            self.sidecar_dir, "msg_b", filename="b.pdf",
            json_name="msg_b_b.json",
        )
        rc = self._run("--backfill")
        self.assertEqual(rc, 0)
        outs = list(self.dryrun_dir.glob("sent-pdf-*.dry.jsonl"))
        self.assertEqual(len(outs), 1)
        rows = [json.loads(l) for l in outs[0].read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertTrue(r["dry_run"])
            self.assertIsNone(r["memory_id"])
            self.assertIsNone(r["inserted"])
            self.assertIn("evidence_id", r)
            self.assertIn("composite_key", r)

    def test_live_mode_calls_record_evidence_artifact(self) -> None:
        make_sidecar(
            self.sidecar_dir, "msg_a", text="Estimate body A",
            filename="a.pdf",
        )
        guarded = self.tmp / "guarded.db"
        make_guarded_db(guarded)
        mock = MagicMock(return_value={
            "memory_id": 42, "evidence_id": "abc123", "inserted": True,
        })
        rc = self._run(
            "--backfill", "--live", "--db-path", str(guarded),
            mock_record=mock,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(mock.call_count, 1)
        _, kwargs = mock.call_args
        self.assertEqual(kwargs["evidence_type"], "sent_pdf")
        self.assertEqual(kwargs["source_system"], "sent_estimate_pdf_miner")
        self.assertEqual(kwargs["source_record_id"], "msg_a:a.pdf")
        self.assertEqual(kwargs["content"], "Estimate body A")
        self.assertEqual(kwargs["capability_id"], "ITEM-SENT-PDF")
        self.assertEqual(kwargs["privacy_class"], "financial")
        self.assertIn("subject", kwargs["extra_metadata"])
        self.assertIn("filename", kwargs["extra_metadata"])
        self.assertIn("page_count", kwargs["extra_metadata"])
        self.assertEqual(kwargs["extra_metadata"]["extraction_method"], "pdfplumber")
        self.assertIsInstance(kwargs["valid_from"], float)
        self.assertGreater(kwargs["valid_from"], 1_700_000_000)

    def test_backfill_ignores_watermark(self) -> None:
        make_sidecar(
            self.sidecar_dir, "msg_a", filename="a.pdf",
            json_name="msg_a_a.json",
        )
        make_sidecar(
            self.sidecar_dir, "msg_b", filename="b.pdf",
            json_name="msg_b_b.json",
        )
        # Pre-seed a v2 watermark with both composite keys
        self.watermark.parent.mkdir(parents=True, exist_ok=True)
        self.watermark.write_text(json.dumps({
            "schema_version": 2,
            "processed_keys": ["msg_a:a.pdf", "msg_b:b.pdf"],
            "last_run_ts": 1700000000,
        }))
        rc = self._run("--backfill")
        self.assertEqual(rc, 0)
        outs = list(self.dryrun_dir.glob("sent-pdf-*.dry.jsonl"))
        rows = [json.loads(l) for l in outs[0].read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertTrue(r.get("dry_run"))
            self.assertNotIn("skipped_watermark", r)

    def test_handles_malformed_sidecar(self) -> None:
        make_sidecar(self.sidecar_dir, "msg_good", filename="g.pdf")
        (self.sidecar_dir / "bad_json.json").write_text("{not valid json")
        (self.sidecar_dir / "missing_field.json").write_text(json.dumps({
            "thread_id": "x", "text": "y", "downloaded_at": "2026-05-02T00:00:00+00:00",
        }))
        rc = self._run("--backfill")
        self.assertEqual(rc, 3)
        outs = list(self.dryrun_dir.glob("sent-pdf-*.dry.jsonl"))
        rows = [json.loads(l) for l in outs[0].read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 3)
        ok = [r for r in rows if "error" not in r]
        bad = [r for r in rows if "error" in r]
        self.assertEqual(len(ok), 1)
        self.assertEqual(len(bad), 2)
        for r in bad:
            self.assertIn("sidecar_path", r)
            self.assertIsInstance(r["error"], str)

    def test_metadata_shape_matches_spec(self) -> None:
        make_sidecar(
            self.sidecar_dir, "msg_meta",
            text="body",
            filename="meta.pdf",
            extra={"dollar_total_guess": 1234.56},
        )
        guarded = self.tmp / "guarded.db"
        make_guarded_db(guarded)
        mock = MagicMock(return_value={
            "memory_id": 1, "evidence_id": "e", "inserted": True,
        })
        rc = self._run(
            "--backfill", "--live", "--db-path", str(guarded),
            mock_record=mock,
        )
        self.assertEqual(rc, 0)
        kwargs = mock.call_args.kwargs
        meta = kwargs["extra_metadata"]
        for key in ("thread_id", "subject", "from", "to", "date",
                    "filename", "dollar_total_guess",
                    # S3 additions
                    "msg_id", "source_record_key_strategy"):
            self.assertIn(key, meta, f"metadata missing required key {key!r}")
        self.assertIn("page_count", meta)
        self.assertEqual(meta["dollar_total_guess"], 1234.56)
        self.assertEqual(meta["msg_id"], "msg_meta")
        self.assertEqual(meta["source_record_key_strategy"], "filename")
        self.assertEqual(kwargs["evidence_type"], "sent_pdf")
        self.assertEqual(kwargs["capability_id"], "ITEM-SENT-PDF")

    # ============================================================
    # AUX
    # ============================================================

    def test_parse_downloaded_at_iso_string(self) -> None:
        ts = ingest._parse_downloaded_at("2026-05-02T05:13:53.556661+00:00")
        self.assertIsInstance(ts, float)
        self.assertGreater(ts, 1700000000)

    def test_parse_downloaded_at_epoch_float_passthrough(self) -> None:
        ts = ingest._parse_downloaded_at(1714627200.5)
        self.assertEqual(ts, 1714627200.5)

    def test_empty_sidecar_dir_no_op(self) -> None:
        rc = self._run("--backfill")
        self.assertEqual(rc, 0)

    def test_check_db_guard_helper_v2_full_shape_passes(self) -> None:
        """S4: v2 = user_version >= 2 + table + index must pass."""
        db = self.tmp / "g1.db"
        make_guarded_db(db, install_table=True, user_version=2, install_index=True)
        ok, reason = ingest._check_db_guard(db)
        self.assertTrue(ok, reason)
        self.assertIn("v2 guard OK", reason)

    def test_check_db_guard_helper_table_only_fails(self) -> None:
        """S4: table present but user_version < 2 must fail."""
        db = self.tmp / "g1b.db"
        make_guarded_db(db, install_table=True, user_version=0, install_index=False)
        ok, reason = ingest._check_db_guard(db)
        self.assertFalse(ok)
        self.assertIn("need >=2", reason)

    def test_check_db_guard_helper_user_version_only_v1_fails(self) -> None:
        """S4: user_version=1, no table must fail (uv < 2)."""
        db = self.tmp / "g2.db"
        make_guarded_db(db, install_table=False, user_version=1)
        ok, reason = ingest._check_db_guard(db)
        self.assertFalse(ok)
        self.assertIn("need >=2", reason)

    def test_check_db_guard_helper_v2_no_index_fails(self) -> None:
        """S4: user_version=2, table present, but index missing must fail."""
        db = self.tmp / "g2b.db"
        make_guarded_db(db, install_table=True, user_version=2, install_index=False)
        ok, reason = ingest._check_db_guard(db)
        self.assertFalse(ok)
        self.assertIn("index missing", reason)

    def test_check_db_guard_helper_unguarded(self) -> None:
        db = self.tmp / "g3.db"
        make_unguarded_db(db)
        ok, reason = ingest._check_db_guard(db)
        self.assertFalse(ok)
        self.assertIn("need >=2", reason)

    def test_check_db_guard_helper_missing_file(self) -> None:
        db = self.tmp / "nope.db"
        ok, reason = ingest._check_db_guard(db)
        self.assertFalse(ok)
        self.assertIn("does not exist", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
