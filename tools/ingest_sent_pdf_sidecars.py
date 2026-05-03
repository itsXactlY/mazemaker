#!/usr/bin/env python3
"""ingest_sent_pdf_sidecars.py — tail AE sent-PDF sidecar JSONs into NM substrate.

Per Sonnet packet S-OptE (NM-side tail of AE sent-PDF sidecars, 2026-05-03):

  AE-builder lane already runs `sent_estimate_pdf_miner.py` which writes typed
  sidecar JSONs to /Users/tito/.../LangGraph/data/sent-estimates-pdfs/. NM just
  tails them. Zero AE-side patch required.

  Tito approved no-privacy-gating ("idc about privacy. i'm only user dude.")
  — all sent-PDF sidecars are ingested by default.

Sidecar shape (verified from sent_estimate_pdf_miner lines 180-194):
  {msg_id, thread_id, subject, from, to, date, filename, size_bytes, text,
   extraction, dollar_total_guess, downloaded_at}

NM mapping:
  evidence_type    = "sent_pdf"           (canonical EVIDENCE_TYPES entry;
                                           packet specified "sent_estimate_pdf"
                                           but that's not in the enum — using
                                           the canonical "sent_pdf" instead)
  source_system    = "sent_estimate_pdf_miner"
  source_record_id = f"{msg_id}:{filename}" composite (S3 packet 2026-05-03).
                     Gmail msg_id alone collides — a single email can carry
                     N attachments (verified: 11 dup groups / 35 sidecars on
                     the 63-sidecar corpus). Falls back to
                     f"{msg_id}:{filehash[:16]}" only when filename missing
                     (currently 0 of 63).
  source_path      = absolute sidecar path
  content          = sidecar.text
  valid_from       = parsed(sidecar.downloaded_at) → epoch float
                     (packet text said "epoch float" but the field is actually
                     an ISO-8601 string; we parse it)
  metadata         = {thread_id, subject, from, to, date, filename,
                      dollar_total_guess, size_bytes, page_count,
                      extraction_method, capability_id: "ITEM-SENT-PDF",
                      msg_id, source_record_key_strategy}

DEFAULT MODE = dry-run. NO substrate write. Validates sidecars and emits
typed records to ~/.neural_memory/ingest-dryruns/sent-pdf-{ts}.jsonl for
inspection. --live is explicit opt-in and writes to canonical substrate
via record_evidence_artifact (which is replay-safe via evidence_id upsert).

Watermark format (default ~/.neural_memory/state/sent-pdf-watermark.json):
  {"processed_keys": [...], "last_run_ts": <epoch_float>,
   "schema_version": 2}
  Tracks the SET of composite (msg_id:filename) keys. schema_version=1
  files (legacy `processed_msg_ids`) are auto-migrated forward on read —
  legacy ids skip nothing because they no longer match the composite key
  space, so on the first run after upgrade every sidecar will re-process
  and re-skip via record_evidence_artifact's evidence_id upsert path
  (idempotent). --backfill ignores the watermark entirely.

Usage:
    # Dry-run smoke against all 47 historical sidecars (READ-ONLY):
    python3 tools/ingest_sent_pdf_sidecars.py --backfill

    # Live ingest of all sidecars (writes canonical substrate):
    python3 tools/ingest_sent_pdf_sidecars.py --backfill --live

    # Tail-only (incremental, watermarked):
    python3 tools/ingest_sent_pdf_sidecars.py --live
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Import canonical helpers from neural-memory python/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from ae_workflow_helpers import (  # noqa: E402
    EVIDENCE_TYPES,
    _compute_evidence_id,
    record_evidence_artifact,
)

DEFAULT_SIDECAR_DIR = Path(
    "/Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph/"
    "data/sent-estimates-pdfs/"
)
DEFAULT_WATERMARK = Path.home() / ".neural_memory" / "state" / "sent-pdf-watermark.json"
DRYRUN_DIR = Path.home() / ".neural_memory" / "ingest-dryruns"
LIVE_DIR = Path.home() / ".neural_memory" / "ingest-live"

# Canonical mapping constants
EVIDENCE_TYPE = "sent_pdf"          # canon, see ae_workflow_helpers.EVIDENCE_TYPES
SOURCE_SYSTEM = "sent_estimate_pdf_miner"
CAPABILITY_ID = "ITEM-SENT-PDF"
PRIVACY_CLASS = "financial"         # estimates carry pricing; matches record_estimate_evidence default
CONFIDENCE = 0.95

REQUIRED_FIELDS = {"msg_id", "text", "downloaded_at"}

# Canonical substrate path. Pre-flight refusal check requires BOTH
# PRAGMA user_version >= v2 target AND the v2 composite index before
# allowing --live against this DB (S4 hardening 2026-05-03).
CANONICAL_DB_PATH = (Path.home() / ".neural_memory" / "memory.db").resolve()
EVIDENCE_LEDGER_TABLE = "evidence_ledger"
EVIDENCE_LEDGER_TARGET_USER_VERSION = 2   # mirrors S2b schema_upgrade target (v2)
EVIDENCE_LEDGER_REQUIRED_INDEX = "idx_evidence_ledger_type_source_record"

WATERMARK_SCHEMA_VERSION = 2   # bumped from 1 (msg_id) → 2 (composite keys)


def _parse_downloaded_at(value: object) -> float:
    """Sidecar `downloaded_at` is ISO-8601 (verified-now). Accept epoch float
    too in case the upstream miner ever changes — be liberal in what we read.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Python 3.11+ supports `Z` and `+HH:MM` natively
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            # Final fallback: try date.parsefmt for RFC-2822-ish strings
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(value).timestamp()
    raise ValueError(f"downloaded_at has unsupported type {type(value).__name__}: {value!r}")


def load_watermark(path: Path) -> set[str]:
    """Load composite-key watermark. Auto-migrates schema_version=1 (legacy
    `processed_msg_ids` listing bare msg_ids) into an empty set — legacy ids
    cannot match composite keys so they would skip nothing anyway. The first
    write after migration replaces the file with schema_version=2.
    """
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
        # Schema 2: composite keys
        keys = data.get("processed_keys")
        if isinstance(keys, list):
            return set(str(x) for x in keys)
        # Schema 1 legacy: bare msg_ids — discard, they can't dedup composites
        if "processed_msg_ids" in data:
            print(
                "INFO: legacy schema_version=1 watermark detected (bare msg_id "
                "keys); migrating to composite-key (schema_version=2). First "
                "post-migration run will re-process every sidecar; "
                "record_evidence_artifact dedup keeps it idempotent.",
                file=sys.stderr,
            )
            return set()
        return set()
    except Exception as e:
        print(f"WARNING: watermark file unreadable ({e}); treating as empty", file=sys.stderr)
        return set()


def save_watermark(path: Path, processed_keys: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": WATERMARK_SCHEMA_VERSION,
        "processed_keys": sorted(processed_keys),
        "last_run_ts": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)  # atomic


def _compute_pdf_filehash(sidecar_path: Path) -> str | None:
    """sha256 of the sibling PDF bytes (sidecar.json → sidecar.pdf).
    Returns None if the PDF doesn't exist — caller decides what to do.
    Used only as filename-fallback so a missing PDF on the fallback path
    is a hard error.
    """
    pdf_path = sidecar_path.with_suffix(".pdf")
    if not pdf_path.exists():
        return None
    h = hashlib.sha256()
    with open(pdf_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_composite_source_record_id(
    sidecar: dict, sidecar_path: Path
) -> tuple[str, str]:
    """Returns (composite_key, strategy_tag).

    Strategy preference (synth contract 2026-05-03 10:41:47Z):
      1. f"{msg_id}:{filename}"            strategy="filename"
      2. f"{msg_id}:{filehash16}"          strategy="filehash"   (fallback)

    filehash fallback only fires when filename is missing/empty. Current
    corpus has 0 missing filenames so path 2 is exercised only by tests
    + future drift defense.
    """
    msg_id = sidecar["msg_id"]
    filename = sidecar.get("filename") or ""
    filename = filename.strip()
    if filename:
        return f"{msg_id}:{filename}", "filename"
    # Fallback: hash the PDF bytes
    fh = _compute_pdf_filehash(sidecar_path)
    if fh is None:
        raise ValueError(
            f"sidecar lacks filename AND sibling PDF for filehash fallback: "
            f"{sidecar_path}"
        )
    return f"{msg_id}:{fh[:16]}", "filehash"


def build_record(sidecar: dict, sidecar_path: Path) -> dict:
    """Build the NM call kwargs from a sidecar dict. Caller passes these
    straight through to record_evidence_artifact in --live mode, OR writes
    the dict to the dry-run JSONL.
    """
    msg_id = sidecar["msg_id"]
    text = sidecar["text"]
    valid_from = _parse_downloaded_at(sidecar["downloaded_at"])

    composite_key, key_strategy = _build_composite_source_record_id(
        sidecar, sidecar_path
    )

    extra_metadata = {
        "thread_id": sidecar.get("thread_id"),
        "subject": sidecar.get("subject"),
        "from": sidecar.get("from"),
        "to": sidecar.get("to"),
        "date": sidecar.get("date"),
        "filename": sidecar.get("filename"),
        "size_bytes": sidecar.get("size_bytes"),
        "dollar_total_guess": sidecar.get("dollar_total_guess"),
        # S3 packet 2026-05-03: keep msg_id queryable even though
        # source_record_id is now composite — recall paths still want to
        # group by Gmail message.
        "msg_id": msg_id,
        "source_record_key_strategy": key_strategy,
    }
    extraction = sidecar.get("extraction") or {}
    if isinstance(extraction, dict):
        if "page_count" in extraction:
            extra_metadata["page_count"] = extraction["page_count"]
        if "method" in extraction:
            extra_metadata["extraction_method"] = extraction["method"]
    # Strip None values — keeps the metadata clean
    extra_metadata = {k: v for k, v in extra_metadata.items() if v is not None}

    # Pre-computed evidence_id for dry-run preview parity with live
    evidence_id = _compute_evidence_id(
        evidence_type=EVIDENCE_TYPE,
        source_system=SOURCE_SYSTEM,
        source_record_id=composite_key,
    )

    return {
        "evidence_type": EVIDENCE_TYPE,
        "capability_id": CAPABILITY_ID,
        "source_system": SOURCE_SYSTEM,
        "source_path": str(sidecar_path),
        "content": text,
        "privacy_class": PRIVACY_CLASS,
        "confidence": CONFIDENCE,
        "source_record_id": composite_key,
        "valid_from": valid_from,
        "extra_metadata": extra_metadata,
        # Pre-computed for dry-run parity (live ingest will recompute identically)
        "_preview_evidence_id": evidence_id,
        "_composite_key": composite_key,
        "_msg_id": msg_id,
    }


def _check_db_guard(db_path: Path) -> tuple[bool, str]:
    """Pre-flight: returns (ok, reason).

    S4 hardening (2026-05-03): DB is guarded only when ALL of:
      1. PRAGMA user_version >= EVIDENCE_LEDGER_TARGET_USER_VERSION (v2), AND
      2. evidence_ledger table exists, AND
      3. EVIDENCE_LEDGER_REQUIRED_INDEX (v2 composite index) exists.

    Rejects: no-table DBs, table-only (v0/v1) DBs, user-version-only DBs
    missing the index, and any DB below v2. Requires the exact v2 shape
    installed by S2b's schema_upgrade._ensure_evidence_ledger().
    """
    if not db_path.exists():
        return False, f"DB file does not exist: {db_path}"
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            uv = conn.execute("PRAGMA user_version").fetchone()[0]
            if uv < EVIDENCE_LEDGER_TARGET_USER_VERSION:
                return False, (
                    f"DB guard failed: user_version={uv} (need "
                    f">={EVIDENCE_LEDGER_TARGET_USER_VERSION}); "
                    f"run SchemaUpgrade to migrate"
                )
            # user_version >= 2 — now verify table + required v2 index exist.
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (EVIDENCE_LEDGER_TABLE,),
            ).fetchone() is not None
            if not has_table:
                return False, (
                    f"DB guard failed: user_version={uv} but "
                    f"{EVIDENCE_LEDGER_TABLE} table missing; malformed v2 DB"
                )
            has_index = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
                (EVIDENCE_LEDGER_REQUIRED_INDEX,),
            ).fetchone() is not None
            if not has_index:
                return False, (
                    f"DB guard failed: user_version={uv}, table present but "
                    f"{EVIDENCE_LEDGER_REQUIRED_INDEX} index missing; "
                    f"malformed v2 DB — run SchemaUpgrade to repair"
                )
            return True, (
                f"v2 guard OK: user_version={uv}, "
                f"{EVIDENCE_LEDGER_TABLE} table + {EVIDENCE_LEDGER_REQUIRED_INDEX} present"
            )
        finally:
            conn.close()
    except sqlite3.Error as e:
        return False, f"DB guard probe failed: {type(e).__name__}: {e}"


def discover_sidecars(sidecar_dir: Path) -> list[Path]:
    if not sidecar_dir.exists():
        return []
    return sorted(p for p in sidecar_dir.iterdir()
                  if p.is_file() and p.suffix == ".json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sidecar-dir", default=str(DEFAULT_SIDECAR_DIR),
        help=f"Directory of sidecar JSONs (default: {DEFAULT_SIDECAR_DIR})",
    )
    parser.add_argument(
        "--watermark", default=str(DEFAULT_WATERMARK),
        help=f"Watermark JSON path (default: {DEFAULT_WATERMARK})",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Ignore watermark; process every sidecar in the dir.",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="LIVE mode — call record_evidence_artifact (writes canonical "
             "substrate). Default is dry-run.",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="(--live only) Path to NM substrate DB. Default = NeuralMemory's "
             "default (~/.neural_memory/memory.db).",
    )
    parser.add_argument(
        "--show", type=int, default=0,
        help="Print first N built records to stdout for inspection.",
    )
    args = parser.parse_args()

    sidecar_dir = Path(args.sidecar_dir).expanduser()
    watermark_path = Path(args.watermark).expanduser()

    # Sanity check
    if EVIDENCE_TYPE not in EVIDENCE_TYPES:
        print(
            f"FATAL: hard-coded EVIDENCE_TYPE={EVIDENCE_TYPE!r} not in "
            f"ae_workflow_helpers.EVIDENCE_TYPES — refusing to run.",
            file=sys.stderr,
        )
        return 4

    sidecars = discover_sidecars(sidecar_dir)
    if not sidecars:
        print(
            f"WARNING: no sidecars found at {sidecar_dir} — nothing to do.",
            file=sys.stderr,
        )
        # Still produce an empty report so callers can detect the no-op cleanly
        out_dir = LIVE_DIR if args.live else DRYRUN_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_now = int(time.time())
        out_path = out_dir / f"sent-pdf-{ts_now}.jsonl"
        out_path.write_text("")
        print(f"empty report: {out_path}")
        return 0

    skip_set: set[str] = set() if args.backfill else load_watermark(watermark_path)

    out_dir = LIVE_DIR if args.live else DRYRUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_now = int(time.time())
    suffix = "live" if args.live else "dry"
    out_path = out_dir / f"sent-pdf-{ts_now}.{suffix}.jsonl"

    # Live mode: pre-flight DB guard refusal + lazy NM init
    mem = None
    target_db_for_report: str | None = None
    if args.live:
        # Resolve target DB path (caller-supplied or canonical default)
        if args.db_path:
            target_db = Path(args.db_path).expanduser().resolve()
        else:
            target_db = CANONICAL_DB_PATH

        # Guard refusal: ALWAYS check, regardless of canonical vs copy.
        # Synth contract: "block canonical --live without DB guard/copy proof".
        # Copy DBs ALSO must have the guard (otherwise the copy isn't proof of
        # safety). The "copy proof" path is: caller intentionally creates a
        # copy DB AND runs schema_upgrade against it before --live.
        ok, reason = _check_db_guard(target_db)
        if not ok:
            print(
                f"REFUSED: --live blocked — DB guard check failed.\n"
                f"  target DB: {target_db}\n"
                f"  reason: {reason}\n"
                f"  fix: run python/schema_upgrade.py against this DB first "
                f"(creates {EVIDENCE_LEDGER_TABLE} + bumps user_version to "
                f">={EVIDENCE_LEDGER_TARGET_USER_VERSION}).\n"
                f"  canonical --live: also requires --db-path proof when "
                f"the canonical DB itself is not yet upgraded.",
                file=sys.stderr,
            )
            return 5

        target_db_for_report = str(target_db)
        from memory_client import NeuralMemory  # noqa: WPS433 (deferred import is intentional)
        mem = NeuralMemory(db_path=str(target_db))

    skipped_watermark = 0
    processed = 0
    errors = 0
    inserted = 0
    deduped = 0
    rows_for_show: list[dict] = []
    new_processed_keys: set[str] = set()

    with open(out_path, "w") as fh:
        for sidecar_path in sidecars:
            row_report: dict = {"sidecar_path": str(sidecar_path)}
            try:
                with open(sidecar_path) as sf:
                    sidecar = json.load(sf)
                missing = REQUIRED_FIELDS - set(sidecar.keys())
                if missing:
                    raise ValueError(f"missing required fields: {sorted(missing)}")
                msg_id = sidecar["msg_id"]
                row_report["msg_id"] = msg_id

                kwargs = build_record(sidecar, sidecar_path)
                preview_evidence_id = kwargs.pop("_preview_evidence_id")
                composite_key = kwargs.pop("_composite_key")
                kwargs.pop("_msg_id")
                row_report["composite_key"] = composite_key

                if composite_key in skip_set:
                    skipped_watermark += 1
                    row_report["skipped_watermark"] = True
                    fh.write(json.dumps(row_report) + "\n")
                    continue

                if args.live:
                    result = record_evidence_artifact(mem, **kwargs)
                    row_report.update({
                        "memory_id": result["memory_id"],
                        "evidence_id": result["evidence_id"],
                        "inserted": result["inserted"],
                    })
                    if result["inserted"]:
                        inserted += 1
                    else:
                        deduped += 1
                else:
                    row_report.update({
                        "evidence_id": preview_evidence_id,
                        "memory_id": None,
                        "inserted": None,
                        "dry_run": True,
                        "would_call": {
                            "evidence_type": kwargs["evidence_type"],
                            "capability_id": kwargs["capability_id"],
                            "source_system": kwargs["source_system"],
                            "source_path": kwargs["source_path"],
                            "source_record_id": kwargs["source_record_id"],
                            "privacy_class": kwargs["privacy_class"],
                            "valid_from": kwargs["valid_from"],
                            "extra_metadata": kwargs["extra_metadata"],
                            "content_len": len(kwargs["content"]),
                        },
                    })
                processed += 1
                new_processed_keys.add(composite_key)
                if args.show and len(rows_for_show) < args.show:
                    rows_for_show.append(row_report)
            except Exception as e:
                errors += 1
                row_report["error"] = f"{type(e).__name__}: {e}"
            fh.write(json.dumps(row_report, default=str) + "\n")

    # Update watermark only on --live success — dry-run is purely informational
    # and shouldn't advance state.
    if args.live and not args.backfill and new_processed_keys:
        merged = (skip_set | new_processed_keys)
        save_watermark(watermark_path, merged)
    elif args.live and args.backfill and new_processed_keys:
        # Backfill should still seed the watermark so a subsequent
        # non-backfill run skips what we just ingested.
        save_watermark(watermark_path, new_processed_keys | skip_set)

    print(f"=== sent-PDF sidecar ingest report ===")
    print(f"  mode: {'LIVE (canonical substrate write)' if args.live else 'dry-run'}")
    if args.live:
        print(f"  target DB: {target_db_for_report}")
    print(f"  sidecar dir: {sidecar_dir}")
    print(f"  total found: {len(sidecars)}")
    print(f"  skipped (watermark): {skipped_watermark}")
    print(f"  processed: {processed}")
    print(f"  distinct composite keys this run: {len(new_processed_keys)}")
    if args.live:
        print(f"    inserted (new): {inserted}")
        print(f"    deduped (existing evidence_id): {deduped}")
    print(f"  errors: {errors}")
    print(f"  output: {out_path}")
    if rows_for_show:
        print(f"\n=== first {len(rows_for_show)} rows ===")
        for row in rows_for_show:
            print(json.dumps(row, indent=2, default=str))

    return 0 if errors == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
