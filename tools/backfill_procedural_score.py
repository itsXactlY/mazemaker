#!/usr/bin/env python3.11
"""backfill_procedural_score.py — set baseline procedural_score=0.7 on
existing kind='procedural' memories that have NULL score.

Phase 7.5-α companion. The wire-fix in remember() (memory_client.py)
auto-populates procedural_score for FUTURE writes; this tool backfills
existing rows so the unified scorer's procedural channel actually
contributes to ranking against historical content.

Idempotent: only updates rows where procedural_score IS NULL. Re-running
is a no-op. Wraps in transaction with explicit dry-run default.

Usage:
    tools/backfill_procedural_score.py            # dry-run; reports counts
    tools/backfill_procedural_score.py --execute  # actually mutate
    tools/backfill_procedural_score.py --score 0.8  # override default
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_DEFAULT_DB = Path.home() / ".neural_memory" / "memory.db"
_DEFAULT_SCORE = 0.7


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(_DEFAULT_DB))
    p.add_argument("--execute", action="store_true",
                   help="Actually mutate (default dry-run)")
    p.add_argument("--score", type=float, default=_DEFAULT_SCORE,
                   help=f"Score to assign (default {_DEFAULT_SCORE})")
    args = p.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db))

    # --- before -----------------------------------------------------
    total_proc = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE kind='procedural'"
    ).fetchone()[0]
    null_proc = conn.execute(
        "SELECT COUNT(*) FROM memories "
        "WHERE kind='procedural' AND procedural_score IS NULL"
    ).fetchone()[0]
    set_proc = total_proc - null_proc

    print(f"BEFORE:")
    print(f"  procedural memories total:           {total_proc}")
    print(f"  procedural memories with NULL score: {null_proc}")
    print(f"  procedural memories with set score:  {set_proc}")
    print(f"  proposed score for backfill:         {args.score}")
    print()

    if not args.execute:
        print(f"DRY-RUN. Pass --execute to actually backfill.")
        return 0

    if null_proc == 0:
        print(f"Nothing to do — all procedural memories already have scores.")
        return 0

    # --- mutate ------------------------------------------------------
    print(f"EXECUTE: setting procedural_score={args.score} on {null_proc} rows...")
    try:
        conn.execute("BEGIN")
        cur = conn.execute(
            "UPDATE memories SET procedural_score = ? "
            "WHERE kind = 'procedural' AND procedural_score IS NULL",
            (args.score,),
        )
        affected = cur.rowcount
        if affected != null_proc:
            print(f"SAFETY ABORT: UPDATE rowcount {affected} != expected {null_proc}")
            conn.rollback()
            return 2
        conn.commit()
        print(f"  updated: {affected} rows")
    except Exception as e:
        conn.rollback()
        print(f"ERROR: {e}; rolled back.", file=sys.stderr)
        return 3

    # --- after -------------------------------------------------------
    final_null = conn.execute(
        "SELECT COUNT(*) FROM memories "
        "WHERE kind='procedural' AND procedural_score IS NULL"
    ).fetchone()[0]
    final_set = conn.execute(
        "SELECT COUNT(*) FROM memories "
        "WHERE kind='procedural' AND procedural_score IS NOT NULL"
    ).fetchone()[0]
    print(f"\nAFTER:")
    print(f"  procedural memories with score:    {final_set}")
    print(f"  procedural memories with NULL:     {final_null}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
