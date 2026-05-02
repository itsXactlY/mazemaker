#!/usr/bin/env python3.11
"""cleanup_junk_entities.py — delete entity rows + their mentions_entity edges
for entity labels that should never have been extracted.

Caught 2026-05-01 by round-2 code-correctness reviewer: even after an earlier
stopword-list expansion, ~28 confirmed junk entity labels still pollute the
top-60 by mention frequency: project meta-vocab (Phase, Session, Sprint),
protocol acronyms (API, URL, RPC, OAuth, HTTP, POST, GET, SET, CDP, SQL),
workflow noise (CHECKPOINT, MEMORY, EOD), single-char (UI, DB), test/dev terms
(Commit, Branch, PR, Test, Bench, Audit, Review).

The stopword filter in entity_extraction.py was extended in commit b69bd85
to prevent these going forward. This tool is the one-shot cleanup for
existing rows.

Usage:
    tools/cleanup_junk_entities.py            # dry-run; reports counts
    tools/cleanup_junk_entities.py --execute  # actually delete
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_DEFAULT_DB = Path.home() / ".neural_memory" / "memory.db"

# Mirrors the additions to entity_extraction._STOPWORDS in commit b69bd85.
# Anything here that's currently a 'kind=entity' row is a junk entity that
# should never have been extracted.
_JUNK_LABELS = [
    # Project meta-vocab
    "Phase", "Session", "Sprint", "CHECKPOINT", "MEMORY", "EOD",
    # Protocol acronyms
    "API", "URL", "RPC", "OAuth", "HTTP", "POST", "GET", "SET",
    "CDP", "SQL", "UI", "DB", "JSON", "YAML", "TOML",
    # Test/dev workflow nouns
    "Commit", "Branch", "Tag", "PR", "Pull",
    "Test", "Tests", "Bench", "Audit", "Review",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(_DEFAULT_DB))
    p.add_argument("--execute", action="store_true",
                   help="Actually delete (default: dry-run)")
    args = p.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db))
    placeholders = ",".join("?" * len(_JUNK_LABELS))

    # ---- before -----------------------------------------------------
    junk_entities = conn.execute(
        f"SELECT id, label FROM memories "
        f"WHERE kind='entity' AND label IN ({placeholders})",
        tuple(_JUNK_LABELS),
    ).fetchall()

    if not junk_entities:
        print("No junk entities found in live DB. Already clean.")
        return 0

    junk_ids = [row[0] for row in junk_entities]
    edge_count = conn.execute(
        f"SELECT COUNT(*) FROM connections "
        f"WHERE edge_type='mentions_entity' AND target_id IN "
        f"({','.join('?' * len(junk_ids))})",
        tuple(junk_ids),
    ).fetchone()[0]

    print(f"BEFORE:")
    print(f"  junk entity rows:        {len(junk_entities)}")
    print(f"  mentions_entity edges:   {edge_count}")
    print(f"  labels (top 10):         {[r[1] for r in junk_entities[:10]]}")
    print()

    if not args.execute:
        print(f"DRY-RUN. Pass --execute to actually delete.")
        return 0

    # ---- mutate -----------------------------------------------------
    print(f"EXECUTE: deleting {len(junk_entities)} entity rows + "
          f"{edge_count} mentions_entity edges...")
    try:
        conn.execute("BEGIN")
        # Delete edges first (FK semantics + safer ordering)
        conn.execute(
            f"DELETE FROM connections "
            f"WHERE edge_type='mentions_entity' AND target_id IN "
            f"({','.join('?' * len(junk_ids))})",
            tuple(junk_ids),
        )
        # Then the entity rows themselves
        conn.execute(
            f"DELETE FROM memories WHERE id IN "
            f"({','.join('?' * len(junk_ids))})",
            tuple(junk_ids),
        )
        conn.commit()
        print(f"  done.")
    except Exception as e:
        conn.rollback()
        print(f"ERROR: {e}; rolled back.", file=sys.stderr)
        return 2

    # ---- after ------------------------------------------------------
    remaining_junk = conn.execute(
        f"SELECT COUNT(*) FROM memories "
        f"WHERE kind='entity' AND label IN ({placeholders})",
        tuple(_JUNK_LABELS),
    ).fetchone()[0]
    print(f"\nAFTER:")
    print(f"  remaining junk entity rows: {remaining_junk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
