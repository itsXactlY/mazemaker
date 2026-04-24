#!/usr/bin/env python3
"""compact.py — H11 periodic neural-memory compaction.

Hard-deletes memories that are:
  - old (age_days > MIN_AGE_DAYS)
  - low effective salience (< MIN_EFFECTIVE_SALIENCE)
  - orphaned in the graph (no currently-valid edges)
  - NOT labeled with a sticky/protected prefix

Dry-run mode prints candidates without deleting. Audit log written per deletion.

Usage:
    python3 tools/compact.py --dry-run
    python3 tools/compact.py                  # execute
    python3 tools/compact.py --min-age 365    # more conservative
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_DB = str(Path.home() / ".neural_memory" / "memory.db")

# Match memory_client.py constants
SALIENCE_AGE_DECAY_K = 0.001
SALIENCE_ACCESS_BOOST = 0.05

# Compaction thresholds (conservative defaults)
MIN_AGE_DAYS = 180
MIN_EFFECTIVE_SALIENCE = 0.15

# Memories with any of these label prefixes are NEVER compacted
STICKY_PREFIXES = (
    "sticky:",
    "user-verified:",
    "identity:",
    "mirror-from-default:",  # H13 Phase 2 — if it was mirrored from MEMORY.md, keep it
    "legacy:",                # explicit preserved content
)


def effective_salience(base: float, access_count: int, age_days: float) -> float:
    """Match NeuralMemory._effective_salience() exactly so compaction agrees with recall."""
    base = base if base is not None else 1.0
    decay = math.exp(-SALIENCE_AGE_DECAY_K * max(0.0, age_days))
    boost = math.log1p(max(0, access_count or 0)) * SALIENCE_ACCESS_BOOST
    eff = base * decay + boost
    return max(0.1, min(2.0, eff))


def _has_valid_to_col(conn: sqlite3.Connection) -> bool:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(connections)")}
    return "valid_to" in cols


def find_candidates(conn, min_age_days: float, min_eff_sal: float):
    """Return list of candidate memories for deletion."""
    now = time.time()
    rows = conn.execute("""
        SELECT id, label, content, salience, access_count, created_at
        FROM memories
    """).fetchall()

    valid_to_exists = _has_valid_to_col(conn)
    edge_filter = ""
    if valid_to_exists:
        edge_filter = "AND (valid_to IS NULL OR valid_to > ?)"

    candidates = []
    for mid, label, content, sal, access, created_at in rows:
        age_days = (now - (created_at or now)) / 86400.0
        if age_days < min_age_days:
            continue
        if label and any(label.startswith(p) for p in STICKY_PREFIXES):
            continue

        eff = effective_salience(sal, access or 0, age_days)
        if eff >= min_eff_sal:
            continue

        if valid_to_exists:
            edge_row = conn.execute(
                f"SELECT COUNT(*) FROM connections "
                f"WHERE (source_id = ? OR target_id = ?) {edge_filter}",
                (mid, mid, now),
            ).fetchone()
        else:
            edge_row = conn.execute(
                "SELECT COUNT(*) FROM connections WHERE source_id = ? OR target_id = ?",
                (mid, mid),
            ).fetchone()
        if (edge_row[0] or 0) > 0:
            continue

        candidates.append({
            "id": mid,
            "label": label or "",
            "preview": (content or "")[:120],
            "effective_salience": round(eff, 4),
            "age_days": round(age_days, 1),
            "access_count": access or 0,
        })
    return candidates


def delete_candidates(conn, candidates):
    """Hard-delete connections (redundant for orphans but safety) + memories."""
    ids = [c["id"] for c in candidates]
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    with conn:
        conn.execute(
            f"DELETE FROM connections WHERE source_id IN ({placeholders}) "
            f"OR target_id IN ({placeholders})",
            ids + ids,
        )
        conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})",
            ids,
        )
    return len(ids)


def write_audit(candidates, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with log_path.open("a") as fh:
        for c in candidates:
            fh.write(json.dumps({**c, "deleted_at": now}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-age", type=float, default=MIN_AGE_DAYS, dest="min_age_days")
    ap.add_argument("--min-salience", type=float, default=MIN_EFFECTIVE_SALIENCE, dest="min_eff_sal")
    ap.add_argument("--audit-log", default=None, help="Audit log path (default: ~/.neural_memory/compaction-YYYY-MM-DD.log)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    candidates = find_candidates(conn, args.min_age_days, args.min_eff_sal)
    summary = {
        "db": args.db,
        "min_age_days": args.min_age_days,
        "min_eff_sal": args.min_eff_sal,
        "candidates": len(candidates),
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        for c in candidates[:50]:
            print(json.dumps(c))
        if len(candidates) > 50:
            print(f"... {len(candidates) - 50} more ...")
        summary["action"] = "dry-run (no deletions)"
    else:
        audit = Path(args.audit_log) if args.audit_log else (
            Path.home() / ".neural_memory" / f"compaction-{time.strftime('%Y-%m-%d')}.log"
        )
        write_audit(candidates, audit)
        deleted = delete_candidates(conn, candidates)
        summary["deleted"] = deleted
        summary["audit_log"] = str(audit)
        summary["action"] = "deleted"

    conn.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
