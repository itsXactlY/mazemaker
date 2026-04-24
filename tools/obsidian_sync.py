#!/usr/bin/env python3
"""obsidian_sync.py — mirror the neural-memory graph into an Obsidian vault.

Emits one Markdown file per memory into a target folder, with wikilinks
matching SQL `connections` edges. The result: Obsidian's native graph view
renders your neural-memory graph directly.

One-way mirror. Edits in Obsidian are NOT propagated back to the DB.

Usage:
    python3 tools/obsidian_sync.py \\
        --db ~/.neural_memory/memory.db \\
        --out "/Users/tito/.obsidian/obsidian-vaults/neural-memory-vault/09 — Live Graph/memories"

    # Limit to first N memories (for dev)
    python3 tools/obsidian_sync.py --db ... --out ... --max 100

    # Filter by salience (skip low-salience)
    python3 tools/obsidian_sync.py --db ... --out ... --min-salience 0.3

    # At-time snapshot (filter bi-temporal edges)
    python3 tools/obsidian_sync.py --db ... --out ... --at-time "2026-04-18T00:00:00"
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import struct
import sys
import time
from pathlib import Path
from typing import Optional

# ---- Config defaults ------------------------------------------------------

DEFAULT_DB = str(Path.home() / ".neural_memory" / "memory.db")
DEFAULT_OUT = "/Users/tito/.obsidian/obsidian-vaults/neural-memory-vault/09 — Live Graph/memories"

# Strip non-filesystem-safe characters from labels
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(text: str, max_len: int = 60) -> str:
    """Filesystem-safe slug from label / content."""
    s = _SLUG_RE.sub("-", text.strip())[:max_len].strip("-_. ")
    return s or "unlabeled"


def _format_frontmatter(meta: dict) -> str:
    """Minimal YAML — enough for Obsidian + Dataview without full yaml library."""
    lines = ["---"]
    for k, v in meta.items():
        if isinstance(v, list):
            if v:
                lines.append(f"{k}: [{', '.join(repr(x) for x in v)}]")
            else:
                lines.append(f"{k}: []")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, str):
            # escape via JSON to handle quotes/newlines
            lines.append(f'{k}: {json.dumps(v)}')
        elif v is None:
            lines.append(f"{k}: null")
        else:
            lines.append(f"{k}: {json.dumps(str(v))}")
    lines.append("---\n")
    return "\n".join(lines)


def _memory_filename(mem_id: int, label: Optional[str]) -> str:
    """Generate stable filename: `mem-0042 — label-slug.md`"""
    slug = _slug(label or "") if label else ""
    return f"mem-{mem_id:05d}{(' — ' + slug) if slug else ''}.md"


def _load_memories(conn: sqlite3.Connection, min_salience: float, max_count: Optional[int]):
    """Returns list of memory dicts. Filters by salience floor + count limit."""
    q = """
        SELECT id, label, content, salience, created_at, last_accessed, access_count
        FROM memories
        WHERE COALESCE(salience, 1.0) >= ?
        ORDER BY id
    """
    params = [min_salience]
    if max_count:
        q += " LIMIT ?"
        params.append(max_count)
    cur = conn.execute(q, params)
    return [
        {
            "id": row[0],
            "label": row[1],
            "content": row[2] or "",
            "salience": row[3] if row[3] is not None else 1.0,
            "created_at": row[4],
            "last_accessed": row[5],
            "access_count": row[6] or 0,
        }
        for row in cur.fetchall()
    ]


def _load_edges(conn: sqlite3.Connection, at_time: Optional[float]):
    """Returns edges keyed by memory id: {mem_id: [{other_id, weight, type}]}"""
    # Check if bi-temporal columns exist
    cols = {r[1] for r in conn.execute("PRAGMA table_info(connections)")}
    has_bitemp = {"valid_from", "valid_to"}.issubset(cols)

    if has_bitemp and at_time is not None:
        q = """
            SELECT source_id, target_id, weight, edge_type
            FROM connections
            WHERE (valid_from IS NULL OR valid_from <= ?)
              AND (valid_to   IS NULL OR valid_to   >  ?)
        """
        params = (at_time, at_time)
    elif has_bitemp:
        # Default: show only currently-valid edges (valid_to IS NULL or in future)
        q = """
            SELECT source_id, target_id, weight, edge_type
            FROM connections
            WHERE valid_to IS NULL OR valid_to > strftime('%s','now')
        """
        params = ()
    else:
        q = "SELECT source_id, target_id, weight, edge_type FROM connections"
        params = ()

    edges_by_id: dict[int, list] = {}
    for src, tgt, w, etype in conn.execute(q, params):
        edges_by_id.setdefault(src, []).append({"other": tgt, "weight": w, "type": etype})
        edges_by_id.setdefault(tgt, []).append({"other": src, "weight": w, "type": etype})
    return edges_by_id


def _parse_at_time(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    # Try ISO 8601
    from datetime import datetime
    try:
        return datetime.fromisoformat(raw).timestamp()
    except Exception:
        raise ValueError(f"--at-time must be epoch seconds or ISO8601: got {raw!r}")


def sync(db_path: str, out_dir: str, min_salience: float = 0.0,
         max_count: Optional[int] = None, at_time_raw: Optional[str] = None,
         include_superseded: bool = False,
         dry_run: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    at_time = _parse_at_time(at_time_raw)
    conn = sqlite3.connect(db_path)

    memories = _load_memories(conn, min_salience, max_count)
    edges = _load_edges(conn, at_time)

    # Build id → filename map so wikilinks resolve correctly across all files
    id_to_filename = {
        m["id"]: _memory_filename(m["id"], m["label"])
        for m in memories
    }

    written = 0
    skipped_superseded = 0
    labels_used = []

    for mem in memories:
        if not include_superseded and mem["content"].startswith("[SUPERSEDED]"):
            skipped_superseded += 1
            continue

        mid = mem["id"]
        fn = id_to_filename[mid]
        mem_edges = edges.get(mid, [])

        # Frontmatter
        meta = {
            "tags": ["live-graph", "neural-memory"],
            "type": "live-memory",
            "memory_id": mid,
            "label": mem["label"] or "",
            "salience": round(mem["salience"], 4),
            "access_count": mem["access_count"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                       time.localtime(mem["created_at"] or 0)) if mem["created_at"] else "",
            "last_accessed": time.strftime("%Y-%m-%dT%H:%M:%S",
                                          time.localtime(mem["last_accessed"] or 0)) if mem["last_accessed"] else "",
            "degree": len(mem_edges),
        }

        # Body
        body_lines = [
            _format_frontmatter(meta),
            f"# Memory #{mid}",
            "",
        ]
        if mem["label"]:
            body_lines.append(f"**Label:** `{mem['label']}`\n")

        body_lines.extend([
            "## Content",
            "",
            mem["content"] or "_(empty)_",
            "",
        ])

        # Connections section with wikilinks
        if mem_edges:
            body_lines.extend(["## Connections", ""])
            # Sort by weight desc
            sorted_edges = sorted(mem_edges, key=lambda e: -(e["weight"] or 0))
            for e in sorted_edges[:30]:  # cap at 30 shown; more below if exceeded
                other_id = e["other"]
                other_fn = id_to_filename.get(other_id)
                if not other_fn:
                    # other memory filtered out (low salience, superseded, etc.)
                    body_lines.append(f"- _orphan → mem {other_id}_ (weight {e['weight']:.3f}, type `{e['type']}`)")
                    continue
                # Obsidian wikilink: use filename without .md
                link_target = other_fn[:-3]  # strip .md
                body_lines.append(f"- [[{link_target}]] (weight {e['weight']:.3f}, type `{e['type']}`)")
            if len(sorted_edges) > 30:
                body_lines.append(f"\n_... {len(sorted_edges) - 30} more connections not shown_")
            body_lines.append("")

        body_lines.append("---")
        body_lines.append("← back to [[00 — Index — Live Graph]]")

        final = "\n".join(body_lines)

        if dry_run:
            print(f"[dry-run] would write {fn} ({len(final)} bytes, {len(mem_edges)} edges)")
        else:
            (out / fn).write_text(final)
        written += 1
        labels_used.append(mem["label"] or "")

    # Report
    conn.close()
    report = {
        "db_path": db_path,
        "out_dir": str(out),
        "memories_total": len(memories),
        "memories_written": written,
        "skipped_superseded": skipped_superseded,
        "edges_total": sum(len(v) for v in edges.values()) // 2,  # edges counted both directions
        "at_time": at_time_raw,
        "min_salience": min_salience,
    }
    if not dry_run:
        # Also write a _manifest.json for reproducibility
        (out / "_manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description="Sync neural-memory into Obsidian as live graph")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=DEFAULT_OUT, help="Target folder in the Obsidian vault")
    ap.add_argument("--min-salience", type=float, default=0.0, help="Skip memories below this")
    ap.add_argument("--max", type=int, default=None, dest="max_count", help="Cap memories (dev)")
    ap.add_argument("--at-time", default=None, help="Bi-temporal filter (epoch or ISO8601)")
    ap.add_argument("--include-superseded", action="store_true",
                    help="Include memories whose content starts with [SUPERSEDED]")
    ap.add_argument("--dry-run", action="store_true", help="Don't write; show what would happen")
    args = ap.parse_args()

    report = sync(
        db_path=args.db,
        out_dir=args.out,
        min_salience=args.min_salience,
        max_count=args.max_count,
        at_time_raw=args.at_time,
        include_superseded=args.include_superseded,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
