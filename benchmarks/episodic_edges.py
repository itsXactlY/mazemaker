#!/usr/bin/env python3
"""episodic_edges.py — emit chronological `before` edges for within-session
AFE Stage C facts.

Why
---
The bench corpus has uniform `created_at` (every memory imported in
one bulk pass), so wall-clock timestamps don't carry chronology.
But AFE labels DO: `session:<sid>::afe::C<idx>` where `idx` increases
along the source session's flow. Within a session, idx0 came before
idx1, idx1 before idx2, etc.

This script emits `before`-typed edges in the existing connections
table:

    mem(sid, idx0) --before--> mem(sid, idx1) --before--> mem(sid, idx2)

PPR + NREM already traverse the connections table, so even without
recall-side changes the chronological chain becomes a soft signal:
activated neighbours along the chain reinforce each other, which
should marginally improve temporal-reasoning recall.

Recall-side temporal channel (detect "after/before/when/initially"
in query, then walk the chain) is the next step — handled by
memory_client.py changes, not this script.

USAGE
    python benchmarks/episodic_edges.py --variant oracle
    python benchmarks/episodic_edges.py --variant oracle --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

CACHE_DB = "mm10m_bench"

# Label seq parsing: `session:answer_X::afe::C0` -> sid=answer_X, stage=C, idx=0
_AFE_LABEL_RE = re.compile(
    r"^(?:session:)?(?P<sid>[^:]+?)::afe::(?P<stage>[ABC])(?P<idx>\d+)$"
)


def parse_afe_label(label: str) -> tuple[str | None, str | None, int | None]:
    m = _AFE_LABEL_RE.match(label or "")
    if not m:
        return None, None, None
    return m.group("sid"), m.group("stage"), int(m.group("idx"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="oracle", choices=["s", "m", "oracle"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--weight", type=float, default=0.8,
                   help="Edge weight for `before` edges (default 0.8)")
    p.add_argument("--stages", default="C",
                   help="Which AFE stages to chain (default 'C' — Stage C "
                        "user-side facts only; pass 'AC' for A+C)")
    args = p.parse_args()

    schema = f"longmemeval_{args.variant}_bgem3_1024"
    os.environ["MM_DB_BACKEND"] = "postgres"
    os.environ["MM_POSTGRES_DB"] = CACHE_DB
    os.environ["MM_POSTGRES_SCHEMA"] = schema

    print(f"[episodic] DB={CACHE_DB} schema={schema} stages={args.stages}",
          flush=True)

    from memory_client import Mazemaker
    nm = Mazemaker(
        db_path="/dev/null",
        embedding_backend="auto",
        lazy_graph=True,
        retrieval_mode="semantic",
        rerank=False,
    )

    # Pull AFE labels matching the requested stages.
    stage_clause = " OR ".join(
        f"label LIKE '%::afe::{s}%'" for s in args.stages
    )
    with nm.store._cursor() as (_conn, cur):
        cur.execute(
            f"SELECT id, label FROM memories "
            f"WHERE label LIKE '%::afe::%' AND ({stage_clause})"
        )
        rows = cur.fetchall()
    print(f"[episodic] Loaded {len(rows):,} AFE rows for stages {args.stages}",
          flush=True)

    # Group by (sid, stage), sort by idx, emit chain pairs.
    chains: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    skipped_parse = 0
    for mid, lbl in rows:
        sid, stage, idx = parse_afe_label(lbl)
        if sid is None:
            skipped_parse += 1
            continue
        chains[(sid, stage)].append((idx, int(mid)))

    print(f"[episodic] {len(chains):,} (sid, stage) chains, "
          f"{skipped_parse} unparsable labels",
          flush=True)

    pairs: list[tuple[int, int, float]] = []
    chain_size_dist: dict[int, int] = defaultdict(int)
    for (sid, stage), members in chains.items():
        if len(members) < 2:
            chain_size_dist[1] += 1
            continue
        chain_size_dist[len(members)] += 1
        members.sort(key=lambda t: t[0])
        for i in range(len(members) - 1):
            a, b = members[i][1], members[i + 1][1]
            pairs.append((a, b, args.weight))

    # Histogram of chain sizes
    print("[episodic] Chain-size histogram (size : count):", flush=True)
    for sz in sorted(chain_size_dist):
        print(f"    {sz:>3} : {chain_size_dist[sz]:>6}", flush=True)
    print(f"[episodic] {len(pairs):,} `before` edges to emit", flush=True)

    if args.dry_run:
        print("[episodic] --dry-run; not writing.")
        return 0

    written = nm.store.add_connections_batch(pairs, edge_type="before")
    print(f"[episodic] DONE  add_connections_batch returned {written}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
