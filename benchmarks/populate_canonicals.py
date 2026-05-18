#!/usr/bin/env python3
"""populate_canonicals.py — build the consolidation_canonicals retrieval
prior table from existing Stage C atomic facts.

The earlier `canonicalize_preferences.py` script wrote canonical
memories straight into the main `memories` table. That regressed the
bench because canonical content out-ranks per-session gold memories
without subsuming all golds in its label (iter25 -3pp R@5).

This populator writes the SAME clusters to a separate
`consolidation_canonicals` table that does NOT participate in primary
recall. The Mazemaker recall pipeline consults it as a side-channel
prior: when the query strongly matches a canonical (cos≥0.85), all
candidate memories whose session-id appears in that canonical's
`evidence_session_ids` array get a relevance boost.

Net effect: the consolidation "knows" the user prefers X across N
sessions, but at retrieval time it lifts the actual session memory
to the top — never competing for that top-K slot itself.

USAGE
    python benchmarks/populate_canonicals.py --variant oracle \\
        --similarity 0.85 --min-cluster 2 --max-cluster-size 30
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

CACHE_DB = "mm10m_bench"


def _sid_from_label(label: str) -> str:
    for seg in label.split(":"):
        if seg and seg not in ("session", "afe", "chunk") and not seg.isdigit():
            if len(seg) <= 2 and seg[0] in "ABC" and seg[1:].isdigit():
                continue
            return seg
    return ""


def _parse_pgvector(raw):
    if isinstance(raw, str):
        return np.fromstring(raw.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(raw, dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="oracle", choices=["s", "m", "oracle"])
    p.add_argument("--similarity", type=float, default=0.85)
    p.add_argument("--min-cluster", type=int, default=2)
    p.add_argument("--max-cluster-size", type=int, default=30)
    p.add_argument("--rebuild", action="store_true",
                   help="Wipe consolidation_canonicals before populating")
    args = p.parse_args()

    schema = f"longmemeval_{args.variant}_bgem3_1024"
    os.environ["MM_DB_BACKEND"] = "postgres"
    os.environ["MM_POSTGRES_DB"] = CACHE_DB
    os.environ["MM_POSTGRES_SCHEMA"] = schema

    print(f"[canon-table] DB={CACHE_DB} schema={schema}", flush=True)

    from memory_client import Mazemaker
    nm = Mazemaker(
        db_path="/dev/null",
        embedding_backend="auto",
        lazy_graph=True,
        retrieval_mode="semantic",
        rerank=False,
    )

    if args.rebuild:
        with nm.store._cursor() as (_conn, cur):
            cur.execute("TRUNCATE TABLE consolidation_canonicals")
        print("[canon-table] rebuild: wiped table", flush=True)

    # Pull Stage C user-side facts with embeddings.
    with nm.store._cursor() as (_conn, cur):
        cur.execute(
            "SELECT id, label, content, embedding FROM memories "
            "WHERE label LIKE %s AND label NOT LIKE %s "
            "AND embedding IS NOT NULL",
            ("%::afe::C%", "derived:cluster:%::afe::%"),
        )
        rows = cur.fetchall()
    n = len(rows)
    print(f"[canon-table] {n:,} Stage C facts loaded", flush=True)
    if n < 2:
        return 0

    labels = [r[1] for r in rows]
    contents = [r[2] for r in rows]
    embs = np.stack([_parse_pgvector(r[3]) for r in rows])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    print(f"[canon-table] {n}×{n} cosine matrix...", flush=True)
    sims = embs @ embs.T
    np.fill_diagonal(sims, 0.0)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in np.argwhere(sims >= args.similarity):
        if i < j:
            union(int(i), int(j))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    clusters = [
        g for g in groups.values()
        if args.min_cluster <= len(g) <= args.max_cluster_size
    ]
    clusters.sort(key=len, reverse=True)
    print(f"[canon-table] {len(clusters)} clusters in [{args.min_cluster}, "
          f"{args.max_cluster_size}]", flush=True)

    emitted = 0
    skipped = 0
    rows_to_insert: list[tuple[str, list[float], list[str], int]] = []
    for idxs in clusters:
        cluster_sids: list[str] = []
        seen = set()
        for i in idxs:
            sid = _sid_from_label(labels[i])
            if sid and sid not in seen:
                seen.add(sid)
                cluster_sids.append(sid)
        if len(cluster_sids) < args.min_cluster:
            skipped += 1
            continue
        cluster_contents = [(contents[i] or "").strip() for i in idxs]
        user_first = [c for c in cluster_contents if c.lower().startswith("user ")]
        pool = user_first if user_first else cluster_contents
        canonical_text = max(pool, key=len)[:280]
        # Canonical embedding = mean of member embeddings (unit-normed).
        cluster_vec = embs[idxs].mean(axis=0)
        nrm = np.linalg.norm(cluster_vec)
        if nrm > 0:
            cluster_vec = cluster_vec / nrm
        rows_to_insert.append(
            (canonical_text, cluster_vec.tolist(), cluster_sids, len(idxs))
        )
        emitted += 1

    # Bulk insert.
    if rows_to_insert:
        with nm.store._cursor() as (_conn, cur):
            cur.executemany(
                "INSERT INTO consolidation_canonicals "
                "(text, embedding, evidence_session_ids, cluster_size) "
                "VALUES (%s, %s::vector, %s, %s)",
                [
                    (t, "[" + ",".join(f"{x:.6f}" for x in v) + "]", sids, sz)
                    for (t, v, sids, sz) in rows_to_insert
                ],
            )
    print(f"\n[canon-table] DONE  emitted={emitted}  skipped={skipped}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
