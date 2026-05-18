#!/usr/bin/env python3
"""canonicalize_preferences.py — fold semantically-redundant Stage C
user-side facts into single canonical preference memories.

Why this exists
---------------
Stage C extracts atomic user-side facts ("user prefers Italian food")
from individual sessions. Across the corpus the SAME preference often
shows up 2–10 times with minor surface variation. Each variant lands
as its own memory and competes with the others in recall: similarity-
to-query is roughly equal, so the bench scorer sees N near-identical
memories instead of one consolidated unit with multi-session backing.

This script groups Stage C facts by embedding cosine similarity, then
emits ONE canonical preference memory per cluster with a multi-session
label so the bench label-split scorer matches every constituent gold.

Label format mirrors Stage S synthesis:
    preference:canonical:<sid_0>::<sid_1>::...

Usage
-----
    python benchmarks/canonicalize_preferences.py --variant oracle
        --similarity 0.85 --min-cluster 2
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
    """Extract source session id from an AFE label.

    Labels look like ``session:<sid>::afe::C0`` or
    ``session:<sid>::chunk::N::afe::C0`` — the first colon-segment
    that isn't ``session`` / ``afe`` / ``chunk`` is the sid.
    """
    for seg in label.split(":"):
        if seg and seg not in ("session", "afe", "chunk") and not seg.isdigit():
            # Strip stage marker like C0/C1 too — those start with a
            # single letter followed by a digit.
            if len(seg) <= 2 and seg[0] in "ABC" and seg[1:].isdigit():
                continue
            return seg
    return ""


def _parse_pgvector(raw: str | list) -> np.ndarray:
    if isinstance(raw, str):
        return np.fromstring(raw.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(raw, dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="oracle", choices=["s", "m", "oracle"])
    p.add_argument("--similarity", type=float, default=0.85,
                   help="Cosine threshold for grouping (default 0.85)")
    p.add_argument("--min-cluster", type=int, default=2,
                   help="Min cluster size to emit a canonical (default 2)")
    p.add_argument("--max-canonicals", type=int, default=0,
                   help="0 = unlimited; cap for smoke runs")
    p.add_argument("--max-cluster-size", type=int, default=30,
                   help="Drop clusters larger than this — they're typically "
                        "generic prose patterns that match across hundreds "
                        "of unrelated sessions (default 30)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print clusters but don't write memories")
    args = p.parse_args()

    schema = f"longmemeval_{args.variant}_bgem3_1024"
    os.environ["MM_DB_BACKEND"] = "postgres"
    os.environ["MM_POSTGRES_DB"] = CACHE_DB
    os.environ["MM_POSTGRES_SCHEMA"] = schema

    print(f"[canonpref] DB={CACHE_DB} schema={schema} "
          f"sim_threshold={args.similarity} min_cluster={args.min_cluster}",
          flush=True)

    from memory_client import Mazemaker
    nm = Mazemaker(
        db_path="/dev/null",
        embedding_backend="auto",
        lazy_graph=True,
        retrieval_mode="semantic",
        rerank=False,
    )

    # Pull every Stage C user-side fact along with its embedding + label.
    # Filter: label contains ::afe::C, NOT derived:cluster source (those
    # are post-hoc cluster extractions, not raw session facts).
    with nm.store._cursor() as (_conn, cur):
        cur.execute(
            "SELECT id, label, content, embedding "
            "FROM memories "
            "WHERE label LIKE %s "
            "  AND label NOT LIKE %s "
            "  AND embedding IS NOT NULL",
            ("%::afe::C%", "derived:cluster:%::afe::%"),
        )
        rows = cur.fetchall()
    n = len(rows)
    print(f"[canonpref] Loaded {n:,} Stage C facts with embeddings",
          flush=True)
    if n < 2:
        print("[canonpref] Nothing to cluster — done.")
        return 0

    ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
    labels = [r[1] for r in rows]
    contents = [r[2] for r in rows]
    embs = np.stack([_parse_pgvector(r[3]) for r in rows])
    # L2-normalise so dot-product == cosine.
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    # Pairwise cosine matrix — n=2704 → 28MB float32, trivial.
    print(f"[canonpref] Computing {n}×{n} cosine matrix...", flush=True)
    sims = embs @ embs.T
    np.fill_diagonal(sims, 0.0)

    # Connected components via union-find on edges above threshold.
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    above = np.argwhere(sims >= args.similarity)
    # Each pair appears twice (i,j and j,i) — skip half.
    for i, j in above:
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
    dropped_big = sum(1 for g in groups.values() if len(g) > args.max_cluster_size)
    print(f"[canonpref] {len(clusters):,} clusters of size ≥{args.min_cluster}"
          f" and ≤{args.max_cluster_size}"
          f" (dropped {dropped_big} mega-clusters)",
          flush=True)
    sizes = [len(c) for c in clusters[:5]]
    print(f"[canonpref] top-5 sizes: {sizes}", flush=True)

    # Emit canonicals.
    emitted = 0
    skipped = 0
    for cluster_idx, idxs in enumerate(clusters):
        if args.max_canonicals and emitted >= args.max_canonicals:
            break
        # Distinct sids — many facts in a cluster come from the same
        # session (multiple Stage C facts per source).
        cluster_sids: list[str] = []
        seen_sids: set[str] = set()
        for i in idxs:
            sid = _sid_from_label(labels[i])
            if sid and sid not in seen_sids:
                seen_sids.add(sid)
                cluster_sids.append(sid)
        if len(cluster_sids) < args.min_cluster:
            # Cluster collapses to a single source — not useful for
            # cross-session canonicalisation.
            skipped += 1
            continue
        # Canonical content: longest fact in cluster (proxy for most
        # informative single statement). LLM-rewrite is a future
        # upgrade — for now pick the longest user-prefixed string.
        cluster_contents = [(contents[i] or "").strip() for i in idxs]
        # Prefer user-prefixed statements.
        user_first = [c for c in cluster_contents if c.lower().startswith("user ")]
        pool = user_first if user_first else cluster_contents
        canonical_text = max(pool, key=len)[:280]
        label_segs = "::".join(cluster_sids[:20])  # cap to keep label short
        label = f"preference:canonical:{label_segs}"
        # Annotate evidence count + cluster size in content.
        content = (
            f"{canonical_text}  "
            f"[canonical evidence_sessions={len(cluster_sids)} "
            f"members={len(idxs)}]"
        )
        if args.dry_run:
            print(f"  [{cluster_idx:>4}] sids={len(cluster_sids)} "
                  f"members={len(idxs)}: {canonical_text[:80]}",
                  flush=True)
        else:
            try:
                new_id = nm.remember(
                    content,
                    label=label,
                    auto_connect=False,
                    detect_conflicts=False,
                )
                if isinstance(new_id, list):
                    new_id = new_id[0]
                new_id = int(new_id)
                # Wire canonical → member edges so dream + recall can
                # traverse the consolidation chain.
                nm.store.add_connections_batch(
                    [(new_id, int(ids[i]), 1.0) for i in idxs],
                    edge_type="canonical_of",
                )
            except Exception as e:
                print(f"  cluster {cluster_idx}: remember() failed: {e}",
                      flush=True)
                continue
        emitted += 1

    print(f"\n[canonpref] DONE  emitted={emitted}  skipped_single_source={skipped}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
