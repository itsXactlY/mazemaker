#!/usr/bin/env python3
"""Multi-recall CLI for BEAM-10M conv-1 — invoked by Sonnet agent.

Each call: query string + k → JSON top-k results (label, content, score).
The agent calls it 1-10x per question, refining queries to triangulate
evidence the way a real Mazemaker user would.

Usage:
    python recall_conv1.py "what is the latest Milvus version" --k 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY_DIR = ROOT / "python"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

os.environ.setdefault("MM_COLBERT_ENABLED", "1")
os.environ.setdefault("MM_DAE_ENABLED", "1")

from memory_client import Mazemaker  # noqa: E402

DB = os.environ.get("BEAM_CONV1_DB", "/tmp/beam-10m/conv-1/memory.db")

_engine = None


def engine() -> Mazemaker:
    global _engine
    if _engine is None:
        _engine = Mazemaker(
            db_path=DB,
            embedding_backend="auto",
            use_cpp=False,
            retrieval_mode="hybrid",
            use_hnsw=False,
            lazy_graph=True,
            rerank=False,
            channel_weights={"colbert": 1.5, "dae": 1.0},
        )
    return _engine


def main() -> int:
    p = argparse.ArgumentParser(description="conv-1 multi-recall CLI")
    p.add_argument("query", help="recall query string")
    p.add_argument("--k", type=int, default=25, help="top-k (default 25)")
    p.add_argument("--snippet-chars", type=int, default=400,
                   help="truncate content per row (default 400)")
    args = p.parse_args()

    nm = engine()
    results = nm.recall(
        args.query, k=args.k,
        hybrid=True,
        enable_colbert=True,
        colbert_weight=1.5,
        enable_dae=True,
        dae_weight=1.0,
    )
    out = []
    for r in results:
        content = (r.get("content", "") or "").strip()
        if len(content) > args.snippet_chars:
            content = content[: args.snippet_chars] + "…"
        out.append({
            "label": r.get("label", ""),
            "score": round(float(r.get("score", 0.0)), 4),
            "content": content,
        })
    print(json.dumps({"query": args.query, "k": args.k, "results": out},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
