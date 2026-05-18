#!/usr/bin/env python3
"""Build the ~/.mazemaker/engine/gpu_cache/ files that gpu_recall.GpuRecallEngine
loads. Until Mazemaker.remember() learns to append to the GPU tensor on the
fly, run this after large bulk-imports to refresh the cache.

Usage:
    python3 python/scripts/build_gpu_cache.py
    python3 python/scripts/build_gpu_cache.py --db /path/to/memory.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np


def build(db_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, label, content, embedding FROM memories WHERE embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        # Fresh customer pods have an empty memory.db at install time —
        # the GPU recall cache simply can't be built yet. Return without
        # creating any files; the next call (after first remember()) will
        # have rows and succeed. Was raise SystemExit which propagated
        # past `except Exception` blocks and killed engine startup.
        print(f"  build_gpu_cache: no rows with embedding in {db_path} — skipping")
        return

    sample = len(rows[0]["embedding"])
    # The previous "% 4 == 0 → float32 else % 2 == 0 → float16" form
    # was ambiguous: a 1024d float16 blob is 2048 bytes, which is
    # divisible by 4 too, so it would be silently interpreted as
    # 512d float32 — every cosine similarity from the resulting tensor
    # would be garbage. Prefer the dim pinned in db_meta (the engine
    # records it at first write); fall back to assuming float32 (which
    # matches sentence-transformers + fastembed defaults) only when the
    # meta row is absent.
    pinned_dim = None
    try:
        row = conn.execute(
            "SELECT value FROM db_meta WHERE key = 'embed_dim'"
        ).fetchone()
        if row and row["value"]:
            pinned_dim = int(row["value"])
    except Exception:
        pass

    if pinned_dim:
        bytes_per_elem = sample // pinned_dim
        if bytes_per_elem == 4:
            dtype, dim = np.float32, pinned_dim
        elif bytes_per_elem == 2:
            dtype, dim = np.float16, pinned_dim
        else:
            raise SystemExit(
                f"unexpected bytes-per-element={bytes_per_elem} for pinned dim={pinned_dim} "
                f"(blob {sample} bytes); refusing to guess dtype"
            )
    else:
        # No pinned dim — default to float32. The engine's embed
        # backends (fastembed, sentence-transformers) all emit float32.
        if sample % 4 == 0:
            dtype, dim = np.float32, sample // 4
        else:
            raise SystemExit(
                f"no embed_dim in db_meta and blob length {sample} not divisible by 4"
            )

    print(f"  rows = {len(rows)}, dim = {dim}, dtype = {np.dtype(dtype).name}")

    emb = np.empty((len(rows), dim), dtype=dtype)
    ids, labels, contents = [], [], []
    for i, r in enumerate(rows):
        emb[i] = np.frombuffer(r["embedding"], dtype=dtype)
        ids.append(r["id"])
        labels.append(r["label"])
        contents.append(r["content"])

    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)

    t0 = time.time()
    np.save(str(out_dir / "embeddings.npy"), emb)
    # JSON, not pickle: gpu_cache lives in operator-writable ~/.mazemaker,
    # and pickle.load on an attacker-controlled file is code execution.
    with open(str(out_dir / "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "labels": labels, "contents": contents}, f)
    print(f"  wrote {emb.nbytes/1e6:.1f} MB embeddings.npy + metadata.json in {time.time()-t0:.2f}s")
    print(f"  → {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",  default=str(Path.home() / ".mazemaker" / "engine" / "memory.db"))
    ap.add_argument("--out", default=str(Path.home() / ".mazemaker" / "engine" / "gpu_cache"))
    args = ap.parse_args()
    build(Path(args.db), Path(args.out))


if __name__ == "__main__":
    main()
