#!/usr/bin/env python3
"""
migrate_colbert_tokens.py — backfill ColBERT token blobs for existing memories.

One-shot, restart-safe, batched. Uses BGE-M3 directly via colbert_helper
(loads its own model copy on GPU; does NOT touch the running embed-server).

Resumability: persists `colbert_tokens_migrated_max_id` in the store's
db_meta/meta key-value table. On rerun, skips memories with id <=
that checkpoint AND any memory that already has colbert_tokens set.

Usage:
    python migrate_colbert_tokens.py                           # default DB
    python migrate_colbert_tokens.py --db /path/to/memory.db
    python migrate_colbert_tokens.py --batch-size 256 --limit 10000
    MM_DB_BACKEND=postgres python migrate_colbert_tokens.py    # PG mirror

Cost (warning):
    1 row ≈ 64 KB → 230k memories ≈ 14.7 GB on the SQLite BLOB column.
    Confirm disk has the headroom before proceeding.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PY_DIR = Path(__file__).resolve().parent
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from colbert_helper import (  # noqa: E402
    DEFAULT_TOP_K,
    colbert_available,
    encode_tokens_batch,
    pack_tokens,
)

logger = logging.getLogger("colbert-migrate")

CHECKPOINT_KEY = "colbert_tokens_migrated_max_id"


def _open_store(db_path: str):
    backend = (os.environ.get("MM_DB_BACKEND") or "").strip().lower()
    if backend == "postgres":
        from postgres_store import PostgresStore
        store = PostgresStore()
    else:
        from memory_client import SQLiteStore
        store = SQLiteStore(db_path)
    return store


def _read_checkpoint(store) -> int:
    try:
        v = store.get_meta(CHECKPOINT_KEY)
        return int(v) if v else 0
    except Exception:
        return 0


def _write_checkpoint(store, value: int) -> None:
    try:
        store.set_meta(CHECKPOINT_KEY, str(int(value)))
    except Exception:
        logger.warning("failed to write checkpoint=%s", value, exc_info=True)


def _count_remaining(store) -> int:
    try:
        if hasattr(store, "conn"):
            row = store.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE colbert_tokens IS NULL"
            ).fetchone()
            return int(row[0])
        with store._cursor() as (_c, cur):
            cur.execute("SELECT COUNT(*) FROM memories WHERE colbert_tokens IS NULL")
            row = cur.fetchone()
            return int(row[0])
    except Exception:
        return -1


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill ColBERT tokens for Mazemaker memories")
    p.add_argument("--db", default=None,
                   help="SQLite DB path (default: ~/.mazemaker/data/memory.db, "
                        "fallback ~/.mazemaker/engine/memory.db). Ignored when "
                        "MM_DB_BACKEND=postgres.")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--restart", action="store_true",
                   help="Ignore the checkpoint and rescan from id 0.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    db_path = args.db
    if not db_path and (os.environ.get("MM_DB_BACKEND") or "").lower() != "postgres":
        for cand in [
            Path.home() / ".mazemaker" / "data" / "memory.db",
            Path.home() / ".mazemaker" / "engine" / "memory.db",
        ]:
            if cand.exists() and cand.stat().st_size > 0:
                db_path = str(cand)
                break
        if not db_path:
            db_path = str(Path.home() / ".mazemaker" / "data" / "memory.db")

    store = _open_store(db_path or "")
    checkpoint = 0 if args.restart else _read_checkpoint(store)
    remaining = _count_remaining(store)
    logger.info(
        "open store=%s checkpoint=%d remaining=%d batch_size=%d encode_batch=%d top_k=%d",
        db_path or "(postgres)", checkpoint, remaining,
        args.batch_size, args.encode_batch, args.top_k,
    )

    if not colbert_available():
        logger.error("colbert_helper unavailable (model not loadable). Aborting.")
        return 2

    processed = 0
    written = 0
    t_run = time.perf_counter()
    last_log = t_run
    last_id = checkpoint

    try:
        while True:
            batch: list[tuple[int, str]] = []
            for mid, content in store.stream_missing_colbert(
                batch_size=args.batch_size, start_after_id=last_id
            ):
                batch.append((mid, content))
                if len(batch) >= args.batch_size:
                    break
            if not batch:
                logger.info("no more memories to migrate; done")
                break
            ids = [b[0] for b in batch]
            texts = [b[1] for b in batch]
            t_enc = time.perf_counter()
            arrs = encode_tokens_batch(
                texts, top_k=args.top_k, batch_size=args.encode_batch,
            )
            enc_ms = (time.perf_counter() - t_enc) * 1000.0
            t_w = time.perf_counter()
            batch_written = 0
            pairs: list[tuple[int, bytes]] = []
            for mid, arr in zip(ids, arrs):
                if arr is None:
                    continue
                pairs.append((mid, pack_tokens(arr)))
            if not args.dry_run and pairs:
                if hasattr(store, "set_colbert_tokens_many"):
                    try:
                        store.set_colbert_tokens_many(pairs)
                        batch_written = len(pairs)
                    except Exception:
                        logger.warning("bulk write failed; falling back to per-row",
                                       exc_info=True)
                        for mid, blob in pairs:
                            try:
                                store.set_colbert_tokens(mid, blob)
                                batch_written += 1
                            except Exception:
                                logger.warning("write failed for id=%d", mid, exc_info=True)
                else:
                    for mid, blob in pairs:
                        try:
                            store.set_colbert_tokens(mid, blob)
                            batch_written += 1
                        except Exception:
                            logger.warning("write failed for id=%d", mid, exc_info=True)
            else:
                batch_written = len(pairs)
            write_ms = (time.perf_counter() - t_w) * 1000.0
            written += batch_written
            processed += len(batch)
            last_id = ids[-1]
            if not args.dry_run:
                _write_checkpoint(store, last_id)
            now = time.perf_counter()
            if not args.quiet and (now - last_log) > 5.0:
                elapsed = now - t_run
                rate = processed / max(0.1, elapsed)
                eta = remaining / rate if (remaining > 0 and rate > 0) else 0.0
                logger.info(
                    "processed=%d written=%d last_id=%d enc=%.0fms write=%.0fms "
                    "rate=%.1f mem/s eta=%.0fs",
                    processed, written, last_id, enc_ms, write_ms, rate, eta,
                )
                last_log = now
            if args.limit and processed >= args.limit:
                logger.info("hit --limit %d; stopping", args.limit)
                break
    except KeyboardInterrupt:
        logger.warning("interrupted; checkpoint=%d processed=%d written=%d",
                       last_id, processed, written)
        return 1

    elapsed = time.perf_counter() - t_run
    bytes_per = (args.top_k * 1024 * 2) + 8
    total_bytes = written * bytes_per
    logger.info(
        "DONE processed=%d written=%d elapsed=%.1fs rate=%.1f mem/s "
        "blob_bytes/mem=%d disk=%.2f GB",
        processed, written, elapsed,
        processed / max(0.1, elapsed),
        bytes_per,
        total_bytes / 1024 ** 3,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
