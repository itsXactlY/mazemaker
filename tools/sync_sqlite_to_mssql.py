#!/usr/bin/env python3
"""
sync_sqlite_to_mssql.py - Sync Mazemaker from SQLite to MSSQL.

One-way sync: SQLite (hot store, source of truth) → MSSQL (cold store, backup).

Preserves original SQLite IDs via IDENTITY_INSERT.
Handles FK constraints, incremental sync, and batch inserts.

Usage:
    python sync_sqlite_to_mssql.py                         # Full sync
    python sync_sqlite_to_mssql.py --incremental           # Only new rows
    python sync_sqlite_to_mssql.py --dry-run               # Preview only
    python sync_sqlite_to_mssql.py --skip-connections      # Memories only
    python sync_sqlite_to_mssql.py --db /path/to/memory.db # Custom SQLite path
    python sync_sqlite_to_mssql.py --filter-garbage        # Skip turn-%, DD% labels
"""
import argparse
import json
import os
import re
import sqlite3
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EMBEDDING_DIM = 1024
DEFAULT_SQLITE = os.path.expanduser("~/.neural_memory/memory.db")
SYNC_STATE_FILE = os.path.expanduser("~/.neural_memory/sync_state.json")

# Label patterns considered garbage (turn-level, DD%, debug noise)
GARBAGE_PATTERNS = [
    re.compile(r'^turn-\d+'),
    re.compile(r'^DD\d+%'),
    re.compile(r'^__'),
]


def is_garbage_label(label: str) -> bool:
    """Check if a label matches garbage patterns."""
    if not label:
        return False
    for pat in GARBAGE_PATTERNS:
        if pat.match(label):
            return True
    return False


def load_sync_state() -> dict:
    """Load sync state from file."""
    try:
        with open(SYNC_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "last_sync_time": None,
            "last_memory_id": 0,
            "last_connection_id": 0,
            "synced_memories": 0,
            "synced_connections": 0,
            "sync_errors": 0,
        }


def save_sync_state(state: dict):
    """Save sync state to file."""
    Path(SYNC_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    state["last_sync_time"] = datetime.now(timezone.utc).isoformat()
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_mssql(password=None):
    """Return a pyodbc connection to local MSSQL Mazemaker database."""
    try:
        import pyodbc
    except ImportError:
        print("pyodbc required: pip install pyodbc")
        sys.exit(1)

    pw = password or os.environ.get("MSSQL_PASSWORD", "")
    # Also try .env file
    if not pw:
        env_file = os.path.expanduser("~/.hermes/.env")
        if os.path.exists(env_file):
            for line in open(env_file):
                if line.startswith("MSSQL_PASSWORD="):
                    pw = line.split("=", 1)[1].strip().strip("\"'")
                    break

    if not pw:
        print("MSSQL_PASSWORD not found. Set env var or add to ~/.hermes/.env")
        sys.exit(1)

    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=localhost;DATABASE=Mazemaker;"
        f"UID=SA;PWD={pw};TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str, autocommit=True)


def unix_to_dt(ts):
    if not ts:
        return "1970-01-01 00:00:00.0000000"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def sync_memories(sqlite_db, mssql_conn, incremental=False, dry_run=False,
                  filter_garbage=False):
    """Sync memories table. Returns (synced, errors, skipped)."""
    sconn = sqlite3.connect(sqlite_db)
    sc = sconn.cursor()
    mc = mssql_conn.cursor()

    if incremental:
        state = load_sync_state()
        max_id = state.get("last_memory_id", 0)
        mc.execute("SELECT ISNULL(MAX(id), 0) FROM memories")
        mssql_max = mc.fetchone()[0]
        max_id = max(max_id, mssql_max)
        sc.execute(
            "SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count "
            "FROM memories WHERE id > ? ORDER BY id", (max_id,)
        )
    else:
        sc.execute(
            "SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count "
            "FROM memories ORDER BY id"
        )

    rows = sc.fetchall()
    sconn.close()

    if not rows:
        print("  No memories to sync.")
        return 0, 0, 0

    # Filter garbage if requested
    skipped = 0
    if filter_garbage:
        filtered = []
        for row in rows:
            label = row[1] or ""
            if is_garbage_label(label):
                skipped += 1
            else:
                filtered.append(row)
        rows = filtered

    if not rows:
        print(f"  No memories to sync ({skipped} garbage labels skipped).")
        return 0, 0, skipped

    print(f"  Memories to sync: {len(rows)}" +
          (f" ({skipped} garbage skipped)" if skipped else ""))

    if dry_run:
        print("  [DRY RUN] Would sync memories:")
        for row in rows[:5]:
            print(f"    ID={row[0]} label={row[1][:40]!r}")
        if len(rows) > 5:
            print(f"    ... and {len(rows)-5} more")
        return len(rows), 0, skipped

    if not incremental:
        mc.execute("ALTER TABLE connections NOCHECK CONSTRAINT ALL")
        mc.execute("DELETE FROM connections")
        mc.execute("DELETE FROM memories")
        mssql_conn.commit()

    mc.execute("SET IDENTITY_INSERT memories ON")
    mssql_conn.commit()

    synced, errors = 0, 0
    t0 = time.time()

    for i, row in enumerate(rows):
        id_, label, content, blob, salience, created, accessed, acc = row

        if blob:
            blob_len = len(blob)
            elem_count = blob_len // 4  # float32 = 4 bytes
            emb = list(struct.unpack(f"{elem_count}f", blob))
            emb_blob = struct.pack(f"{elem_count}f", *emb)
            emb_dim = elem_count
        else:
            emb_blob = None
            emb_dim = EMBEDDING_DIM

        try:
            mc.execute(
                "INSERT INTO memories (id, label, content, embedding, vector_dim, "
                "salience, created_at, last_accessed, access_count) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                id_, label, content, emb_blob, emb_dim,
                salience or 1.0, unix_to_dt(created), unix_to_dt(accessed), acc or 0,
            )
            synced += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Error memory {id_}: {e}")

        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            mssql_conn.commit()
            rate = (i + 1) / (time.time() - t0) if time.time() - t0 > 0 else 0
            print(f"    {i+1}/{len(rows)} ({rate:.0f}/s)")

    mc.execute("SET IDENTITY_INSERT memories OFF")
    mssql_conn.commit()

    return synced, errors, skipped


def sync_connections(sqlite_db, mssql_conn, incremental=False, dry_run=False):
    """Sync connections table. Returns (synced, errors)."""
    sconn = sqlite3.connect(sqlite_db)
    sc = sconn.cursor()
    mc = mssql_conn.cursor()

    if incremental:
        mc.execute("SELECT source_id, target_id FROM connections")
        mssql_set = set((r.source_id, r.target_id) for r in mc.fetchall())
        sc.execute("SELECT source_id, target_id, weight FROM connections ORDER BY id")
        missing = [(s, t, w) for s, t, w in sc.fetchall() if (s, t) not in mssql_set]
        sconn.close()
        if not missing:
            print("  No new connections to sync.")
            return 0, 0
        rows_to_insert = missing
        mc.execute("SELECT ISNULL(MAX(id), 0) FROM connections")
        next_id = mc.fetchone()[0] + 1
    else:
        sc.execute("SELECT source_id, target_id, weight FROM connections ORDER BY id")
        rows_to_insert = sc.fetchall()
        sconn.close()
        next_id = 1

    print(f"  Connections to sync: {len(rows_to_insert)}")

    if dry_run:
        print("  [DRY RUN] Would sync connections:")
        for s, t, w in rows_to_insert[:5]:
            print(f"    {s} -> {t} (w={w:.4f})")
        if len(rows_to_insert) > 5:
            print(f"    ... and {len(rows_to_insert)-5} more")
        return len(rows_to_insert), 0

    mc.execute("ALTER TABLE connections NOCHECK CONSTRAINT ALL")
    if not incremental:
        mc.execute("DELETE FROM connections")
    mc.execute("SET IDENTITY_INSERT connections ON")
    mssql_conn.commit()

    synced, errors = 0, 0
    t0 = time.time()
    batch = 1000

    for i in range(0, len(rows_to_insert), batch):
        chunk = rows_to_insert[i:i + batch]
        data = [
            (next_id + j, s, t, round(w, 6), "similar")
            for j, (s, t, w) in enumerate(chunk)
        ]
        try:
            mc.fast_executemany = True
            mc.executemany(
                "INSERT INTO connections (id, source_id, target_id, weight, edge_type) "
                "VALUES (?,?,?,?,?)",
                data,
            )
            mssql_conn.commit()
            synced += len(chunk)
            next_id += len(chunk)
        except Exception as ex:
            print(f"    Batch error at {i}: {ex}")
            mssql_conn.rollback()
            for row in data:
                try:
                    mc.execute(
                        "INSERT INTO connections (id, source_id, target_id, weight, edge_type) "
                        "VALUES (?,?,?,?,?)", *row
                    )
                    synced += 1
                    next_id += 1
                except Exception:
                    errors += 1
            mssql_conn.commit()

        done = min(i + batch, len(rows_to_insert))
        rate = done / (time.time() - t0) if time.time() - t0 > 0 else 0
        if done % 5000 == 0 or done >= len(rows_to_insert):
            print(f"    {done}/{len(rows_to_insert)} ({rate:.0f}/s)")

    mc.execute("SET IDENTITY_INSERT connections OFF")
    mc.execute("ALTER TABLE connections WITH CHECK CHECK CONSTRAINT ALL")
    mssql_conn.commit()

    return synced, errors


def verify(sqlite_db, mssql_conn):
    """Print comparison of SQLite vs MSSQL counts."""
    sconn = sqlite3.connect(sqlite_db)
    sc = sconn.cursor()
    sm = sc.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    sc2 = sc.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
    sconn.close()

    mc = mssql_conn.cursor()
    mm = mc.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    mc2 = mc.execute("SELECT COUNT(*) FROM connections").fetchone()[0]

    print(f"\n{'='*50}")
    print(f"  SQLite:   {sm:>5} memories, {sc2:>6} connections")
    print(f"  MSSQL:    {mm:>5} memories, {mc2:>6} connections")
    if sm == mm and sc2 == mc2:
        print(f"  Status:   SYNCED ✓")
    else:
        mem_diff = sm - mm
        conn_diff = sc2 - mc2
        print(f"  Status:   PARTIAL (diff: {mem_diff:+d} mem, {conn_diff:+d} conn)")
    print(f"{'='*50}")
    return sm == mm and sc2 == mc2


def main():
    parser = argparse.ArgumentParser(description="Sync Mazemaker SQLite → MSSQL")
    parser.add_argument("--db", default=DEFAULT_SQLITE, help="SQLite database path")
    parser.add_argument("--incremental", action="store_true", help="Only sync new rows")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--skip-connections", action="store_true", help="Skip connections")
    parser.add_argument("--filter-garbage", action="store_true",
                        help="Skip turn-%, DD% labels during sync")
    parser.add_argument("--password", default=None, help="MSSQL SA password")
    args = parser.parse_args()

    print("=== Mazemaker SQLite → MSSQL Sync ===\n")
    if args.dry_run:
        print("  [DRY RUN MODE] No changes will be made.\n")

    print(f"[1/3] SQLite: {args.db}")
    mconn = get_mssql(args.password)

    print("[2/3] Syncing memories...")
    t0 = time.time()
    ms, me, msk = sync_memories(args.db, mconn, args.incremental,
                                 args.dry_run, args.filter_garbage)
    print(f"  Done: {ms} synced, {me} errors, {msk} skipped ({time.time()-t0:.1f}s)\n")

    cs, ce = 0, 0
    if not args.skip_connections:
        print("[3/3] Syncing connections...")
        t1 = time.time()
        cs, ce = sync_connections(args.db, mconn, args.incremental, args.dry_run)
        print(f"  Done: {cs} synced, {ce} errors ({time.time()-t1:.1f}s)")
    else:
        print("[3/3] Skipped connections")

    if not args.dry_run:
        # Clean orphans and re-enable FK constraints
        mc = mconn.cursor()
        mc.execute("ALTER TABLE connections NOCHECK CONSTRAINT ALL")
        mc.execute("""
            DELETE FROM connections
            WHERE source_id NOT IN (SELECT id FROM memories)
               OR target_id NOT IN (SELECT id FROM memories)
        """)
        orphans = mc.rowcount
        if orphans:
            print(f"  Cleaned {orphans} orphaned connections")
        mc.execute("ALTER TABLE connections WITH CHECK CHECK CONSTRAINT ALL")
        mconn.commit()

        # Save sync state
        state = load_sync_state()
        state["synced_memories"] = ms
        state["synced_connections"] = cs
        state["sync_errors"] = me + ce
        # Update last memory ID
        sconn = sqlite3.connect(args.db)
        last_id = sconn.execute("SELECT MAX(id) FROM memories").fetchone()[0] or 0
        sconn.close()
        state["last_memory_id"] = last_id
        save_sync_state(state)

        verify(args.db, mconn)
    else:
        print("\n  [DRY RUN] No changes made.")

    mconn.close()


if __name__ == "__main__":
    main()
