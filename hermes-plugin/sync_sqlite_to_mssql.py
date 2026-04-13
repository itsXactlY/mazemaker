#!/usr/bin/env python3
"""
sync_sqlite_to_mssql.py - Full sync from SQLite neural memory to MSSQL
Preserves original SQLite IDs via IDENTITY_INSERT.
Usage: python sync_sqlite_to_mssql.py [--batch-size N] [--skip-connections]
"""
import sqlite3
import struct
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mssql_store import MSSQLStore

SQLITE_DB = os.path.expanduser("~/.neural_memory/memory.db")
EMBEDDING_DIM = 384


def read_sqlite_memories(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count FROM memories ORDER BY id")
    rows = cur.fetchall()
    memories = []
    for row in rows:
        id_, label, content, blob, salience, created_at, last_accessed, access_count = row
        embedding = list(struct.unpack(f'{EMBEDDING_DIM}f', blob)) if blob else None
        memories.append({
            'id': id_, 'label': label, 'content': content,
            'embedding': embedding, 'salience': salience or 1.0,
            'created_at': created_at, 'last_accessed': last_accessed,
            'access_count': access_count or 0,
        })
    conn.close()
    return memories


def read_sqlite_connections(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT source_id, target_id, weight, edge_type FROM connections ORDER BY id")
    rows = cur.fetchall()
    connections = [{'source_id': r[0], 'target_id': r[1], 'weight': r[2], 'edge_type': r[3] or 'similar'} for r in rows]
    conn.close()
    return connections


def unix_to_datetime2(unix_ts: float) -> str:
    if not unix_ts:
        return '1970-01-01 00:00:00.000'
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')


def sync_memories(store: MSSQLStore, memories: list[dict], batch_size: int = 200):
    """Bulk insert memories with IDENTITY_INSERT."""
    cursor = store.conn.cursor()
    cursor.execute("SET NOCOUNT ON")
    cursor.execute("SET IDENTITY_INSERT memories ON")
    store.conn.commit()

    # Truncate for clean sync - connections first (FK constraints)
    cursor.execute("ALTER TABLE connections NOCHECK CONSTRAINT ALL")
    cursor.execute("DELETE FROM connections")
    cursor.execute("DELETE FROM memories")
    store.conn.commit()

    total = len(memories)
    synced = 0
    errors = 0
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch = memories[i:i + batch_size]
        # Build multi-row INSERT
        values = []
        params = []
        for mem in batch:
            blob = struct.pack(f'{EMBEDDING_DIM}f', *mem['embedding']) if mem['embedding'] else None
            created = unix_to_datetime2(mem['created_at'])
            accessed = unix_to_datetime2(mem['last_accessed'])
            values.append("(?, ?, ?, ?, ?, ?, ?, ?, ?)")
            params.extend([mem['id'], mem['label'], mem['content'], blob, EMBEDDING_DIM,
                          mem['salience'], created, accessed, mem['access_count']])

        sql = f"INSERT INTO memories (id, label, content, embedding, vector_dim, salience, created_at, last_accessed, access_count) VALUES {', '.join(values)}"
        try:
            cursor.execute(sql, *params)
            store.conn.commit()
            synced += len(batch)
        except Exception as e:
            errors += 1
            print(f"  Batch error at {i}: {e}")
            # Fall back to row-by-row
            for mem in batch:
                try:
                    blob = struct.pack(f'{EMBEDDING_DIM}f', *mem['embedding']) if mem['embedding'] else None
                    cursor.execute(
                        "INSERT INTO memories (id, label, content, embedding, vector_dim, salience, created_at, last_accessed, access_count) VALUES (?,?,?,?,?,?,?,?,?)",
                        mem['id'], mem['label'], mem['content'], blob, EMBEDDING_DIM,
                        mem['salience'], unix_to_datetime2(mem['created_at']),
                        unix_to_datetime2(mem['last_accessed']), mem['access_count']
                    )
                    synced += 1
                except Exception as e2:
                    errors += 1
                    if errors <= 3:
                        print(f"    Row error {mem['id']}: {e2}")
            store.conn.commit()

        elapsed = time.time() - t0
        rate = synced / elapsed if elapsed > 0 else 0
        print(f"  Memories: {min(i+batch_size, total)}/{total} ({rate:.0f}/s)")

    cursor.execute("SET IDENTITY_INSERT memories OFF")
    # Re-enable FK constraints after both tables are populated
    # (done at the end of sync_connections or here if skipping connections)
    store.conn.commit()
    return synced, errors


def sync_connections(store: MSSQLStore, connections: list[dict], batch_size: int = 1000):
    """Bulk insert connections with IDENTITY_INSERT."""
    cursor = store.conn.cursor()
    cursor.execute("SET NOCOUNT ON")

    # Can't use IDENTITY_INSERT with FK without memories being present
    # Just disable FK checks, insert, re-enable
    cursor.execute("ALTER TABLE connections NOCHECK CONSTRAINT ALL")
    cursor.execute("SET IDENTITY_INSERT connections ON")
    cursor.execute("DELETE FROM connections")
    store.conn.commit()

    total = len(connections)
    synced = 0
    errors = 0
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch = connections[i:i + batch_size]
        values = []
        params = []
        for j, conn in enumerate(batch):
            conn_id = i + j + 1
            values.append("(?, ?, ?, ?, ?)")
            params.extend([conn_id, conn['source_id'], conn['target_id'], conn['weight'], conn['edge_type']])

        sql = f"INSERT INTO connections (id, source_id, target_id, weight, edge_type) VALUES {', '.join(values)}"
        try:
            cursor.execute(sql, *params)
            store.conn.commit()
            synced += len(batch)
        except Exception as e:
            errors += 1
            print(f"  Batch error at {i}: {e}")
            store.conn.rollback()
            # Row by row fallback
            for j, conn in enumerate(batch):
                try:
                    cursor.execute(
                        "INSERT INTO connections (id, source_id, target_id, weight, edge_type) VALUES (?,?,?,?,?)",
                        i+j+1, conn['source_id'], conn['target_id'], conn['weight'], conn['edge_type']
                    )
                    synced += 1
                except:
                    errors += 1
            store.conn.commit()

        if (i + batch_size) % 5000 == 0 or i + batch_size >= total:
            elapsed = time.time() - t0
            rate = synced / elapsed if elapsed > 0 else 0
            print(f"  Connections: {min(i+batch_size, total)}/{total} ({rate:.0f}/s)")

    cursor.execute("SET IDENTITY_INSERT connections OFF")
    cursor.execute("ALTER TABLE connections WITH CHECK CHECK CONSTRAINT ALL")
    store.conn.commit()
    return synced, errors


def main():
    parser = argparse.ArgumentParser(description="Sync Neural Memory SQLite -> MSSQL")
    parser.add_argument('--batch-size', type=int, default=200, help='Batch size for memory inserts')
    parser.add_argument('--skip-connections', action='store_true', help='Skip syncing connections')
    args = parser.parse_args()

    print("=== Neural Memory SQLite -> MSSQL Sync ===\n")

    print(f"[1/4] Reading SQLite: {SQLITE_DB}")
    memories = read_sqlite_memories(SQLITE_DB)
    connections = read_sqlite_connections(SQLITE_DB)
    print(f"  Found {len(memories)} memories, {len(connections)} connections\n")

    print("[2/4] Connecting to MSSQL...")
    store = MSSQLStore()
    pre = store.stats()
    print(f"  Current MSSQL: {pre['memories']} memories, {pre['connections']} connections\n")

    print("[3/4] Syncing memories...")
    t0 = time.time()
    mem_synced, mem_errors = sync_memories(store, memories, args.batch_size)
    t1 = time.time()
    print(f"  Done: {mem_synced} synced, {mem_errors} errors ({t1-t0:.1f}s)\n")

    if not args.skip_connections:
        print("[4/4] Syncing connections...")
        t2 = time.time()
        conn_synced, conn_errors = sync_connections(store, connections)
        t3 = time.time()
        print(f"  Done: {conn_synced} synced, {conn_errors} errors ({t3-t2:.1f}s)\n")
    else:
        # Re-enable FK constraints if skipping connections
        cursor = store.conn.cursor()
        cursor.execute("ALTER TABLE connections WITH CHECK CHECK CONSTRAINT ALL")
        store.conn.commit()
        print("[4/4] Skipped connections\n")

    post = store.stats()
    print(f"=== RESULT ===")
    print(f"  MSSQL: {post['memories']} memories, {post['connections']} connections")
    print(f"  SQLite: {len(memories)} memories, {len(connections)} connections")
    print(f"  Total: {time.time()-t0:.1f}s")

    store.close()


if __name__ == "__main__":
    main()
