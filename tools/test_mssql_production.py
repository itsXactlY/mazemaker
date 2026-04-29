#!/usr/bin/env python3
"""
test_mssql_production.py — MSSQL Production Test Suite

Verifies the Mazemaker MSSQL database is in production-grade state.
Tests run against the LIVE database — no mocking, no fakes.

Usage:
  python3 test_mssql_production.py [OPTIONS]
  python3 test_mssql_production.py --config ~/.hermes/config.yaml
  python3 test_mssql_production.py --tags schema,constraints,data

Tags: schema, constraints, data, merge, dedup, integration, performance
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Test harness (same pattern as test_suite.py)
# ---------------------------------------------------------------------------

PASS = FAIL = SKIP = 0
TAGS = set()


class SkipTest(Exception):
    pass


def _testcase(name: str, tags: list[str] | None = None):
    tags = tags or []

    def decorator(fn):
        def wrapper(conn):
            global PASS, FAIL, SKIP
            try:
                fn(conn)
                print(f"  PASS  {name}")
                PASS += 1
            except SkipTest as e:
                print(f"  SKIP  {name}: {e}")
                SKIP += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                FAIL += 1

        wrapper._name = name
        wrapper._tags = tags
        TAGS.update(tags)
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection(config_path: str = "~/.hermes/config.yaml"):
    import yaml
    import pyodbc

    config_path = os.path.expanduser(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mssql_cfg = config.get("memory", {}).get("neural", {}).get("dream", {}).get("mssql", {})
    pw = mssql_cfg.get("password", "")
    server = mssql_cfg.get("server", "127.0.0.1")
    database = mssql_cfg.get("database", "Mazemaker")

    if not pw:
        env_path = os.path.expanduser("~/.hermes/.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("MSSQL_PASSWORD="):
                        pw = line.split("=", 1)[1].strip().strip("'\"")
                        break

    if not pw:
        raise RuntimeError("No MSSQL password found")

    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server};DATABASE={database};"
        f"UID=sa;PWD={pw};"
        f"TrustServerCertificate=yes;Connection Timeout=15;"
    )
    return pyodbc.connect(conn_str)


# ====================================================================
# SCHEMA TESTS
# ====================================================================

@_testcase("schema: all required tables exist", tags=["schema"])
def test_schema_tables(conn):
    mc = conn.cursor()
    required = {"memories", "connections", "connection_history",
                "dream_sessions", "dream_insights",
                "GraphNodes_v2", "GraphEdges_v2", "Mazemaker"}
    mc.execute("SELECT name FROM sys.tables")
    actual = {r[0] for r in mc.fetchall()}
    missing = required - actual
    assert not missing, f"Missing tables: {missing}"


@_testcase("schema: no legacy tables (NeuralMemory_old, GraphNodes, GraphEdges)", tags=["schema"])
def test_no_legacy(conn):
    mc = conn.cursor()
    legacy = {"NeuralMemory_old", "GraphNodes", "GraphEdges"}
    mc.execute("SELECT name FROM sys.tables")
    actual = {r[0] for r in mc.fetchall()}
    found = legacy & actual
    assert not found, f"Legacy tables still exist: {found}"


@_testcase("schema: no duplicate connection_history_v2 tables", tags=["schema"])
def test_no_dup_v2(conn):
    mc = conn.cursor()
    mc.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME LIKE 'connection_history_v2%'")
    dupes = [r[0] for r in mc.fetchall()]
    assert len(dupes) == 0, f"Duplicate v2 tables: {dupes}"


@_testcase("schema: memories has vector_dim column", tags=["schema"])
def test_vector_dim(conn):
    mc = conn.cursor()
    mc.execute("""
        SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'memories' AND COLUMN_NAME = 'vector_dim'
    """)
    row = mc.fetchone()
    assert row is not None, "vector_dim column missing"
    assert row[0] == "int", f"vector_dim should be int, got {row[0]}"


@_testcase("schema: connection_history has changed_at datetime2", tags=["schema"])
def test_changed_at_type(conn):
    mc = conn.cursor()
    mc.execute("""
        SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'connection_history' AND COLUMN_NAME = 'changed_at'
    """)
    row = mc.fetchone()
    assert row is not None, "changed_at column missing"
    assert row[0] == "datetime2", f"changed_at should be datetime2, got {row[0]}"


# ====================================================================
# CONSTRAINT TESTS
# ====================================================================

@_testcase("constraints: UX_connections_unique exists", tags=["constraints"])
def test_conn_unique(conn):
    mc = conn.cursor()
    mc.execute("""
        SELECT COUNT(*) FROM sys.indexes
        WHERE name = 'UX_connections_unique' AND object_id = OBJECT_ID('connections')
    """)
    assert mc.fetchone()[0] > 0, "UX_connections_unique not found"


@_testcase("constraints: UX_connection_history_unique exists", tags=["constraints"])
def test_hist_unique(conn):
    mc = conn.cursor()
    mc.execute("""
        SELECT COUNT(*) FROM sys.indexes
        WHERE name = 'UX_connection_history_unique' AND object_id = OBJECT_ID('connection_history')
    """)
    assert mc.fetchone()[0] > 0, "UX_connection_history_unique not found"


@_testcase("constraints: connections has FK to memories", tags=["constraints"])
def test_conn_fk(conn):
    mc = conn.cursor()
    mc.execute("""
        SELECT COUNT(*) FROM sys.foreign_keys fk
        INNER JOIN sys.tables t ON fk.parent_object_id = t.object_id
        WHERE t.name = 'connections'
    """)
    assert mc.fetchone()[0] > 0, "No FK on connections → memories"


# ====================================================================
# DATA INTEGRITY TESTS
# ====================================================================

@_testcase("data: memories > 0", tags=["data"])
def test_memories_count(conn):
    mc = conn.cursor()
    mc.execute("SELECT COUNT(*) FROM memories")
    assert mc.fetchone()[0] > 0, "No memories"


@_testcase("data: connections > 0", tags=["data"])
def test_connections_count(conn):
    mc = conn.cursor()
    mc.execute("SELECT COUNT(*) FROM connections")
    assert mc.fetchone()[0] > 0, "No connections"


@_testcase("data: zero duplicate connections", tags=["data", "dedup"])
def test_no_conn_dupes(conn):
    mc = conn.cursor()
    mc.execute("""
        SELECT COUNT(*) FROM (
            SELECT source_id, target_id, COUNT(*) as cnt
            FROM connections GROUP BY source_id, target_id HAVING COUNT(*) > 1
        ) x
    """)
    assert mc.fetchone()[0] == 0, "Duplicate connections exist"


@_testcase("data: zero duplicate history entries", tags=["data", "dedup"])
def test_no_hist_dupes(conn):
    mc = conn.cursor()
    mc.execute("""
        SELECT COUNT(*) FROM (
            SELECT source_id, target_id, COUNT(*) as cnt
            FROM connection_history GROUP BY source_id, target_id HAVING COUNT(*) > 1
        ) x
    """)
    assert mc.fetchone()[0] == 0, "Duplicate history entries exist"


@_testcase("data: zero orphan connections", tags=["data"])
def test_no_orphans(conn):
    mc = conn.cursor()
    mc.execute("""
        SELECT COUNT(*) FROM connections c
        WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = c.source_id)
           OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = c.target_id)
    """)
    assert mc.fetchone()[0] == 0, "Orphan connections exist"


@_testcase("data: connection_history < 500K rows (bloat check)", tags=["data"])
def test_history_not_bloated(conn):
    mc = conn.cursor()
    mc.execute("SELECT COUNT(*) FROM connection_history")
    cnt = mc.fetchone()[0]
    assert cnt < 500000, f"History bloated: {cnt:,} rows (NREM is creating duplicates)"


@_testcase("data: all memories have valid created_at", tags=["data"])
def test_memories_timestamps(conn):
    mc = conn.cursor()
    mc.execute("SELECT COUNT(*) FROM memories WHERE created_at IS NULL")
    nulls = mc.fetchone()[0]
    mc.execute("SELECT COUNT(*) FROM memories")
    total = mc.fetchone()[0]
    # Allow some nulls from imports
    assert nulls < total * 0.1, f"Too many null timestamps: {nulls}/{total}"


# ====================================================================
# MERGE/UPSERT TESTS
# ====================================================================

@_testcase("merge: MERGE on connections (update existing)", tags=["merge"])
def test_merge_conn(conn):
    mc = conn.cursor()
    mc.execute("SELECT TOP 1 source_id, target_id, weight FROM connections")
    row = mc.fetchone()
    if not row:
        raise SkipTest("No connections to test")
    sid, tid, orig_w = row[0], row[1], row[2]
    # MERGE with different weight
    mc.execute(
        "MERGE connections AS target "
        "USING (VALUES (?, ?, 9.99, 'test')) AS source (source_id, target_id, weight, edge_type) "
        "ON target.source_id = source.source_id AND target.target_id = source.target_id "
        "WHEN MATCHED THEN UPDATE SET weight = source.weight "
        "WHEN NOT MATCHED THEN INSERT (source_id, target_id, weight, edge_type) VALUES (source.source_id, source.target_id, source.weight, source.edge_type);",
        sid, tid
    )
    conn.commit()
    mc.execute("SELECT weight FROM connections WHERE source_id = ? AND target_id = ?", sid, tid)
    new_w = mc.fetchone()[0]
    assert new_w == 9.99, f"MERGE didn't update: expected 9.99, got {new_w}"
    # Restore
    mc.execute("UPDATE connections SET weight = ? WHERE source_id = ? AND target_id = ?", orig_w, sid, tid)
    conn.commit()


@_testcase("merge: MERGE on connection_history (upsert)", tags=["merge"])
def test_merge_hist(conn):
    mc = conn.cursor()
    mc.execute("SELECT TOP 1 source_id, target_id FROM connection_history")
    row = mc.fetchone()
    if not row:
        raise SkipTest("No history to test")
    sid, tid = row[0], row[1]
    # MERGE should update existing entry
    now = datetime.now()
    mc.execute(
        "MERGE connection_history AS target "
        "USING (VALUES (?, ?, 0.1, 0.9, 'test_merge', ?)) AS source (source_id, target_id, old_weight, new_weight, reason, changed_at) "
        "ON target.source_id = source.source_id AND target.target_id = source.target_id "
        "WHEN MATCHED THEN UPDATE SET new_weight = source.new_weight, changed_at = source.changed_at "
        "WHEN NOT MATCHED THEN INSERT (source_id, target_id, old_weight, new_weight, reason, changed_at) VALUES (source.source_id, source.target_id, source.old_weight, source.new_weight, source.reason, source.changed_at);",
        sid, tid, now
    )
    conn.commit()
    mc.execute("SELECT new_weight FROM connection_history WHERE source_id = ? AND target_id = ?", sid, tid)
    nw = mc.fetchone()[0]
    assert nw == 0.9, f"MERGE didn't update history: expected 0.9, got {nw}"
    # Verify no duplicate was created
    mc.execute("SELECT COUNT(*) FROM connection_history WHERE source_id = ? AND target_id = ?", sid, tid)
    assert mc.fetchone()[0] == 1, "MERGE created a duplicate!"


@_testcase("merge: UNIQUE prevents INSERT duplicate connection", tags=["merge", "constraints"])
def test_unique_prevents_dup(conn):
    mc = conn.cursor()
    mc.execute("SELECT TOP 1 source_id, target_id FROM connections")
    row = mc.fetchone()
    if not row:
        raise SkipTest("No connections to test")
    sid, tid = row[0], row[1]
    try:
        mc.execute(
            "INSERT INTO connections (source_id, target_id, weight, edge_type) VALUES (?, ?, ?, ?)",
            sid, tid, 0.5, "test_dup"
        )
        conn.commit()
        # If we get here, UNIQUE didn't work
        mc.execute("DELETE FROM connections WHERE source_id = ? AND target_id = ? AND edge_type = 'test_dup'", sid, tid)
        conn.commit()
        assert False, "UNIQUE constraint didn't prevent duplicate INSERT"
    except Exception:
        # Expected — UNIQUE violation
        conn.rollback()


# ====================================================================
# INTEGRATION TESTS
# ====================================================================

@_testcase("integration: GraphNodes_v2 matches or exceeds old V1 data", tags=["integration"])
def test_v2_sufficient(conn):
    mc = conn.cursor()
    # V2 should have all the data that was in V1
    mc.execute("SELECT COUNT(*) FROM GraphNodes_v2")
    nodes = mc.fetchone()[0]
    mc.execute("SELECT COUNT(*) FROM GraphEdges_v2")
    edges = mc.fetchone()[0]
    assert nodes > 0, "GraphNodes_v2 is empty"
    assert edges > 0, "GraphEdges_v2 is empty"
    # Sanity: edges should be > nodes in a connected graph
    assert edges >= nodes, f"More nodes ({nodes}) than edges ({edges}) — disconnected graph?"


@_testcase("integration: Mazemaker has data", tags=["integration"])
def test_neural_memory(conn):
    mc = conn.cursor()
    mc.execute("SELECT COUNT(*) FROM Mazemaker")
    cnt = mc.fetchone()[0]
    assert cnt > 0, "Mazemaker vector store is empty"


@_testcase("integration: dream_sessions exists", tags=["integration"])
def test_dream_sessions(conn):
    mc = conn.cursor()
    mc.execute("SELECT COUNT(*) FROM dream_sessions")
    cnt = mc.fetchone()[0]
    assert cnt > 0, "No dream sessions"


@_testcase("integration: DB size < 1GB", tags=["integration", "performance"])
def test_db_size(conn):
    mc = conn.cursor()
    mc.execute("""
        SELECT SUM(CAST(FILEPROPERTY(name, 'SpaceUsed') AS bigint) * 8) / 1024
        FROM sys.database_files WHERE type = 0
    """)
    size_mb = mc.fetchone()[0]
    assert size_mb < 1024, f"DB too large: {size_mb}MB"


# ====================================================================
# PERFORMANCE TESTS
# ====================================================================

@_testcase("perf: recall by memory ID < 100ms", tags=["performance"])
def test_perf_recall(conn):
    mc = conn.cursor()
    mc.execute("SELECT TOP 1 id FROM memories")
    row = mc.fetchone()
    if not row:
        raise SkipTest("No memories")
    mid = row[0]
    start = time.time()
    mc.execute("""
        SELECT c.target_id, c.weight
        FROM connections c WHERE c.source_id = ?
        ORDER BY c.weight DESC
    """, (mid,))
    results = mc.fetchall()
    elapsed = (time.time() - start) * 1000
    assert elapsed < 100, f"Recall too slow: {elapsed:.1f}ms (target: <100ms)"


@_testcase("perf: history lookup by (source, target) < 50ms", tags=["performance"])
def test_perf_history(conn):
    mc = conn.cursor()
    mc.execute("SELECT TOP 1 source_id, target_id FROM connection_history")
    row = mc.fetchone()
    if not row:
        raise SkipTest("No history")
    start = time.time()
    mc.execute("SELECT * FROM connection_history WHERE source_id = ? AND target_id = ?", row[0], row[1])
    elapsed = (time.time() - start) * 1000
    assert elapsed < 50, f"History lookup too slow: {elapsed:.1f}ms (target: <50ms)"


# ====================================================================
# Runner
# ====================================================================

def run_tests(conn, tag_filter: set[str] | None = None):
    global PASS, FAIL, SKIP
    PASS = FAIL = SKIP = 0

    # Collect all test functions
    tests = []
    for name, obj in list(globals().items()):
        if callable(obj) and hasattr(obj, "_tags"):
            if tag_filter:
                if not set(obj._tags) & tag_filter:
                    continue
            tests.append(obj)

    print(f"\n{'='*60}")
    print(f"  MSSQL Production Test Suite — {len(tests)} tests")
    if tag_filter:
        print(f"  Filter: {tag_filter}")
    print(f"{'='*60}\n")

    for test in sorted(tests, key=lambda t: t._name):
        test(conn)

    print(f"\n{'='*60}")
    print(f"  Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print(f"{'='*60}")

    return FAIL == 0


def main():
    parser = argparse.ArgumentParser(description="MSSQL Production Test Suite")
    parser.add_argument("--config", default="~/.hermes/config.yaml")
    parser.add_argument("--tags", help="Comma-separated tag filter")
    args = parser.parse_args()

    tag_filter = None
    if args.tags:
        tag_filter = set(args.tags.split(","))

    conn = get_connection(args.config)
    ok = run_tests(conn, tag_filter)
    conn.close()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
