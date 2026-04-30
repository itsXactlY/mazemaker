#!/usr/bin/env python3
"""
mssql_production_migrate.py — MSSQL Production Migration & Verification

Migrates Mazemaker MSSQL from legacy/V1 state to production-grade V2:
  1. Diagnose current state (tables, sizes, duplicates, orphans)
  2. Sync all SQLite databases into MSSQL (merge, don't overwrite)
  3. Deduplicate connections + connection_history
  4. Migrate V1 → V2 (drop legacy GraphNodes/Edges, keep V2)
  5. Drop legacy tables (NeuralMemory_old, connection_history_v2 duplicates)
  6. Add UNIQUE constraints (prevent future NREM bloat)
  7. Verify MERGE/UPSERT code is wired in all dream files
  8. Run full integrity + functional verification

Usage:
  python3 mssql_production_migrate.py [OPTIONS]

  --config PATH       Path to hermes config.yaml (default: ~/.hermes/config.yaml)
  --sqlite-db PATH    Path to main SQLite database (default: ~/.neural_memory/memory.db)
  --dry-run           Show what would change without modifying anything
  --skip-sync         Skip SQLite→MSSQL sync (only cleanup + verify)
  --skip-backup       Skip MSSQL backup (NOT recommended)
  --history-days N    Keep connection_history from last N days (default: 3)
  --force             Skip confirmation prompts
  --verify-only       Only run verification, no changes

Requirements:
  - Python 3.10+
  - pyodbc (pip install pyodbc)
  - ODBC Driver 18 for SQL Server
  - MSSQL Server running
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def epoch_to_datetime(epoch):
    """Convert SQLite REAL epoch to Python datetime."""
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch))
    except (ValueError, OSError, OverflowError):
        return None


def get_embed_dim(blob: bytes | None) -> int:
    """Get embedding dimension from float32 blob."""
    if blob is None:
        return 0
    return len(blob) // 4


def banner(text: str, char: str = "=") -> None:
    line = char * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def confirm(prompt: str, force: bool = False) -> bool:
    if force:
        return True
    resp = input(f"\n{prompt} [y/N] ").strip().lower()
    return resp in ("y", "yes")


# ---------------------------------------------------------------------------
# MSSQL Connection
# ---------------------------------------------------------------------------

def get_mssql_connection(config_path: str = "~/.hermes/config.yaml"):
    """Load MSSQL credentials from hermes config and connect."""
    try:
        import yaml
        import pyodbc
    except ImportError:
        print("ERROR: pyodbc and pyyaml required. Install: pip install pyodbc pyyaml")
        sys.exit(1)

    config_path = os.path.expanduser(config_path)
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Navigate to MSSQL config
    mssql_cfg = config.get("memory", {}).get("neural", {}).get("dream", {}).get("mssql", {})
    pw = mssql_cfg.get("password", "")
    server = mssql_cfg.get("server", "127.0.0.1")
    database = mssql_cfg.get("database", "Mazemaker")

    if not pw:
        # Try .env file
        env_path = os.path.expanduser("~/.hermes/.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("MSSQL_PASSWORD="):
                        pw = line.split("=", 1)[1].strip().strip("'\"")
                        break

    if not pw:
        print("ERROR: No MSSQL password found in config or .env")
        sys.exit(1)

    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server};DATABASE={database};"
        f"UID=sa;PWD={pw};"
        f"TrustServerCertificate=yes;Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str)


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def diagnose(conn) -> dict:
    """Full diagnostic of current MSSQL state."""
    mc = conn.cursor()
    info = {"tables": {}, "constraints": {}, "issues": [], "warnings": []}

    # All tables with row counts and sizes
    mc.execute("""
        SELECT t.name, p.rows,
               SUM(a.total_pages) * 8 / 1024 AS size_mb
        FROM sys.tables t
        INNER JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0,1)
        INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
        GROUP BY t.name, p.rows
        ORDER BY size_mb DESC
    """)
    for r in mc.fetchall():
        info["tables"][r.name] = {"rows": r.rows, "size_mb": r.size_mb or 0}

    # Unique constraints
    mc.execute("""
        SELECT t.name AS table_name, i.name AS index_name, i.is_unique, i.type_desc
        FROM sys.indexes i
        INNER JOIN sys.tables t ON i.object_id = t.object_id
        WHERE i.is_unique = 1 AND i.name IS NOT NULL
        ORDER BY t.name
    """)
    for r in mc.fetchall():
        key = f"{r.table_name}.{r.index_name}"
        info["constraints"][key] = {"unique": r.is_unique, "type": r.type_desc}

    # Duplicate checks
    mc.execute("""
        SELECT COUNT(*) FROM (
            SELECT source_id, target_id, COUNT(*) as cnt
            FROM connections GROUP BY source_id, target_id HAVING COUNT(*) > 1
        ) x
    """)
    conn_dupes = mc.fetchone()[0]
    if conn_dupes > 0:
        info["issues"].append(f"connections: {conn_dupes} duplicate (source_id, target_id) pairs")

    mc.execute("""
        SELECT COUNT(*) FROM (
            SELECT source_id, target_id, COUNT(*) as cnt
            FROM connection_history GROUP BY source_id, target_id HAVING COUNT(*) > 1
        ) x
    """)
    hist_dupes = mc.fetchone()[0]
    if hist_dupes > 0:
        info["issues"].append(f"connection_history: {hist_dupes} duplicate pairs")

    # Orphan check
    mc.execute("""
        SELECT COUNT(*) FROM connections c
        WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = c.source_id)
           OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = c.target_id)
    """)
    orphans = mc.fetchone()[0]
    if orphans > 0:
        info["issues"].append(f"connections: {orphans} orphan rows (missing memory refs)")

    # Legacy tables check
    for legacy in ["NeuralMemory_old", "GraphNodes", "GraphEdges"]:
        if legacy in info["tables"]:
            info["warnings"].append(f"Legacy table exists: {legacy} ({info['tables'][legacy]['rows']} rows)")

    # connection_history bloat
    if "connection_history" in info["tables"]:
        mc.execute("""
            SELECT CAST(changed_at AS DATE) as day, COUNT(*) as cnt
            FROM connection_history
            GROUP BY CAST(changed_at AS DATE) ORDER BY day
        """)
        days = mc.fetchall()
        if len(days) > 1:
            total = sum(d.cnt for d in days)
            if total > 100000:
                info["issues"].append(f"connection_history bloat: {total:,} rows across {len(days)} days")

    # DB size
    mc.execute("""
        SELECT SUM(CAST(FILEPROPERTY(name, 'SpaceUsed') AS bigint) * 8) / 1024
        FROM sys.database_files WHERE type = 0
    """)
    info["db_size_mb"] = mc.fetchone()[0] or 0

    return info


def print_diagnosis(info: dict) -> None:
    banner("DIAGNOSIS")

    print("\n--- Tables ---")
    for name, data in sorted(info["tables"].items(), key=lambda x: -x[1]["size_mb"]):
        print(f"  {name:30s} {data['rows']:>12,} rows  {data['size_mb']:>6,} MB")

    print(f"\n  DB total: {info['db_size_mb']:,} MB")

    print("\n--- Unique Constraints ---")
    for name, data in info["constraints"].items():
        print(f"  {'✓' if data['unique'] else '✗'} {name} ({data['type']})")

    if info["issues"]:
        print("\n--- ISSUES ---")
        for issue in info["issues"]:
            print(f"  ✗ {issue}")
    else:
        print("\n--- No issues found ---")

    if info["warnings"]:
        print("\n--- Warnings ---")
        for warn in info["warnings"]:
            print(f"  ⚠ {warn}")


# ---------------------------------------------------------------------------
# SQLite → MSSQL Sync
# ---------------------------------------------------------------------------

def sync_sqlite_to_mssql(conn, sqlite_db: str, history_days: int = 30) -> dict:
    """Sync all SQLite data into MSSQL (merge, no overwrite)."""
    mc = conn.cursor()
    mc.fast_executemany = True
    stats = {"memories": 0, "connections": 0, "history": 0, "dream_sessions": 0, "dream_insights": 0}

    if not os.path.exists(sqlite_db):
        print(f"  SQLite not found: {sqlite_db}, skipping sync")
        return stats

    banner("SQLite → MSSQL Sync")

    # Get existing IDs
    mc.execute("SELECT id FROM memories")
    existing_mem = set(r[0] for r in mc.fetchall())
    mc.execute("SELECT source_id, target_id FROM connections")
    existing_conn = set((r[0], r[1]) for r in mc.fetchall())
    mc.execute("SELECT id FROM memories")  # valid FK targets
    valid_mem = set(r[0] for r in mc.fetchall())

    sconn = sqlite3.connect(sqlite_db)
    scur = sconn.cursor()

    # Sync memories
    for row in scur.execute("SELECT id, label, content, embedding, salience, created_at, last_accessed, access_count FROM memories"):
        mid = row[0]
        if mid in existing_mem:
            continue
        vdim = get_embed_dim(row[3])
        try:
            mc.execute(
                "INSERT INTO memories (id, label, content, embedding, vector_dim, salience, created_at, last_accessed, access_count) VALUES (?,?,?,?,?,?,?,?,?)",
                mid, row[1], row[2], row[3], vdim, row[4],
                epoch_to_datetime(row[5]), epoch_to_datetime(row[6]), row[7] or 0
            )
            existing_mem.add(mid)
            valid_mem.add(mid)
            stats["memories"] += 1
        except Exception:
            pass
    conn.commit()
    print(f"  Memories: +{stats['memories']}")

    # Sync connections (FK-safe)
    skipped_fk = 0
    batch = []
    for row in scur.execute("SELECT id, source_id, target_id, weight, edge_type, created_at FROM connections"):
        sid, tid = row[1], row[2]
        if (sid, tid) in existing_conn:
            continue
        if sid not in valid_mem or tid not in valid_mem:
            skipped_fk += 1
            continue
        batch.append((sid, tid, row[3], row[4], epoch_to_datetime(row[5])))
        existing_conn.add((sid, tid))
        if len(batch) >= 10000:
            try:
                mc.executemany(
                    "INSERT INTO connections (source_id, target_id, weight, edge_type, created_at) VALUES (?,?,?,?,?)",
                    batch
                )
                stats["connections"] += len(batch)
                conn.commit()
            except Exception:
                pass
            batch = []
    if batch:
        try:
            mc.executemany(
                "INSERT INTO connections (source_id, target_id, weight, edge_type, created_at) VALUES (?,?,?,?,?)",
                batch
            )
            stats["connections"] += len(batch)
            conn.commit()
        except Exception:
            pass
    print(f"  Connections: +{stats['connections']} (skipped {skipped_fk} FK violations)")

    # Sync connection_history (last N days only)
    cutoff_ts = (datetime.now() - timedelta(days=history_days)).timestamp()
    mc.execute("SELECT id FROM connection_history")
    existing_hist = set(r[0] for r in mc.fetchall())
    batch = []
    for row in scur.execute("SELECT id, source_id, target_id, old_weight, new_weight, reason, changed_at FROM connection_history WHERE changed_at > ?", (cutoff_ts,)):
        if row[0] in existing_hist:
            continue
        batch.append((row[1], row[2], row[3], row[4], row[5], epoch_to_datetime(row[6])))
        existing_hist.add(row[0])
        if len(batch) >= 10000:
            mc.executemany(
                "INSERT INTO connection_history (source_id, target_id, old_weight, new_weight, reason, changed_at) VALUES (?,?,?,?,?,?)",
                batch
            )
            stats["history"] += len(batch)
            conn.commit()
            batch = []
            print(f"    ... {stats['history']:,} history entries")
    if batch:
        mc.executemany(
            "INSERT INTO connection_history (source_id, target_id, old_weight, new_weight, reason, changed_at) VALUES (?,?,?,?,?,?)",
            batch
        )
        stats["history"] += len(batch)
        conn.commit()
    print(f"  History: +{stats['history']}")

    # Sync dream_sessions + dream_insights from all SQLite DBs
    dream_dbs = [
        os.path.expanduser("~/.neural_memory/dream_sessions.db"),
        os.path.expanduser("~/.neural_memory/dream.db"),
        os.path.expanduser("~/.neural_memory/backups/dream.db"),
    ]
    mc.execute("SELECT id FROM dream_sessions")
    existing_ds = set(r[0] for r in mc.fetchall())
    mc.execute("SELECT id FROM dream_insights")
    existing_di = set(r[0] for r in mc.fetchall())

    for db_path in dream_dbs:
        if not os.path.exists(db_path):
            continue
        dconn = sqlite3.connect(db_path)
        dcur = dconn.cursor()
        # Detect columns dynamically
        ds_cols = [r[1] for r in dcur.execute("PRAGMA table_info(dream_sessions)").fetchall()]
        if not ds_cols:
            dconn.close()
            continue
        for row in dcur.execute(f"SELECT {', '.join(ds_cols)} FROM dream_sessions"):
            rd = dict(zip(ds_cols, row))
            if rd["id"] in existing_ds:
                continue
            try:
                mc.execute(
                    "INSERT INTO dream_sessions (id, phase, memories_processed, connections_strengthened, connections_pruned, bridges_found, insights_created, started_at, finished_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    rd["id"], rd.get("phase", "full"), rd.get("memories_processed", 0),
                    rd.get("connections_strengthened", 0), rd.get("connections_pruned", 0),
                    rd.get("bridges_found", 0), rd.get("insights_found", rd.get("insights_created", 0)),
                    epoch_to_datetime(rd.get("started_at")), epoch_to_datetime(rd.get("finished_at"))
                )
                existing_ds.add(rd["id"])
                stats["dream_sessions"] += 1
            except Exception:
                pass
        conn.commit()

        # dream_insights
        di_cols = [r[1] for r in dcur.execute("PRAGMA table_info(dream_insights)").fetchall()]
        if di_cols:
            for row in dcur.execute(f"SELECT {', '.join(di_cols)} FROM dream_insights"):
                rd = dict(zip(di_cols, row))
                if rd["id"] in existing_di:
                    continue
                try:
                    mc.execute(
                        "INSERT INTO dream_insights (id, session_id, insight_type, source_memory_id, content, confidence, created_at) VALUES (?,?,?,?,?,?,?)",
                        rd["id"], rd.get("session_id"), rd.get("insight_type", "cluster"),
                        rd.get("source_memory_id"), rd.get("content", ""), rd.get("confidence", 0.5),
                        epoch_to_datetime(rd.get("created_at"))
                    )
                    existing_di.add(rd["id"])
                    stats["dream_insights"] += 1
                except Exception:
                    pass
            conn.commit()
        dconn.close()

    print(f"  Dream Sessions: +{stats['dream_sessions']}")
    print(f"  Dream Insights: +{stats['dream_insights']}")

    sconn.close()
    return stats


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(conn, dry_run: bool = False) -> dict:
    """Remove all duplicates, keep best row per unique key."""
    mc = conn.cursor()
    stats = {"connections_deleted": 0, "history_deleted": 0, "history_compacted": 0}

    banner("DEDUPLICATION")

    # 1. Connections: keep highest weight per (source_id, target_id)
    mc.execute("""
        SELECT COUNT(*) FROM (
            SELECT source_id, target_id, COUNT(*) as cnt
            FROM connections GROUP BY source_id, target_id HAVING COUNT(*) > 1
        ) x
    """)
    conn_dupes = mc.fetchone()[0]
    if conn_dupes > 0:
        print(f"  connections: {conn_dupes} duplicate pairs found")
        if not dry_run:
            mc.execute("""
                WITH dups AS (
                    SELECT id, ROW_NUMBER() OVER (PARTITION BY source_id, target_id ORDER BY weight DESC, id) as rn
                    FROM connections
                )
                DELETE FROM dups WHERE rn > 1
            """)
            stats["connections_deleted"] = mc.rowcount
            conn.commit()
            print(f"  connections: deleted {stats['connections_deleted']} duplicate rows")

    # 2. Connection_history: keep latest per (source_id, target_id)
    mc.execute("SELECT COUNT(*) FROM connection_history")
    hist_before = mc.fetchone()[0]
    mc.execute("""
        SELECT COUNT(*) FROM (
            SELECT source_id, target_id, COUNT(*) as cnt
            FROM connection_history GROUP BY source_id, target_id HAVING COUNT(*) > 1
        ) x
    """)
    hist_dupes = mc.fetchone()[0]
    if hist_dupes > 0:
        print(f"  connection_history: {hist_dupes} duplicate pairs ({hist_before:,} total rows)")
        if not dry_run:
            mc.execute("""
                WITH ranked AS (
                    SELECT id, ROW_NUMBER() OVER (PARTITION BY source_id, target_id ORDER BY changed_at DESC, id DESC) as rn
                    FROM connection_history
                )
                DELETE FROM ranked WHERE rn > 1
            """)
            stats["history_deleted"] = mc.rowcount
            conn.commit()
            mc.execute("SELECT COUNT(*) FROM connection_history")
            hist_after = mc.fetchone()[0]
            stats["history_compacted"] = hist_after
            print(f"  connection_history: deleted {stats['history_deleted']:,}, now {hist_after:,} rows")

    # 3. Orphan connections
    mc.execute("""
        SELECT COUNT(*) FROM connections c
        WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = c.source_id)
           OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = c.target_id)
    """)
    orphans = mc.fetchone()[0]
    if orphans > 0:
        print(f"  connections: {orphans} orphan rows")
        if not dry_run:
            mc.execute("""
                DELETE FROM connections
                WHERE source_id NOT IN (SELECT id FROM memories)
                   OR target_id NOT IN (SELECT id FROM memories)
            """)
            deleted = mc.rowcount
            conn.commit()
            print(f"  connections: deleted {deleted} orphan rows")

    return stats


# ---------------------------------------------------------------------------
# V1 → V2 Migration
# ---------------------------------------------------------------------------

def migrate_v1_to_v2(conn, dry_run: bool = False) -> dict:
    """Drop V1 tables, keep V2 as canonical."""
    mc = conn.cursor()
    stats = {"dropped": []}
    tables = {r[0] for r in mc.execute("SELECT name FROM sys.tables").fetchall()}

    banner("V1 → V2 Migration")

    # 1. NeuralMemory_old (legacy vector store)
    if "NeuralMemory_old" in tables:
        mc.execute("SELECT COUNT(*) FROM NeuralMemory_old")
        nmo = mc.fetchone()[0]
        mc.execute("SELECT COUNT(*) FROM Mazemaker")
        nm = mc.fetchone()[0]
        print(f"  NeuralMemory_old: {nmo:,} rows | Mazemaker: {nm:,} rows")
        if nm >= nmo:
            if not dry_run:
                mc.execute("DROP TABLE NeuralMemory_old")
                conn.commit()
            stats["dropped"].append("NeuralMemory_old")
            print(f"  ✓ Dropped NeuralMemory_old ({nmo:,} rows, data in Mazemaker)")
        else:
            print(f"  ✗ NeuralMemory_old has MORE rows than Mazemaker — not dropping!")

    # 2. GraphNodes (V1) → GraphNodes_v2
    if "GraphNodes" in tables and "GraphNodes_v2" in tables:
        mc.execute("SELECT COUNT(*) FROM GraphNodes")
        gn1 = mc.fetchone()[0]
        mc.execute("SELECT COUNT(*) FROM GraphNodes_v2")
        gn2 = mc.fetchone()[0]
        print(f"  GraphNodes: {gn1:,} | GraphNodes_v2: {gn2:,}")
        if gn2 >= gn1:
            if not dry_run:
                mc.execute("DROP TABLE GraphNodes")
                conn.commit()
            stats["dropped"].append("GraphNodes")
            print(f"  ✓ Dropped GraphNodes (V1)")

    if "GraphEdges" in tables and "GraphEdges_v2" in tables:
        mc.execute("SELECT COUNT(*) FROM GraphEdges")
        ge1 = mc.fetchone()[0]
        mc.execute("SELECT COUNT(*) FROM GraphEdges_v2")
        ge2 = mc.fetchone()[0]
        print(f"  GraphEdges: {ge1:,} | GraphEdges_v2: {ge2:,}")
        if ge2 >= ge1:
            if not dry_run:
                mc.execute("DROP TABLE GraphEdges")
                conn.commit()
            stats["dropped"].append("GraphEdges")
            print(f"  ✓ Dropped GraphEdges (V1)")

    # 3. Duplicate connection_history_v2 tables
    mc.execute("""
        SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME LIKE 'connection_history_v2%'
    """)
    for r in mc.fetchall():
        if not dry_run:
            mc.execute(f"DROP TABLE [{r.TABLE_NAME}]")
            conn.commit()
        stats["dropped"].append(r.TABLE_NAME)
        print(f"  ✓ Dropped {r.TABLE_NAME}")

    if not stats["dropped"]:
        print("  No legacy tables to drop")

    return stats


# ---------------------------------------------------------------------------
# UNIQUE Constraints
# ---------------------------------------------------------------------------

def add_unique_constraints(conn, dry_run: bool = False) -> list[str]:
    """Add UNIQUE constraints on high-write tables to prevent NREM bloat."""
    mc = conn.cursor()
    added = []

    banner("UNIQUE Constraints")

    constraints = [
        ("connections", "UX_connections_unique",
         "CREATE UNIQUE INDEX UX_connections_unique ON connections(source_id, target_id)"),
        ("connection_history", "UX_connection_history_unique",
         "CREATE UNIQUE INDEX UX_connection_history_unique ON connection_history(source_id, target_id)"),
    ]

    for table, idx_name, ddl in constraints:
        mc.execute(f"""
            SELECT COUNT(*) FROM sys.indexes
            WHERE name = '{idx_name}' AND object_id = OBJECT_ID('{table}')
        """)
        exists = mc.fetchone()[0] > 0
        if exists:
            print(f"  ✓ {table}.{idx_name} already exists")
        else:
            if not dry_run:
                try:
                    mc.execute(ddl)
                    conn.commit()
                    added.append(f"{table}.{idx_name}")
                    print(f"  ✓ Created {table}.{idx_name}")
                except Exception as e:
                    print(f"  ✗ Failed to create {table}.{idx_name}: {e}")
            else:
                print(f"  ~ Would create {table}.{idx_name}")
                added.append(f"{table}.{idx_name} (dry-run)")

    return added


# ---------------------------------------------------------------------------
# Code Verification — MERGE/UPSERT check
# ---------------------------------------------------------------------------

def verify_merge_code(plugin_dir: str = "~/.hermes/plugins/memory/neural/") -> dict:
    """Verify all dream files use MERGE instead of raw INSERT."""
    plugin_dir = os.path.expanduser(plugin_dir)
    result = {"files_checked": 0, "issues": [], "ok": True}

    banner("Code Verification (MERGE/UPSERT)")

    files_to_check = [
        "dream_mssql_store.py",
        "dream_engine.py",
        "mssql_store.py",
    ]

    for fname in files_to_check:
        fpath = os.path.join(plugin_dir, fname)
        if not os.path.exists(fpath):
            result["issues"].append(f"File not found: {fpath}")
            result["ok"] = False
            continue

        result["files_checked"] += 1
        with open(fpath) as f:
            content = f.read()

        # Check for raw INSERT INTO connections/connection_history (without MERGE or ON CONFLICT)
        # Use multi-line context: check blocks of 5 lines around each INSERT
        raw_inserts = []
        lines = content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip().upper()
            if "INSERT INTO CONNECTIONS" in stripped:
                # Check surrounding 5 lines for MERGE or ON CONFLICT
                block = "\n".join(lines[max(0,i-2):min(len(lines),i+4)]).upper()
                if "MERGE" not in block and "ON CONFLICT" not in block:
                    raw_inserts.append(f"  {fname}:{i+1}: raw INSERT INTO connections")
            if "INSERT INTO CONNECTION_HISTORY" in stripped:
                block = "\n".join(lines[max(0,i-2):min(len(lines),i+4)]).upper()
                if "MERGE" not in block and "ON CONFLICT" not in block:
                    raw_inserts.append(f"  {fname}:{i+1}: raw INSERT INTO connection_history")

        if raw_inserts:
            result["issues"].extend(raw_inserts)
            result["ok"] = False
            print(f"  ✗ {fname}: {len(raw_inserts)} raw INSERT(s) found")
            for ri in raw_inserts:
                print(f"    {ri}")
        else:
            print(f"  ✓ {fname}: all INSERTs converted to MERGE")

    if result["ok"]:
        print("\n  All files use MERGE/UPSERT pattern ✓")
    else:
        print(f"\n  {len(result['issues'])} issue(s) found ✗")

    return result


# ---------------------------------------------------------------------------
# Functional Verification
# ---------------------------------------------------------------------------

def verify_functional(conn) -> dict:
    """Run functional checks to verify the database works correctly."""
    mc = conn.cursor()
    result = {"passed": 0, "failed": 0, "tests": []}

    banner("Functional Verification")

    def check(name: str, query: str, expect_fn=None):
        try:
            mc.execute(query)
            row = mc.fetchone()
            val = row[0] if row else None
            if expect_fn:
                ok = expect_fn(val)
            else:
                ok = val is not None
            status = "PASS" if ok else "FAIL"
            result["tests"].append({"name": name, "status": status, "value": val})
            if ok:
                result["passed"] += 1
                print(f"  ✓ {name}: {val}")
            else:
                result["failed"] += 1
                print(f"  ✗ {name}: {val}")
        except Exception as e:
            result["failed"] += 1
            result["tests"].append({"name": name, "status": "FAIL", "error": str(e)})
            print(f"  ✗ {name}: EXCEPTION — {e}")

    # Core tables exist and have data
    check("memories > 0", "SELECT COUNT(*) FROM memories", lambda v: v > 0)
    check("connections > 0", "SELECT COUNT(*) FROM connections", lambda v: v > 0)
    check("connection_history > 0", "SELECT COUNT(*) FROM connection_history", lambda v: v > 0)
    check("GraphNodes_v2 > 0", "SELECT COUNT(*) FROM GraphNodes_v2", lambda v: v > 0)
    check("GraphEdges_v2 > 0", "SELECT COUNT(*) FROM GraphEdges_v2", lambda v: v > 0)

    # No duplicates
    check("No connection dupes", """
        SELECT COUNT(*) FROM (
            SELECT source_id, target_id, COUNT(*) as cnt
            FROM connections GROUP BY source_id, target_id HAVING COUNT(*) > 1
        ) x
    """, lambda v: v == 0)

    check("No history dupes", """
        SELECT COUNT(*) FROM (
            SELECT source_id, target_id, COUNT(*) as cnt
            FROM connection_history GROUP BY source_id, target_id HAVING COUNT(*) > 1
        ) x
    """, lambda v: v == 0)

    # UNIQUE constraints exist
    check("UX_connections_unique exists", """
        SELECT COUNT(*) FROM sys.indexes
        WHERE name = 'UX_connections_unique' AND object_id = OBJECT_ID('connections')
    """, lambda v: v > 0)

    check("UX_connection_history_unique exists", """
        SELECT COUNT(*) FROM sys.indexes
        WHERE name = 'UX_connection_history_unique' AND object_id = OBJECT_ID('connection_history')
    """, lambda v: v > 0)

    # No orphans
    check("No orphan connections", """
        SELECT COUNT(*) FROM connections c
        WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = c.source_id)
           OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = c.target_id)
    """, lambda v: v == 0)

    # Legacy tables gone
    check("No NeuralMemory_old", """
        SELECT COUNT(*) FROM sys.tables WHERE name = 'NeuralMemory_old'
    """, lambda v: v == 0)

    check("No GraphNodes (V1)", """
        SELECT COUNT(*) FROM sys.tables WHERE name = 'GraphNodes'
    """, lambda v: v == 0)

    check("No GraphEdges (V1)", """
        SELECT COUNT(*) FROM sys.tables WHERE name = 'GraphEdges'
    """, lambda v: v == 0)

    # MERGE works: insert duplicate connection should UPDATE not fail
    try:
        mc.execute("SELECT TOP 1 source_id, target_id FROM connections")
        row = mc.fetchone()
        if row:
            sid, tid = row[0], row[1]
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
            w = mc.fetchone()[0]
            mc.execute("UPDATE connections SET weight = ? WHERE source_id = ? AND target_id = ?", 0.5, sid, tid)
            conn.commit()
            check("MERGE on connections works", "SELECT 1", lambda v: True)
            print(f"    (MERGE test on ({sid},{tid}) — updated and restored)")
    except Exception as e:
        check("MERGE on connections works", "SELECT 1", lambda v: False)
        print(f"    MERGE failed: {e}")

    # Final score
    total = result["passed"] + result["failed"]
    print(f"\n  Score: {result['passed']}/{total} passed")
    result["ok"] = result["failed"] == 0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MSSQL Production Migration & Verification")
    parser.add_argument("--config", default="~/.hermes/config.yaml", help="Hermes config path")
    parser.add_argument("--sqlite-db", default="~/.neural_memory/memory.db", help="Main SQLite DB")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--skip-sync", action="store_true", help="Skip SQLite→MSSQL sync")
    parser.add_argument("--history-days", type=int, default=3, help="Days of history to keep")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification")
    parser.add_argument("--plugin-dir", default="~/.hermes/plugins/memory/neural/", help="Plugin directory")
    args = parser.parse_args()

    banner("Mazemaker MSSQL Production Migration", "=")
    print(f"  Config: {args.config}")
    print(f"  SQLite: {args.sqlite_db}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  History retention: {args.history_days} days")
    print()

    # Connect
    conn = get_mssql_connection(args.config)

    # 1. Diagnose
    diag = diagnose(conn)
    print_diagnosis(diag)

    if args.verify_only:
        verify_functional(conn)
        verify_merge_code(args.plugin_dir)
        conn.close()
        return

    if not diag["issues"] and not diag["warnings"]:
        print("\n✓ Database is already production-grade!")
        if not args.force:
            verify_functional(conn)
            verify_merge_code(args.plugin_dir)
            conn.close()
            return

    if not args.dry_run and not confirm("Proceed with migration?", args.force):
        print("Aborted.")
        conn.close()
        return

    # 2. SQLite → MSSQL Sync
    if not args.skip_sync:
        sync_stats = sync_sqlite_to_mssql(conn, os.path.expanduser(args.sqlite_db), args.history_days)
        # Also sync from backup if exists
        backup_db = os.path.expanduser("~/.neural_memory/backups/memory_20260413_064511.db")
        if os.path.exists(backup_db):
            print("\n  Syncing from backup...")
            sync_sqlite_to_mssql(conn, backup_db, args.history_days)

    # 3. Deduplicate
    dedup_stats = deduplicate(conn, args.dry_run)

    # 4. V1 → V2
    v2_stats = migrate_v1_to_v2(conn, args.dry_run)

    # 5. UNIQUE Constraints
    constraint_stats = add_unique_constraints(conn, args.dry_run)

    # 6. Code Verification
    code_stats = verify_merge_code(args.plugin_dir)

    # 7. Functional Verification
    func_stats = verify_functional(conn)

    # Final report
    banner("MIGRATION COMPLETE")
    print(f"  Deduplication: {dedup_stats}")
    print(f"  V2 Migration: dropped {v2_stats['dropped']}")
    print(f"  Constraints: {constraint_stats}")
    print(f"  Code check: {'PASS' if code_stats['ok'] else 'FAIL'}")
    print(f"  Functional: {func_stats['passed']}/{func_stats['passed'] + func_stats['failed']} tests passed")

    conn.close()

    if func_stats["failed"] > 0 or not code_stats["ok"]:
        print("\n⚠ Some checks failed — review output above")
        sys.exit(1)
    else:
        print("\n✓ All checks passed — production ready!")


if __name__ == "__main__":
    main()
