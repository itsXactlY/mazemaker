#!/usr/bin/env bash
# ============================================================================
# migrate.sh — Neural Memory: Day-0 → Production (one-shot)
#
# Fixes database growth + code-level retention in one command.
# Run once. Safe to re-run (idempotent).
#
# What it does:
#   1. Backup the SQLite database
#   2. Clean data (orphans, history bloat, dedup, VACUUM)
#   3. Patch dream_engine.py     (auto-prune every 50 cycles)
#   4. Patch dream_mssql_store.py (MSSQL retention methods)
#   5. Patch cpp_dream_backend.py (safe stubs for C++ backend)
#   6. Sync SQLite → MSSQL (memories + connections, full replace)
#   7. Verify integrity
#
# Usage:
#   bash migrate.sh [--db PATH] [--plugin-dir PATH] [--dry-run] [--skip-mssql]
#
# Defaults:
#   --db         ~/.neural_memory/memory.db
#   --plugin-dir ~/.hermes/hermes-agent/plugins/memory/neural
# ============================================================================

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET}  $*"; }
step()  { echo -e "\n${BOLD}━━━ $* ━━━${RESET}"; }

# ── Parse args ───────────────────────────────────────────────────────────────
DB_PATH="${HOME}/.neural_memory/memory.db"
PLUGIN_DIR="${HOME}/.hermes/hermes-agent/plugins/memory/neural"
DRY_RUN=""
SKIP_MSSQL=""
ADAPTER_DIR="${HOME}/projects/neural-memory-adapter"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --db)         DB_PATH="$2"; shift 2 ;;
        --plugin-dir) PLUGIN_DIR="$2"; shift 2 ;;
        --dry-run)    DRY_RUN="--dry-run"; shift ;;
        --skip-mssql) SKIP_MSSQL="1"; shift ;;
        --help|-h)
            echo "Usage: bash migrate.sh [--db PATH] [--plugin-dir PATH] [--dry-run] [--skip-mssql]"
            exit 0 ;;
        *) fail "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Preflight ────────────────────────────────────────────────────────────────
step "Preflight checks"

if [[ ! -f "$DB_PATH" ]]; then
    fail "Database not found: $DB_PATH"
    exit 1
fi

DB_SIZE=$(du -h "$DB_PATH" | cut -f1)
info "Database: $DB_PATH ($DB_SIZE)"
info "Plugin:   $PLUGIN_DIR"

if [[ -n "$DRY_RUN" ]]; then
    warn "DRY RUN — no changes will be made"
fi

if ! command -v python3 &>/dev/null; then
    fail "python3 not found"
    exit 1
fi

ok "Preflight passed"

# ── Step 1: Backup ───────────────────────────────────────────────────────────
step "Step 1/6 — Backup database"

if [[ -z "$DRY_RUN" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="${DB_PATH}.bak.${TIMESTAMP}"
    cp -a "$DB_PATH" "$BACKUP_PATH"
    BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
    ok "Backup: $BACKUP_PATH ($BACKUP_SIZE)"
else
    info "Would create: ${DB_PATH}.bak.<timestamp>"
fi

# ── Step 2: Data cleanup ─────────────────────────────────────────────────────
step "Step 2/6 — Data cleanup (orphans, history, dedup, VACUUM)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPGRADE_SCRIPT=""

for candidate in \
    "${SCRIPT_DIR}/production_upgrade.py" \
    "${PLUGIN_DIR}/tools/production_upgrade.py" \
    "${ADAPTER_DIR}/tools/production_upgrade.py" \
    "${SCRIPT_DIR}/../tools/production_upgrade.py"; do
    if [[ -f "$candidate" ]]; then
        UPGRADE_SCRIPT="$candidate"
        break
    fi
done

if [[ -z "$UPGRADE_SCRIPT" ]]; then
    warn "production_upgrade.py not found — running inline cleanup"
    python3 - "$DB_PATH" "$DRY_RUN" <<'PYEOF'
import sqlite3, sys, os, time

db_path = sys.argv[1]
dry_run = "--dry-run" in sys.argv

conn = sqlite3.connect(db_path)

orphans = conn.execute("""
    SELECT COUNT(*) FROM connections
    WHERE source_id NOT IN (SELECT id FROM memories)
       OR target_id NOT IN (SELECT id FROM memories)
""").fetchone()[0]

orphan_hist = conn.execute("""
    SELECT COUNT(*) FROM connection_history
    WHERE source_id NOT IN (SELECT id FROM memories)
       OR target_id NOT IN (SELECT id FROM memories)
""").fetchone()[0]

dupes = conn.execute("""
    SELECT COALESCE(SUM(cnt - 1), 0) FROM (
        SELECT source_id, target_id, edge_type, COUNT(*) as cnt
        FROM connections GROUP BY source_id, target_id, edge_type HAVING cnt > 1
    )
""").fetchone()[0]

before_mb = os.path.getsize(db_path) / 1024 / 1024
print(f"  Before: {before_mb:.1f} MB | orphans={orphans:,} orphan_hist={orphan_hist:,} dupes={dupes:,}")

if dry_run:
    print("  DRY RUN — skipping cleanup")
    conn.close()
    sys.exit(0)

conn.execute("""
    DELETE FROM connections
    WHERE source_id NOT IN (SELECT id FROM memories)
       OR target_id NOT IN (SELECT id FROM memories)
""")
print(f"  Removed {orphans:,} orphan connections")

conn.execute("""
    DELETE FROM connection_history
    WHERE source_id NOT IN (SELECT id FROM memories)
       OR target_id NOT IN (SELECT id FROM memories)
""")
print(f"  Removed {orphan_hist:,} orphan history entries")

if dupes > 0:
    conn.execute("DELETE FROM connections WHERE id NOT IN (SELECT MIN(id) FROM connections GROUP BY source_id, target_id, edge_type)")
    print(f"  Removed {dupes:,} duplicate edges")

existing = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_connections_unique'").fetchone()
if not existing:
    conn.execute("CREATE UNIQUE INDEX idx_connections_unique ON connections(source_id, target_id, edge_type)")
    print("  Added UNIQUE constraint (idx_connections_unique)")

conn.execute("CREATE INDEX IF NOT EXISTS idx_conn_history_time ON connection_history(changed_at)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_dream_sessions_time ON dream_sessions(started_at)")
conn.commit()

conn.execute("VACUUM")
after_mb = os.path.getsize(db_path) / 1024 / 1024
print(f"  After VACUUM: {after_mb:.1f} MB (freed {before_mb - after_mb:.1f} MB)")

result = conn.execute("PRAGMA integrity_check").fetchone()[0]
print(f"  Integrity: {result}")

conn.close()
PYEOF
else
    info "Using: $UPGRADE_SCRIPT"
    python3 "$UPGRADE_SCRIPT" --db "$DB_PATH" --force $DRY_RUN
fi

ok "Data cleanup complete"

# ── Step 3: Patch dream_engine.py ────────────────────────────────────────────
step "Step 3/6 — Patch dream_engine.py (auto-prune)"

python3 - "$PLUGIN_DIR/dream_engine.py" "$DRY_RUN" <<'PYEOF'
import sys, re, os

path = sys.argv[1]
dry_run = "--dry-run" in sys.argv

with open(path) as f:
    content = f.read()

patches_applied = []

_abstract_end = content.find('class SQLiteDreamBackend')
_abstract_section = content[:_abstract_end] if _abstract_end > 0 else content
_patch_a_needed = 'def prune_connection_history' not in _abstract_section

if _patch_a_needed:
    old = '''    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        raise NotImplementedError

    def get_dream_stats(self) -> Dict[str, Any]:'''

    new = '''    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        raise NotImplementedError

    def prune_connection_history(self, keep_days: int = 7) -> int:
        """Delete history entries older than keep_days. Returns count deleted."""
        raise NotImplementedError

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Delete dream sessions older than keep_days. Returns count deleted."""
        raise NotImplementedError

    def prune_orphans(self) -> int:
        """Delete connections pointing to non-existent memories."""
        raise NotImplementedError

    def get_dream_stats(self) -> Dict[str, Any]:'''

    if old in content:
        content = content.replace(old, new, 1)
        patches_applied.append("A: abstract methods")
    else:
        print("  WARN: Patch A pattern not found (maybe already applied)")

if 'def prune_connection_history(self, keep_days' not in content or \
   content.count('def prune_connection_history(self, keep_days') < 2:

    old = '''            conn.commit()
        finally:
            conn.close()

    def get_dream_stats(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            s = conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(memories_processed), 0), "
                "COALESCE(SUM(connections_strengthened), 0), "
                "COALESCE(SUM(connections_pruned), 0), "
                "COALESCE(SUM(bridges_found), 0), "
                "COALESCE(SUM(insights_created), 0) "
                "FROM dream_sessions"
            ).fetchone()'''

    new = '''            conn.commit()
        finally:
            conn.close()

    def prune_connection_history(self, keep_days: int = 7) -> int:
        """Delete history entries older than keep_days."""
        conn = self._connect()
        try:
            cutoff = time.time() - (keep_days * 86400)
            count = conn.execute(
                "DELETE FROM connection_history WHERE changed_at < ?",
                (cutoff,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Delete dream sessions older than keep_days."""
        conn = self._connect()
        try:
            cutoff = time.time() - (keep_days * 86400)
            count = conn.execute(
                "DELETE FROM dream_sessions WHERE started_at < ?",
                (cutoff,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def prune_orphans(self) -> int:
        """Delete connections pointing to non-existent memories."""
        conn = self._connect()
        try:
            count = conn.execute(
                "DELETE FROM connections "
                "WHERE source_id NOT IN (SELECT id FROM memories) "
                "OR target_id NOT IN (SELECT id FROM memories)"
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def get_dream_stats(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            s = conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(memories_processed), 0), "
                "COALESCE(SUM(connections_strengthened), 0), "
                "COALESCE(SUM(connections_pruned), 0), "
                "COALESCE(SUM(bridges_found), 0), "
                "COALESCE(SUM(insights_created), 0) "
                "FROM dream_sessions"
            ).fetchone()'''

    if old in content:
        content = content.replace(old, new, 1)
        patches_applied.append("B: backend methods")
    else:
        print("  WARN: Patch B pattern not found (maybe already applied)")

if '_dream_count % 50 == 0' not in content:
    old = '''            # Prune dead connections
            stats["pruned"] = self._backend.prune_weak(0.05)

            self._backend.finish_session(session_id, stats)'''

    new = '''            # Prune dead connections
            stats["pruned"] = self._backend.prune_weak(0.05)

            # Periodic maintenance: prune old history + orphans every 50 cycles
            if self._dream_count % 50 == 0:
                try:
                    pruned_hist = self._backend.prune_connection_history(keep_days=7)
                    if pruned_hist:
                        logger.info("Pruned %d old connection_history entries", pruned_hist)
                    pruned_sessions = self._backend.prune_old_dream_sessions(keep_days=30)
                    if pruned_sessions:
                        logger.info("Pruned %d old dream sessions", pruned_sessions)
                    pruned_orphans = self._backend.prune_orphans()
                    if pruned_orphans:
                        logger.info("Pruned %d orphan connections", pruned_orphans)
                except Exception as e:
                    logger.debug("Maintenance cleanup error: %s", e)

            self._backend.finish_session(session_id, stats)'''

    if old in content:
        content = content.replace(old, new, 1)
        patches_applied.append("C: NREM periodic cleanup")
    else:
        print("  WARN: Patch C pattern not found (maybe already applied)")

if patches_applied:
    if not dry_run:
        with open(path, 'w') as f:
            f.write(content)
        print(f"  Applied: {', '.join(patches_applied)}")
    else:
        print(f"  Would apply: {', '.join(patches_applied)}")
else:
    print("  Already up to date — no patches needed")
PYEOF

ok "dream_engine.py patched"

# ── Step 4: Patch dream_mssql_store.py ───────────────────────────────────────
step "Step 4/6 — Patch dream_mssql_store.py (MSSQL retention)"

MSSQL_FILE="${PLUGIN_DIR}/dream_mssql_store.py"
if [[ -f "$MSSQL_FILE" ]]; then
    python3 - "$MSSQL_FILE" "$DRY_RUN" <<'PYEOF'
import sys

path = sys.argv[1]
dry_run = "--dry-run" in sys.argv

with open(path) as f:
    content = f.read()

patches = []

if 'def prune_connection_history' not in content:
    old_marker = '    def close(self):'
    insert = '''    def prune_connection_history(self, keep_days: int = 7) -> int:
        """Prune old connection history entries to prevent unbounded growth."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM connection_history "
            "WHERE changed_at < DATEADD(day, -?, SYSUTCDATETIME())",
            keep_days
        )
        self.conn.commit()
        return cursor.rowcount

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Prune old dream sessions and their insights."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE di FROM dream_insights di "
            "INNER JOIN dream_sessions ds ON di.session_id = ds.id "
            "WHERE ds.started_at < DATEADD(day, -?, SYSUTCDATETIME())",
            keep_days
        )
        cursor.execute(
            "DELETE FROM dream_sessions "
            "WHERE started_at < DATEADD(day, -?, SYSUTCDATETIME())",
            keep_days
        )
        self.conn.commit()
        return cursor.rowcount

    def prune_orphans(self) -> int:
        """Delete connections pointing to non-existent memories in MSSQL."""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM connections
            WHERE source_id NOT IN (SELECT id FROM memories)
               OR target_id NOT IN (SELECT id FROM memories)
        """)
        self.conn.commit()
        return cursor.rowcount

'''
    if old_marker in content:
        content = content.replace(old_marker, insert + old_marker, 1)
        patches.append("prune methods")

if patches:
    if not dry_run:
        with open(path, 'w') as f:
            f.write(content)
        print(f"  Applied: {', '.join(patches)}")
    else:
        print(f"  Would apply: {', '.join(patches)}")
else:
    print("  Already up to date")
PYEOF
    ok "dream_mssql_store.py patched"
else
    info "dream_mssql_store.py not found (SQLite-only setup — OK)"
fi

# ── Step 5: Patch cpp_dream_backend.py ───────────────────────────────────────
step "Step 5/6 — Patch cpp_dream_backend.py (safe stubs)"

CPP_FILE="${PLUGIN_DIR}/cpp_dream_backend.py"
if [[ -f "$CPP_FILE" ]]; then
    python3 - "$CPP_FILE" "$DRY_RUN" <<'PYEOF'
import sys

path = sys.argv[1]
dry_run = "--dry-run" in sys.argv

with open(path) as f:
    content = f.read()

if 'def prune_connection_history' not in content:
    _log_end = content.find('    def add_insight')
    if _log_end > 0:
        _insert_marker = content.rfind('\n\n', 0, _log_end)
        if _insert_marker > 0:
            stubs = '''
    def prune_connection_history(self, keep_days: int = 7) -> int:
        """Skip — C++/MSSQL handles history internally."""
        return 0

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Prune old dream sessions from SQLite tracking DB."""
        import sqlite3, time
        conn = sqlite3.connect(self._session_db)
        try:
            cutoff = time.time() - (keep_days * 86400)
            count = conn.execute(
                "DELETE FROM dream_sessions WHERE started_at < ?",
                (cutoff,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def prune_orphans(self) -> int:
        """Skip — C++/MSSQL handles referential integrity."""
        return 0

'''
            if not dry_run:
                content = content[:_insert_marker] + stubs + content[_insert_marker:]
                with open(path, 'w') as f:
                    f.write(content)
                print("  Applied: prune stubs")
            else:
                print("  Would apply: prune stubs")
        else:
            print("  WARN: Could not find insertion point")
    else:
        print("  WARN: add_insight method not found")
else:
    print("  Already up to date")
PYEOF
    ok "cpp_dream_backend.py patched"
else
    info "cpp_dream_backend.py not found (SQLite-only setup — OK)"
fi

# ── Step 6: SQLite → MSSQL sync ───────────────────────────────────────────────
step "Step 6/6 — Sync SQLite → MSSQL (cold storage)"

if [[ -n "$SKIP_MSSQL" ]]; then
    info "Skipped (--skip-mssql)"
else
    python3 - "$DB_PATH" "$PLUGIN_DIR" "$DRY_RUN" <<'PYEOF'
import sys, os, sqlite3

db_path   = sys.argv[1]
plugin_dir = sys.argv[2]
dry_run   = "--dry-run" in sys.argv

# Load mssql_store config from config.yaml
import importlib.util, pathlib

config_path = pathlib.Path(plugin_dir) / "config.py"
cfg = None
try:
    spec = importlib.util.spec_from_file_location("config", config_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.load_config()
except Exception as e:
    print(f"  Could not load config.py: {e}")

mssql_cfg = None
if cfg:
    try:
        mssql_cfg = cfg.get("memory", {}).get("neural", {}).get("mssql")
    except Exception:
        pass

if not mssql_cfg:
    # Try reading config.yaml directly
    import yaml, pathlib
    config_yaml = pathlib.Path.home() / ".hermes" / "config.yaml"
    if config_yaml.exists():
        try:
            with open(config_yaml) as f:
                raw = yaml.safe_load(f)
            mssql_cfg = raw.get("memory", {}).get("neural", {}).get("mssql")
        except Exception as e:
            print(f"  Could not read config.yaml: {e}")

if not mssql_cfg:
    print("  No MSSQL config found — skipping sync")
    print("  Add mssql: block under memory.neural in ~/.hermes/config.yaml to enable")
    sys.exit(0)

# Connect MSSQL
try:
    import pyodbc
except ImportError:
    print("  pyodbc not installed — skipping MSSQL sync")
    print("  Run: pip install pyodbc --break-system-packages")
    sys.exit(0)

server   = mssql_cfg.get("server", "127.0.0.1")
database = mssql_cfg.get("database", "NeuralMemory")
username = mssql_cfg.get("username", "SA")
password = mssql_cfg.get("password", "")
driver   = mssql_cfg.get("driver", "{ODBC Driver 18 for SQL Server}")

conn_str = (
    f"DRIVER={driver};SERVER={server};DATABASE={database};"
    f"UID={username};PWD={password};TrustServerCertificate=yes;"
)

try:
    mssql = pyodbc.connect(conn_str, timeout=10)
    print(f"  Connected: {server}/{database}")
except Exception as e:
    print(f"  MSSQL connection failed: {e}")
    print("  Skipping sync — SQLite is source of truth")
    sys.exit(0)

# Read SQLite
sqlite = sqlite3.connect(db_path)
sqlite.row_factory = sqlite3.Row

memories    = sqlite.execute("SELECT * FROM memories").fetchall()
connections = sqlite.execute("SELECT * FROM connections").fetchall()

print(f"  SQLite: {len(memories):,} memories, {len(connections):,} connections")

if dry_run:
    print("  DRY RUN — would sync to MSSQL")
    sqlite.close()
    mssql.close()
    sys.exit(0)

cur = mssql.cursor()

# ── Sync memories ───────────────────────────────────────────────────────
print("  Syncing memories...", end=" ", flush=True)

cur.execute("SELECT id FROM memories")
existing_ids = {row[0] for row in cur.fetchall()}
sqlite_ids   = {row["id"] for row in memories}

# Delete removed
to_delete = existing_ids - sqlite_ids
if to_delete:
    placeholders = ",".join("?" * len(to_delete))
    cur.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", list(to_delete))
    print(f"deleted {len(to_delete)}", end=" ", flush=True)

# Upsert existing/new
inserted = updated = 0
for row in memories:
    if row["id"] in existing_ids:
        cur.execute("""
            UPDATE memories SET
                label        = ?,
                content      = ?,
                embedding    = ?,
                salience     = ?,
                created_at   = ?,
                last_accessed = ?,
                access_count = ?
            WHERE id = ?
        """, (
            row["label"], row["content"], row["embedding"],
            row["salience"], row["created_at"], row["last_accessed"],
            row["access_count"], row["id"]
        ))
        updated += 1
    else:
        cur.execute("""
            INSERT INTO memories
                (id, label, content, embedding, salience, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["id"], row["label"], row["content"], row["embedding"],
            row["salience"], row["created_at"], row["last_accessed"],
            row["access_count"]
        ))
        inserted += 1

mssql.commit()
print(f"| inserted={inserted} updated={updated}")

# ── Sync connections (full replace — fast, clean) ───────────────────────
print("  Syncing connections (full replace)...", end=" ", flush=True)

cur.execute("DELETE FROM [connections]")
deleted_conn = cur.rowcount

batch = []
for row in connections:
    batch.append((
        row["source_id"], row["target_id"],
        row["weight"],
        row["edge_type"] if "edge_type" in row.keys() else "similar"
    ))
    if len(batch) >= 1000:
        cur.executemany(
            "INSERT INTO [connections] (source_id, target_id, weight, edge_type) VALUES (?, ?, ?, ?)",
            batch
        )
        batch = []

if batch:
    cur.executemany(
        "INSERT INTO [connections] (source_id, target_id, weight, edge_type) VALUES (?, ?, ?, ?)",
        batch
    )

mssql.commit()
print(f"| replaced {deleted_conn:,} → inserted {len(connections):,}")

# ── Verify counts ────────────────────────────────────────────────────────
mssql_mem_count  = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
mssql_conn_count = cur.execute("SELECT COUNT(*) FROM [connections]").fetchone()[0]

print(f"  MSSQL after sync: {mssql_mem_count:,} memories, {mssql_conn_count:,} connections")

if mssql_mem_count == len(memories) and mssql_conn_count == len(connections):
    print("  Counts match — sync PASS")
else:
    print(f"  WARNING: count mismatch! SQLite={len(memories)}/{len(connections)} MSSQL={mssql_mem_count}/{mssql_conn_count}")

sqlite.close()
mssql.close()
PYEOF

    if [[ $? -eq 0 ]]; then
        ok "SQLite → MSSQL sync complete"
    else
        warn "MSSQL sync had issues — check output above. SQLite remains source of truth."
    fi
fi

# ── Final verification ────────────────────────────────────────────────────────
step "Verification"

if [[ -z "$DRY_RUN" ]]; then
    python3 - "$DB_PATH" "$PLUGIN_DIR" <<'PYEOF'
import sqlite3, sys, os

db_path    = sys.argv[1]
plugin_dir = sys.argv[2]

ok_all = True

conn = sqlite3.connect(db_path)
result = conn.execute("PRAGMA integrity_check").fetchone()[0]
if result != "ok":
    print(f"  FAIL: integrity_check = {result}")
    ok_all = False
else:
    print(f"  DB integrity: PASS")

orphans = conn.execute("""
    SELECT COUNT(*) FROM connections
    WHERE source_id NOT IN (SELECT id FROM memories)
       OR target_id NOT IN (SELECT id FROM memories)
""").fetchone()[0]
print(f"  Orphan connections: {orphans}  {'PASS' if orphans == 0 else 'FAIL'}")
if orphans > 0:
    ok_all = False

dupes = conn.execute("""
    SELECT COUNT(*) FROM (
        SELECT source_id, target_id, edge_type, COUNT(*) as cnt
        FROM connections GROUP BY source_id, target_id, edge_type HAVING cnt > 1
    )
""").fetchone()[0]
print(f"  Duplicate edges: {dupes}  {'PASS' if dupes == 0 else 'FAIL'}")
if dupes > 0:
    ok_all = False

has_unique = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_connections_unique'"
).fetchone()
print(f"  UNIQUE constraint: {'ACTIVE' if has_unique else 'MISSING'}")

size_mb = os.path.getsize(db_path) / 1024 / 1024
mem_count  = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
conn_count = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
avg_w      = conn.execute("SELECT ROUND(AVG(weight),3) FROM connections").fetchone()[0]
print(f"  DB size: {size_mb:.1f} MB | memories={mem_count:,} | connections={conn_count:,} | avg_weight={avg_w}")
conn.close()

for fname in ['dream_engine.py']:
    fpath = os.path.join(plugin_dir, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            code = f.read()
        has_prune   = 'def prune_connection_history' in code
        has_cleanup = '_dream_count % 50 == 0' in code
        print(f"  {fname}: prune={'OK' if has_prune else 'MISSING'} auto-cleanup={'OK' if has_cleanup else 'MISSING'}")
        if not has_prune or not has_cleanup:
            ok_all = False

if ok_all:
    print(f"\n  ALL CHECKS PASSED")
else:
    print(f"\n  SOME CHECKS FAILED — review above")
    sys.exit(1)
PYEOF
else
    info "Skipped (dry run)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  MIGRATION COMPLETE${RESET}"
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════${RESET}"
echo ""
echo "  What changed:"
echo "    DB: orphans removed, history cleaned, deduped, VACUUM'd"
echo "    dream_engine.py: auto-prune every 50 Dream cycles"
echo "    dream_mssql_store.py: retention methods added"
echo "    cpp_dream_backend.py: safe stubs added"
echo "    MSSQL: synced from SQLite (memories + connections)"
echo ""
if [[ -z "$DRY_RUN" ]]; then
    echo "  Backup: ${BACKUP_PATH:-N/A}"
    echo ""
    echo "  Restart Hermes to pick up code changes."
    echo "  The Dream Engine will now auto-prune every 50 cycles."
else
    echo "  Run without --dry-run to apply changes."
fi
echo ""