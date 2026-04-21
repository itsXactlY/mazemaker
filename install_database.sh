#!/bin/bash
# install_database.sh — Neural Memory Database Setup
# Sets up SQLite and/or MSSQL databases with all required tables.
#
# Usage:
#   bash install_database.sh install [--lite|--full]
#   bash install_database.sh verify
#   bash install_database.sh sync [--dry-run] [--incremental]
#   bash install_database.sh status
#
# Modes:
#   --lite   SQLite only (default, no MSSQL dependency)
#   --full   SQLite + MSSQL (production)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NEURAL_DIR="$HOME/.neural_memory"
ENV_FILE="$HOME/.hermes/.env"
SYNC_STATE="$NEURAL_DIR/sync_state.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
print_info() { echo -e "  ${BLUE}[..]${NC} $1"; }
print_warn() { echo -e "  ${YELLOW}[!!]${NC} $1"; }
print_err()  { echo -e "  ${RED}[XX]${NC} $1"; }

# ---------------------------------------------------------------------------
# Load MSSQL credentials from .env
# ---------------------------------------------------------------------------
load_mssql_env() {
    MSSQL_SERVER="${MSSQL_SERVER:-}"
    MSSQL_DATABASE="${MSSQL_DATABASE:-}"
    MSSQL_USERNAME="${MSSQL_USERNAME:-}"
    MSSQL_PASSWORD="${MSSQL_PASSWORD:-}"
    MSSQL_DRIVER="${MSSQL_DRIVER:-}"

    if [ -f "$ENV_FILE" ]; then
        [ -z "$MSSQL_SERVER" ]   && MSSQL_SERVER=$(grep '^MSSQL_SERVER=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        [ -z "$MSSQL_DATABASE" ] && MSSQL_DATABASE=$(grep '^MSSQL_DATABASE=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        [ -z "$MSSQL_USERNAME" ] && MSSQL_USERNAME=$(grep '^MSSQL_USERNAME=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        [ -z "$MSSQL_PASSWORD" ] && MSSQL_PASSWORD=$(grep '^MSSQL_PASSWORD=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        [ -z "$MSSQL_DRIVER" ]   && MSSQL_DRIVER=$(grep '^MSSQL_DRIVER=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
    fi

    MSSQL_SERVER="${MSSQL_SERVER:-localhost}"
    MSSQL_DATABASE="${MSSQL_DATABASE:-NeuralMemory}"
    MSSQL_USERNAME="${MSSQL_USERNAME:-SA}"
    MSSQL_DRIVER="${MSSQL_DRIVER:-{ODBC Driver 18 for SQL Server}}"
}

# ---------------------------------------------------------------------------
# SQLite Setup (always included)
# ---------------------------------------------------------------------------
setup_sqlite() {
    echo ""
    echo -e "${CYAN}--- SQLite Setup ---${NC}"

    mkdir -p "$NEURAL_DIR"

    local DB_PATH="$NEURAL_DIR/memory.db"
    python3 << PYEOF
import sqlite3, os, sys

db = "$DB_PATH"
conn = sqlite3.connect(db)
conn.execute("PRAGMA journal_mode=WAL")

# Core tables
conn.executescript("""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    content TEXT,
    embedding BLOB,
    salience REAL DEFAULT 1.0,
    created_at REAL DEFAULT 0,
    last_accessed REAL DEFAULT 0,
    access_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER,
    target_id INTEGER,
    weight REAL DEFAULT 0.5,
    created_at REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_label ON memories(label);
CREATE INDEX IF NOT EXISTS idx_conn_source ON connections(source_id);
CREATE INDEX IF NOT EXISTS idx_conn_target ON connections(target_id);

-- Dream tables
CREATE TABLE IF NOT EXISTS dream_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    finished_at REAL,
    phase TEXT NOT NULL,
    memories_processed INTEGER DEFAULT 0,
    connections_strengthened INTEGER DEFAULT 0,
    connections_pruned INTEGER DEFAULT 0,
    bridges_found INTEGER DEFAULT 0,
    insights_created INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS dream_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    insight_type TEXT NOT NULL,
    source_memory_id INTEGER,
    content TEXT,
    confidence REAL DEFAULT 0.0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS connection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    old_weight REAL,
    new_weight REAL,
    reason TEXT,
    changed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_insights_type ON dream_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_dream_insights_session ON dream_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_conn_history_nodes ON connection_history(source_id, target_id);
""")

# Verify
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
conn_count = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
print(f"  Tables: {', '.join(t[0] for t in tables)}")
print(f"  Memories: {mem_count}, Connections: {conn_count}")
conn.close()
print(f"  Database: {db}")
print("  SQLite: OK")
PYEOF
    print_ok "SQLite database ready: $DB_PATH"
}

# ---------------------------------------------------------------------------
# MSSQL Setup (Full Stack only)
# ---------------------------------------------------------------------------
setup_mssql() {
    echo ""
    echo -e "${CYAN}--- MSSQL Setup ---${NC}"

    load_mssql_env

    # Check if MSSQL is running
    if ! systemctl is-active --quiet mssql-server 2>/dev/null; then
        print_warn "MSSQL service not running. Trying to start..."
        sudo systemctl start mssql-server 2>/dev/null || {
            print_err "Cannot start MSSQL. Is it installed? Run install.sh first."
            return 1
        }
    fi
    print_ok "MSSQL service running"

    # Prompt for password if not set
    if [ -z "$MSSQL_PASSWORD" ]; then
        echo ""
        echo -n "  Enter MSSQL SA password: "
        read -s MSSQL_PASSWORD
        echo ""
    fi

    if [ -z "$MSSQL_PASSWORD" ]; then
        print_err "No SA password provided."
        return 1
    fi

    local SQLCMD="/opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P '$MSSQL_PASSWORD' -C"

    # Create database
    print_info "Creating NeuralMemory database..."
    eval "$SQLCMD -Q \"
        IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'NeuralMemory')
            CREATE DATABASE NeuralMemory;
    \"" 2>/dev/null
    print_ok "Database NeuralMemory"

    # Create tables (1024d embeddings)
    print_info "Creating tables (1024d embedding support)..."
    eval "$SQLCMD -d NeuralMemory -Q \"
        -- Core tables
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'memories')
        CREATE TABLE memories (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            label NVARCHAR(256),
            content NVARCHAR(MAX),
            embedding VARBINARY(8000),
            vector_dim INT NOT NULL DEFAULT 1024,
            salience FLOAT DEFAULT 1.0,
            created_at DATETIME2(7) DEFAULT SYSUTCDATETIME(),
            last_accessed DATETIME2(7) DEFAULT SYSUTCDATETIME(),
            access_count INT DEFAULT 0
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'connections')
        CREATE TABLE connections (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            source_id BIGINT,
            target_id BIGINT,
            weight FLOAT DEFAULT 0.5,
            edge_type NVARCHAR(50) DEFAULT 'similar',
            created_at DATETIME2(7) DEFAULT SYSUTCDATETIME()
        );

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conn_source')
        CREATE INDEX idx_conn_source ON connections(source_id);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conn_target')
        CREATE INDEX idx_conn_target ON connections(target_id);

        -- Dream tables
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dream_sessions')
        CREATE TABLE dream_sessions (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            started_at FLOAT NOT NULL,
            finished_at FLOAT,
            phase NVARCHAR(20) NOT NULL,
            memories_processed INT DEFAULT 0,
            connections_strengthened INT DEFAULT 0,
            connections_pruned INT DEFAULT 0,
            bridges_found INT DEFAULT 0,
            insights_created INT DEFAULT 0
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dream_insights')
        CREATE TABLE dream_insights (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            session_id BIGINT,
            insight_type NVARCHAR(50) NOT NULL,
            source_memory_id BIGINT,
            content NVARCHAR(MAX),
            confidence FLOAT DEFAULT 0.0,
            created_at FLOAT NOT NULL
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'connection_history')
        CREATE TABLE connection_history (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            source_id BIGINT NOT NULL,
            target_id BIGINT NOT NULL,
            old_weight FLOAT,
            new_weight FLOAT,
            reason NVARCHAR(100),
            changed_at FLOAT NOT NULL
        );

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dream_insights_type')
        CREATE INDEX idx_dream_insights_type ON dream_insights(insight_type);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dream_insights_session')
        CREATE INDEX idx_dream_insights_session ON dream_insights(session_id);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dream_conn_history')
        CREATE INDEX idx_dream_conn_history ON connection_history(source_id, target_id);
    \"" 2>/dev/null
    print_ok "All tables created"

    # Verify
    local TABLES=$(eval "$SQLCMD -d NeuralMemory -Q 'SELECT name FROM sys.tables ORDER BY name' -h -1 -W" 2>/dev/null | grep -v '^$' | grep -v '^-' | tr '\n' ', ' | sed 's/,$//')
    print_ok "Tables: $TABLES"

    # Update .env
    print_info "Updating $ENV_FILE..."
    mkdir -p "$(dirname "$ENV_FILE")"
    if [ -f "$ENV_FILE" ]; then
        sed -i '/^MSSQL_/d' "$ENV_FILE" 2>/dev/null
    fi
    cat >> "$ENV_FILE" << ENVEOF

# MSSQL (Neural Memory)
MSSQL_SERVER=localhost
MSSQL_DATABASE=NeuralMemory
MSSQL_USERNAME=SA
MSSQL_PASSWORD=${MSSQL_PASSWORD}
MSSQL_DRIVER={ODBC Driver 18 for SQL Server}
ENVEOF
    print_ok "Credentials saved to $ENV_FILE"
}

# ---------------------------------------------------------------------------
# Verify databases
# ---------------------------------------------------------------------------
cmd_verify() {
    echo ""
    echo -e "${CYAN}--- Verification ---${NC}"

    python3 << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.expanduser("~/projects/neural-memory-adapter/python"))

# SQLite check
import sqlite3
db = os.path.expanduser("~/.neural_memory/memory.db")
if os.path.exists(db):
    conn = sqlite3.connect(db)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn_count = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
    conn.close()
    print(f"  [OK] SQLite: {len(tables)} tables, {mem_count} memories, {conn_count} connections ({db})")
else:
    print(f"  [--] SQLite: not found at {db}")

# MSSQL check (optional)
try:
    import pyodbc
    from mssql_store import MSSQLStore
    store = MSSQLStore()
    s = store.stats()
    store.close()
    print(f"  [OK] MSSQL: {s['memories']} memories, {s['connections']} connections")
except ImportError:
    print("  [--] MSSQL: pyodbc not installed (OK for lite mode)")
except Exception as e:
    print(f"  [--] MSSQL: not available ({e})")
PYEOF
}

# ---------------------------------------------------------------------------
# Sync SQLite -> MSSQL
# ---------------------------------------------------------------------------
cmd_sync() {
    local EXTRA_ARGS=""
    local DRY_RUN=false

    for arg in "$@"; do
        case "$arg" in
            --dry-run)    DRY_RUN=true; EXTRA_ARGS="$EXTRA_ARGS --dry-run" ;;
            --incremental) EXTRA_ARGS="$EXTRA_ARGS --incremental" ;;
            *)            EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
        esac
    done

    echo ""
    echo -e "${CYAN}--- SQLite → MSSQL Sync ---${NC}"

    load_mssql_env
    if [ -z "$MSSQL_PASSWORD" ]; then
        print_err "MSSQL_PASSWORD not set. Run 'install_database.sh install --full' first."
        return 1
    fi

    local SYNC_SCRIPT="$SCRIPT_DIR/tools/sync_sqlite_to_mssql.py"
    if [ ! -f "$SYNC_SCRIPT" ]; then
        print_err "sync_sqlite_to_mssql.py not found at $SYNC_SCRIPT"
        return 1
    fi

    python3 "$SYNC_SCRIPT" $EXTRA_ARGS
}

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
cmd_status() {
    echo ""
    echo -e "${CYAN}--- Database Status ---${NC}"

    # SQLite
    local DB_PATH="$NEURAL_DIR/memory.db"
    if [ -f "$DB_PATH" ]; then
        python3 -c "
import sqlite3, os
db = '$DB_PATH'
conn = sqlite3.connect(db)
m = conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
c = conn.execute('SELECT COUNT(*) FROM connections').fetchone()[0]
ds = conn.execute('SELECT COUNT(*) FROM dream_sessions').fetchone()[0]
sz = os.path.getsize(db) / (1024*1024)
print(f'  SQLite: {m} memories, {c} connections, {ds} dream sessions ({sz:.1f} MB)')
conn.close()
"
    else
        print_warn "SQLite: not found"
    fi

    # MSSQL
    load_mssql_env
    if [ -n "$MSSQL_PASSWORD" ]; then
        python3 -c "
try:
    import sys, os
    sys.path.insert(0, os.path.expanduser('~/projects/neural-memory-adapter/python'))
    from mssql_store import MSSQLStore
    s = MSSQLStore()
    st = s.stats()
    print(f'  MSSQL:  {st[\"memories\"]} memories, {st[\"connections\"]} connections')
    s.close()
except Exception as e:
    print(f'  MSSQL:  not available ({e})')
" 2>/dev/null
    else
        print_info "MSSQL: no credentials configured"
    fi

    # Sync state
    if [ -f "$SYNC_STATE" ]; then
        echo ""
        print_info "Last sync state:"
        python3 -c "
import json
with open('$SYNC_STATE') as f:
    s = json.load(f)
print(f'  Last sync:    {s.get(\"last_sync_time\", \"never\")}')
print(f'  Memories:     {s.get(\"synced_memories\", 0)}')
print(f'  Connections:  {s.get(\"synced_connections\", 0)}')
print(f'  Errors:       {s.get(\"sync_errors\", 0)}')
"
    fi
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
print_banner() {
    echo ""
    echo -e "${BOLD}=============================================="
    echo "  Neural Memory — Database Setup"
    echo -e "==============================================${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
CMD="${1:-install}"
shift 2>/dev/null || true

case "$CMD" in
    install)
        print_banner
        MODE="${1:-}"
        if [ -z "$MODE" ]; then
            echo "  Select installation mode:"
            echo ""
            echo -e "    ${GREEN}[1]${NC} Lite        — SQLite only (hash/tfidf embeddings)"
            echo -e "                     Budget VPS friendly (~50MB RAM, no GPU)"
            echo ""
            echo -e "    ${BLUE}[2]${NC} Full Stack  — SQLite + MSSQL"
            echo -e "                     Production (~500MB RAM, optional GPU)"
            echo ""
            echo -n "  Choice [1/2]: "
            read -n 1 -r CHOICE
            echo ""
            case "$CHOICE" in
                2) MODE="--full" ;;
                *) MODE="--lite" ;;
            esac
        fi

        case "$MODE" in
            --full|full)
                echo -e "  Mode: ${BLUE}Full Stack${NC}"
                setup_sqlite
                setup_mssql
                ;;
            --lite|lite|*)
                echo -e "  Mode: ${GREEN}Lite${NC}"
                setup_sqlite
                echo ""
                print_warn "MSSQL skipped (lite mode). Run with install --full for MSSQL."
                ;;
        esac

        cmd_verify
        echo ""
        echo -e "${BOLD}=============================================="
        echo "  Database Setup Complete!"
        echo -e "==============================================${NC}"
        echo ""
        echo "  SQLite: ~/.neural_memory/memory.db"
        if [ "$MODE" = "--full" ] || [ "$MODE" = "full" ]; then
            echo "  MSSQL:  localhost/NeuralMemory"
        fi
        echo ""
        echo "  Next: bash install.sh (install plugin)"
        echo ""
        ;;
    verify)
        print_banner
        cmd_verify
        ;;
    sync)
        print_banner
        cmd_sync "$@"
        ;;
    status)
        print_banner
        cmd_status
        ;;
    --help|-h)
        print_banner
        echo "Usage: bash install_database.sh <command> [OPTIONS]"
        echo ""
        echo "Commands:"
        echo "  install [--lite|--full]   Create databases and tables"
        echo "  verify                    Check database accessibility"
        echo "  sync [--dry-run]          Sync SQLite → MSSQL"
        echo "  status                    Show database stats"
        echo ""
        echo "Options (install):"
        echo "  --lite   SQLite only (default)"
        echo "  --full   SQLite + MSSQL"
        echo ""
        echo "Options (sync):"
        echo "  --dry-run        Show what would be synced"
        echo "  --incremental    Only sync new/changed records"
        ;;
    *)
        print_err "Unknown command: $CMD"
        echo "Run: bash install_database.sh --help"
        exit 1
        ;;
esac
