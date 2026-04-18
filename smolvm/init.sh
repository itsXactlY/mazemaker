#!/usr/bin/env bash
# ============================================================
# SmolVM Entrypoint — Neural Memory PoC Init Sequence
# Steps: embed_server -> sqlite_init -> dlm_connect -> health_check
# ============================================================
set -euo pipefail

LOG_DIR="${SMOLVM_LOG_DIR:-/app/logs}"
DATA_DIR="${SMOLVM_DATA_DIR:-/app/data}"
SQLITE_PATH="${SMOLVM_SQLITE_PATH:-/app/data/neural_memory.db}"
EMBED_PORT="${SMOLVM_EMBED_PORT:-8501}"
DLM_HOST="${DLM_HOST:-host.docker.internal}"
DLM_PORT="${DLM_PORT:-37373}"
DLM_TIMEOUT="${DLM_TIMEOUT:-10}"

mkdir -p "$LOG_DIR" "$DATA_DIR"

log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] [init] $*" | tee -a "$LOG_DIR/init.log"
}

fail() {
    log "FATAL: $*"
    exit 1
}

# ============================================================
# Step 1: Start Embed Server (loads bge-m3 on CPU)
# ============================================================
log "=== Step 1/4: Starting embedding server ==="
/app/start-embed-server.sh &
EMBED_PID=$!
log "Embed server PID: $EMBED_PID"

# Wait for embed server to be ready (up to 120s for model load)
log "Waiting for embed server on port $EMBED_PORT..."
EMBED_READY=false
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${EMBED_PORT}/health" >/dev/null 2>&1; then
        EMBED_READY=true
        break
    fi
    if ! kill -0 "$EMBED_PID" 2>/dev/null; then
        fail "Embed server process died during startup"
    fi
    sleep 1
done

if [ "$EMBED_READY" != "true" ]; then
    fail "Embed server did not become ready within 120s"
fi
log "Embed server READY on port $EMBED_PORT"

# ============================================================
# Step 2: Initialize SQLite Neural Memory Store
# ============================================================
log "=== Step 2/4: Initializing SQLite neural memory ==="

# Create DB if not exists, run schema migrations
python3 -c "
import sqlite3
import os

db_path = os.environ.get('SMOLVM_SQLITE_PATH', '$SQLITE_PATH')
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Core tables for neural memory
cur.execute('''CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    embedding BLOB,
    metadata TEXT,
    created_at REAL DEFAULT (unixepoch()),
    updated_at REAL DEFAULT (unixepoch()),
    namespace TEXT DEFAULT 'default'
)''')

cur.execute('''CREATE TABLE IF NOT EXISTS memory_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    relation TEXT DEFAULT 'related',
    strength REAL DEFAULT 1.0,
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
)''')

cur.execute('''CREATE INDEX IF NOT EXISTS idx_memories_namespace
    ON memories(namespace)''')
cur.execute('''CREATE INDEX IF NOT EXISTS idx_memories_created
    ON memories(created_at)''')
cur.execute('''CREATE INDEX IF NOT EXISTS idx_links_source
    ON memory_links(source_id)''')

conn.commit()
conn.close()
print('SQLite initialized: ' + db_path)
" 2>&1 | tee -a "$LOG_DIR/sqlite_init.log"

if [ $? -ne 0 ]; then
    fail "SQLite initialization failed"
fi
log "SQLite neural memory READY at $SQLITE_PATH"

# ============================================================
# Step 3: Connect to JackrabbitDLM Server on Host (port 37373)
# ============================================================
log "=== Step 3/4: JackrabbitDLM connection ==="

DLM_CONNECTED=false

log "Attempting DLM connection to ${DLM_HOST}:${DLM_PORT}..."

# Check if DLM server is reachable
if python3 -c "
import socket, sys
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(${DLM_TIMEOUT})
    s.connect(('${DLM_HOST}', ${DLM_PORT}))
    s.close()
    sys.exit(0)
except Exception as e:
    print(f'DLM not reachable: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null; then
    log "DLM server reachable at ${DLM_HOST}:${DLM_PORT}"

    # Start DLM client in background
    /app/start-dlm-client.sh &
    DLM_PID=$!
    log "DLM client PID: $DLM_PID"

    # Wait for DLM registration confirmation
    for i in $(seq 1 "$DLM_TIMEOUT"); do
        if [ -f "$DATA_DIR/dlm_registered" ] 2>/dev/null; then
            DLM_CONNECTED=true
            break
        fi
        sleep 1
    done

    if [ "$DLM_CONNECTED" = "true" ]; then
        log "JackrabbitDLM CONNECTED — registered as neural-memory-smolvm"
    else
        log "WARNING: DLM registration timed out (${DLM_TIMEOUT}s) — continuing without DLM bridge"
    fi
else
    log "DLM server not reachable at ${DLM_HOST}:${DLM_PORT} — continuing without DLM bridge (optional)"
fi

# ============================================================
# Step 4: Health Check Loop
# ============================================================
log "=== Step 4/4: Running health checks ==="

check_health() {
    local embed_ok=false
    local sqlite_ok=false

    # Check embed server
    if curl -sf "http://localhost:${EMBED_PORT}/health" >/dev/null 2>&1; then
        embed_ok=true
    fi

    # Check SQLite
    if [ -f "$SQLITE_PATH" ]; then
        sqlite_ok=true
    fi

    if [ "$embed_ok" = "true" ] && [ "$sqlite_ok" = "true" ]; then
        return 0
    fi
    return 1
}

HEALTH_PASSES=0
for i in 1 2 3; do
    if check_health; then
        HEALTH_PASSES=$((HEALTH_PASSES + 1))
    fi
    sleep 1
done

if [ $HEALTH_PASSES -lt 2 ]; then
    fail "Health check failed (only $HEALTH_PASSES/3 passes)"
fi

log "============================================"
log "  SmolVM Neural Memory — ALL SYSTEMS READY"
log "  Embed server: http://localhost:${EMBED_PORT}"
log "  SQLite:       $SQLITE_PATH"
log "  DLM bridge:   ${DLM_CONNECTED:-false}"
if [ "$DLM_CONNECTED" = "true" ]; then
    log "  DLM target:   ${DLM_HOST}:${DLM_PORT}"
fi
log "============================================"

# Keep container running — wait for embed server
wait $EMBED_PID
