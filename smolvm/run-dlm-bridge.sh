#!/usr/bin/env bash
# ============================================================
# JackrabbitDLM Bridge — Host-Side Script
# Starts JackrabbitDLM server on 0.0.0.0:37373
# Also runs the MSSQL bridge adapter
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SMOLVM_DATA_DIR:-$SCRIPT_DIR/data}"
LOG_FILE="$DATA_DIR/dlm_bridge.log"
DLM_PORT="${DLM_PORT:-37373}"
DLM_BIND="${DLM_BIND:-0.0.0.0}"

# MSSQL connection settings
MSSQL_HOST="${MSSQL_HOST:-localhost}"
MSSQL_PORT="${MSSQL_PORT:-1433}"
MSSQL_USER="${MSSQL_USER:-sa}"
MSSQL_PASS="${MSSQL_PASS:-}"
MSSQL_DB="${MSSQL_DB:-NeuralMemory}"
MSSQL_DRIVER="${MSSQL_DRIVER:-ODBC Driver 18 for SQL Server}"

mkdir -p "$DATA_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [dlm-bridge] $*" | tee -a "$LOG_FILE"
}

cleanup() {
    log "Shutting down JackrabbitDLM bridge..."
    exit 0
}
trap cleanup SIGINT SIGTERM

log "============================================"
log "  JackrabbitDLM Bridge — Host Mode"
log "  DLM listen: ${DLM_BIND}:${DLM_PORT}"
log "  MSSQL: ${MSSQL_HOST}:${MSSQL_PORT}/${MSSQL_DB}"
log "============================================"

# Verify pyodbc is available
python3 -c "import pyodbc" 2>/dev/null || {
    log "ERROR: pyodbc not installed on host. Run: pip install pyodbc"
    exit 1
}

python3 "${SCRIPT_DIR}/dlm_mssql_bridge.py" 2>&1 | tee -a "$LOG_FILE"
