#!/usr/bin/env bash
#
# hermes-clean-restart.sh — kill running hermes session(s), drop regen-able
# caches, leave user data intact. Run before relaunching hermes when you
# need the new neural-memory-adapter code to take effect.
#
# Killed:
#   - any process matching `python.* hermes(-agent)?` (the running session)
#   - the shared-embed UNIX socket (~/.neural_memory/embed.sock) — stale
#     after a session dies; new hermes will rebuild it on first embed.
#
# Cleared (regenerates on demand):
#   - ~/.neural_memory/embed_cache.pkl       (LRU embedding cache; 32-37 MB)
#   - ~/.neural_memory/gpu_cache/            (GPU recall tensor + metadata)
#   - ~/.neural_memory/memory.db-{shm,wal}   (SQLite WAL artifacts)
#   - all __pycache__ dirs under the project + deployed plugin path so the
#     new code is freshly imported (skipped if directories don't exist)
#
# Preserved (your actual data — NEVER touched):
#   - ~/.neural_memory/memory.db             (the persistent memories)
#   - ~/.neural_memory/dream_sessions.db     (dream history)
#   - ~/.neural_memory/lstm_weights.bin      (trained predictor)
#   - ~/.neural_memory/models/               (HuggingFace model cache;
#                                             expensive to re-download)
#   - ~/.neural_memory/access_logs/          (forensic JSONL)
#   - ~/.neural_memory/backups/              (your backups)
#
# Auxiliary services that are NOT killed (separate from hermes itself):
#   - tools/dashboard/live_server.py (the /neural dashboard)
#   - neural-memory-mcp/mcp_local.py (MCP server)
# Restart those manually if you want the new code in them too.
#
# Usage:
#   bash tools/hermes-clean-restart.sh        # do the cleanup
#   bash tools/hermes-clean-restart.sh --dry  # show what would happen
#

set -euo pipefail

DRY=0
if [[ "${1:-}" == "--dry" || "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
    DRY=1
fi

YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
say() { echo -e "${CYAN}→${NC} $*"; }
ok()  { echo -e "${GREEN}✓${NC} $*"; }
warn(){ echo -e "${YELLOW}⚠${NC} $*"; }

run() {
    if (( DRY )); then
        echo "  [dry] $*"
    else
        eval "$@"
    fi
}

NEURAL_DIR="$HOME/.neural_memory"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLUGIN_DIR="$HOME/.hermes/hermes-agent/plugins/memory/neural"

# 1. Kill running hermes session(s) ─────────────────────────────────────
say "Looking for running hermes sessions..."
HERMES_PIDS=$(pgrep -f "python.*hermes(-agent)?($| --|/bin/hermes)" 2>/dev/null | tr '\n' ' ' || true)
# Filter out unrelated hermes-crypto / lan_gateway by checking cmdline
ACTUAL_PIDS=""
for pid in $HERMES_PIDS; do
    if [[ -r /proc/$pid/cmdline ]]; then
        cmd=$(tr '\0' ' ' </proc/$pid/cmdline)
        if [[ "$cmd" == *"hermes-agent"* || "$cmd" == *"/bin/hermes"* ]]; then
            ACTUAL_PIDS+="$pid "
        fi
    fi
done

if [[ -z "$ACTUAL_PIDS" ]]; then
    ok "no hermes sessions running"
else
    for pid in $ACTUAL_PIDS; do
        cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | head -c 80)
        say "killing PID $pid ($cmd...)"
        run "kill -TERM $pid 2>/dev/null || true"
    done
    sleep 2
    # Hard-kill anything that didn't go down
    for pid in $ACTUAL_PIDS; do
        if (( DRY == 0 )) && kill -0 "$pid" 2>/dev/null; then
            warn "PID $pid still alive, sending SIGKILL"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    ok "hermes session(s) terminated"
fi

# 2. Drop regen-able caches ─────────────────────────────────────────────
say "Clearing regen-able caches under $NEURAL_DIR"
for f in \
    "$NEURAL_DIR/embed.sock" \
    "$NEURAL_DIR/embed_cache.pkl" \
    "$NEURAL_DIR/memory.db-shm" \
    "$NEURAL_DIR/memory.db-wal"
do
    if [[ -e "$f" ]]; then
        run "rm -f -- '$f'"
    fi
done
if [[ -d "$NEURAL_DIR/gpu_cache" ]]; then
    run "rm -rf -- '$NEURAL_DIR/gpu_cache'"
fi
ok "caches cleared"

# 3. Purge __pycache__ so new code re-imports cleanly ───────────────────
say "Purging __pycache__ under project + plugin paths"
for base in "$PROJECT_DIR" "$PLUGIN_DIR"; do
    if [[ -d "$base" ]]; then
        if (( DRY )); then
            find "$base" -name __pycache__ -type d -print 2>/dev/null | sed 's/^/  [dry] rm -rf /'
        else
            find "$base" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
        fi
    fi
done
ok "__pycache__ purged"

# 4. Sanity report ──────────────────────────────────────────────────────
say "Preserved (user data, untouched):"
for f in memory.db dream_sessions.db lstm_weights.bin; do
    if [[ -e "$NEURAL_DIR/$f" ]]; then
        sz=$(du -h "$NEURAL_DIR/$f" 2>/dev/null | cut -f1)
        echo "  ✓ $NEURAL_DIR/$f ($sz)"
    fi
done
if [[ -d "$NEURAL_DIR/models" ]]; then
    sz=$(du -sh "$NEURAL_DIR/models" 2>/dev/null | cut -f1)
    echo "  ✓ $NEURAL_DIR/models ($sz)"
fi

# 5. Reminder about auxiliary services ─────────────────────────────────
say "Auxiliary services NOT touched (restart manually if you want):"
for pat in 'live_server.py' 'mcp_local.py'; do
    pids=$(pgrep -f "$pat" 2>/dev/null | tr '\n' ' ' || true)
    if [[ -n "$pids" ]]; then
        echo "  · $pat — PID(s):$pids"
    fi
done

if (( DRY )); then
    echo
    warn "DRY RUN — no changes made. Drop --dry to actually run."
else
    echo
    ok "Ready. Launch hermes for a fresh boot."
fi
