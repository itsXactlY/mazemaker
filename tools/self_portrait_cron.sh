#!/usr/bin/env bash
# self_portrait_cron.sh — every-6-hour scaffolder for the agent self-portrait cycle.
#
# Per packet S-PORTRAIT-3 + the self-portrait feature handoff
# (project_self_portrait_feature_handoff_2026-05-02.md):
#
# This wrapper is invoked by com.ae.neural-self-portrait.plist on the
# 06:00/12:00/18:00/00:00 cadence. For each agent in the v0 set
# (claude-code, hermes, codex), it invokes
#   python3 tools/self_portrait_cycle.py --agent <agent> --mode scaffold
# which writes that agent's substrate input packet (STEP 1 of the cycle).
#
# The wrapper does NOT do the complete-cycle (with reasoning_text +
# prompt_text). That's the agent's job per Tito rule #1 — agents come back
# later, read their input packet, write reasoning + prompt, then invoke
# complete-cycle themselves. This wrapper just SCAFFOLDS the input every
# 6 hours.
#
# Error-handling stance:
#   - `set -uo pipefail` (NOT `set -e`): we want to continue past per-agent
#     failures and report at the end, so a single agent's substrate hiccup
#     doesn't starve the other two.
#   - Exit 0 only if ALL agents scaffolded successfully.
#   - Exit non-zero if ANY agent failed, so launchd's
#     KeepAlive=SuccessfulExit-false will retry.

set -uo pipefail

REPO_ROOT="/Users/tito/lWORKSPACEl/research/neural-memory"
CYCLE_SCRIPT="${REPO_ROOT}/tools/self_portrait_cycle.py"
LOG_DIR="${HOME}/Library/Logs/ae"
LOG_FILE="${LOG_DIR}/neural-self-portrait.log"

mkdir -p "${LOG_DIR}"

# v0 agent set — per S-PORTRAIT-1 substrate helpers (claude-code, hermes,
# codex). Stored as a bash array so future-extensibility is one append away.
AGENTS=(
    "claude-code"
    "hermes"
    "codex"
)

ts() { date -Iseconds; }

log() {
    # Single-line log entries with ISO-8601 timestamp prefix; both stdout
    # (for launchd's StandardOutPath) and the rolling log file.
    local msg="$1"
    echo "$(ts)  ${msg}"
    echo "$(ts)  ${msg}" >> "${LOG_FILE}"
}

cd "${REPO_ROOT}"

log "self-portrait cron START  agents=${AGENTS[*]}"

FAIL_COUNT=0
FAILED_AGENTS=()

for AGENT in "${AGENTS[@]}"; do
    log "agent=${AGENT}  scaffold START"
    # Capture stdout+stderr so the per-agent log is contiguous in
    # neural-self-portrait.stdout.log (launchd separates them otherwise).
    if python3 "${CYCLE_SCRIPT}" --agent "${AGENT}" --mode scaffold 2>&1; then
        RC=$?
        log "agent=${AGENT}  scaffold OK  rc=${RC}"
    else
        RC=$?
        log "agent=${AGENT}  scaffold FAIL  rc=${RC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_AGENTS+=("${AGENT}")
    fi
done

if [[ "${FAIL_COUNT}" -eq 0 ]]; then
    log "self-portrait cron DONE  all_ok=true"
    exit 0
else
    log "self-portrait cron DONE  fail_count=${FAIL_COUNT}  failed=${FAILED_AGENTS[*]}"
    exit 1
fi
