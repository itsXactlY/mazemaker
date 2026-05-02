#!/bin/bash
# codex_todo_sweep.sh — daily codex-driven audit of TODO/FIXME/XXX/HACK
# comments in the codebase. Compounds the codex auxiliary surface.
#
# Per Tito 2026-05-02: "compound usefulness" — codex buildout should keep
# growing into the gaps the builder doesn't have time to manually sweep.
#
# Pattern:
#   1. Find all TODO/FIXME/XXX/HACK comments in python/ + tools/
#   2. For each, capture: file:line, comment text, git blame age
#   3. Bundle to gpt-5.4, ask: which are stale / still valid / trivially fixable
#   4. Emit per-comment classification + suggested action
#   5. Stale or trivially-fixable items → bridge FYI to builder
#
# Output: ~/.neural_memory/codex-todo-sweeps/<ts>-report.md
# Cron: daily 05:00 (offset from codex-archaeology 04:00) via plist

set -uo pipefail

LANE="${NM_LANE:-cron}"
MODEL="${TODO_SWEEP_MODEL:-gpt-5.4}"
REPO="/Users/tito/lWORKSPACEl/research/neural-memory"
OUT_DIR="${HOME}/.neural_memory/codex-todo-sweeps"
PRIMER="${HOME}/.neural_memory/codex-orchestrator/project-primer.md"
CODEX="/Applications/Codex.app/Contents/Resources/codex"
BRIDGE_CLI="/Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph/plugins/hermes-skills-layer/plugins/claude-hermes-bridge/scripts/agent_bridge_mcp.mjs"

SWEEPER_AGENT="codex-todo-sweeper"
CALLER_AGENT="claude-code-${LANE}"

mkdir -p "$OUT_DIR"

TS=$(date +%Y%m%d-%H%M%S)
OUT="${OUT_DIR}/${TS}-report.md"
BUNDLE=$(mktemp -t codex-todo-bundle.XXXXXX.txt)

# Gather all TODO/FIXME/XXX/HACK in python/ + tools/ with file:line + age
{
    cd "$REPO" || exit 1
    grep -rn -E '#\s*(TODO|FIXME|XXX|HACK)|//\s*(TODO|FIXME|XXX|HACK)' python/ tools/ 2>/dev/null \
        | head -100 \
        | while IFS= read -r line; do
            file=$(echo "$line" | cut -d: -f1)
            lineno=$(echo "$line" | cut -d: -f2)
            text=$(echo "$line" | cut -d: -f3-)
            # Get blame age (commit date)
            blame_date=$(git blame -L "${lineno},${lineno}" --date=short -- "$file" 2>/dev/null | awk '{print $4}')
            echo "${file}:${lineno} (${blame_date:-unknown}) — ${text}"
        done
} > "$BUNDLE"

TODO_COUNT=$(wc -l < "$BUNDLE" | tr -d ' ')

if [ "$TODO_COUNT" -eq 0 ]; then
    echo "No TODOs found — clean codebase." > "$OUT"
    rm -f "$BUNDLE"
    exit 0
fi

PRIMER_PREAMBLE=""
[ -f "$PRIMER" ] && PRIMER_PREAMBLE="Read project-primer at ${PRIMER} for context first.

"

PROMPT="${PRIMER_PREAMBLE}You are the codex-todo-sweeper. Read the bundle of TODO/FIXME/XXX/HACK comments at ${BUNDLE} (use 'cat ${BUNDLE}'). Each line: file:line (commit-date) — comment text.

For each TODO, classify:
  STALE       — comment references files/concepts/tasks no longer relevant (use git log/blame to verify the surrounding code is gone or refactored)
  STILL-VALID — comment is current and the work is still needed
  TRIVIAL     — comment describes a 1-2 line fix that could be done immediately
  UNCLEAR     — insufficient context to classify; needs human review

Output as markdown table:
  | file:line | age | category | classification | reasoning |

Then a separate section '## Recommended actions' with up to 5 highest-priority items the builder should address (TRIVIAL first, then STILL-VALID > 90 days old).

CONSTRAINTS:
- Sandbox is read-only (cannot edit code, only read + classify)
- Limit total output to ~3000 words MAX
- Be terse, evidence-grounded — cite file:line when claiming staleness
- Don't manufacture findings; if comment text is too cryptic, classify UNCLEAR

Output to stdout (wrapper captures)."

# Header
{
    echo "# Codex TODO Sweep Report"
    echo "**Date:** $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "**Model:** ${MODEL}"
    echo "**Repo HEAD:** $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null)"
    echo "**TODO count:** ${TODO_COUNT}"
    echo ""
    echo "---"
    echo ""
} > "${OUT}.partial"

"$CODEX" exec \
    --model "$MODEL" \
    --sandbox read-only \
    --cd "$REPO" \
    "$PROMPT" \
    >> "${OUT}.partial" 2>/dev/null
RC=$?

if [ "$RC" = "0" ] && [ "$(wc -c < "${OUT}.partial" 2>/dev/null || echo 0)" -gt 800 ]; then
    mv "${OUT}.partial" "$OUT"
else
    OUT="${OUT}.partial"
fi

# Bridge FYI — count STALE + TRIVIAL items as actionable
ACTIONABLE=$(grep -cE 'STALE|TRIVIAL' "$OUT" 2>/dev/null || echo 0)

if [ -x "$(command -v node)" ] && [ -f "$BRIDGE_CLI" ]; then
    node "$BRIDGE_CLI" send \
        --from "$SWEEPER_AGENT" \
        --to "$CALLER_AGENT" \
        --subject "todo-sweep: ${TODO_COUNT} TODOs scanned, ~${ACTIONABLE} actionable (stale/trivial)" \
        --body "Codex TODO sweep complete.
TODO count: ${TODO_COUNT}
Actionable (STALE or TRIVIAL): ${ACTIONABLE}
Report: ${OUT}
Builder review: cat ${OUT} → focus on '## Recommended actions' section." \
        --urgency low \
        2>/dev/null || true
fi

rm -f "$BUNDLE"
echo "→ saved: $OUT"
echo "→ todo_count=${TODO_COUNT} actionable=${ACTIONABLE}"
exit $RC
