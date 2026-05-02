#!/bin/bash
# codex_archaeology_pick.sh — daily cron driver. Picks a random python module
# and dispatches a Codex archaeology run on it.
#
# Mirrors the Claude reconciliation-reviewer pattern but cheaper per call.
# Claude reconciliation-reviewer (60min cron) keeps doing its interpretive
# work; this gives us a SECOND independent opinion on different modules
# without burning Claude tokens.

set -uo pipefail

REPO="/Users/tito/lWORKSPACEl/research/neural-memory"
WRAPPER="${REPO}/tools/codex_subagent.sh"

# Pool of meaningful modules to rotate through
MODULES=(
    "python/memory_client.py"
    "python/neural_memory.py"
    "python/mssql_store.py"
    "python/embed_provider.py"
    "python/dream_engine.py"
    "python/cpp_bridge.py"
    "python/access_logger.py"
    "tools/nm.py"
    "tools/ingest_ae_corpus.py"
    "tools/observer.py"
    "tools/scorer_ablation.py"
)

PICK="${MODULES[$RANDOM % ${#MODULES[@]}]}"
TOPIC=$(basename "$PICK" .py)

PROMPT="Code archaeology on ${PICK}.

You are a second-opinion reviewer running parallel to Claude's reconciliation-reviewer. Be terse.

Tasks:
1. Read ${PICK}.
2. Look for: dead code (no callers per grep), drift between docstring and behavior, silent failure modes (bare except, return [] on error), missing tests for public APIs.
3. For each finding: file:line, severity (LOW/MED/HIGH), what's wrong, suggested fix in <=2 sentences.
4. If the file is clean, say so in one line. Don't manufacture findings.

Constraints: read-only sandbox, no edits. End with a one-line verdict (CLEAN / N findings)."

exec "$WRAPPER" archaeology "$TOPIC" "$PROMPT"
