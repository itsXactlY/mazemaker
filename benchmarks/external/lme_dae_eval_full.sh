#!/usr/bin/env bash
# Full LongMemEval-S DAE sweep (~7.5h on dev box).
# Run manually when ready; the orchestrator drives three sequential
# longmemeval_s.py subprocesses and writes a verdict markdown.
set -euo pipefail
cd "$(dirname "$0")"
exec python lme_dae_eval.py --self-weight 0.4 --neighbour-k 20 --tag dae_2026-05-11
