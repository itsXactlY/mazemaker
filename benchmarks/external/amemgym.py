#!/usr/bin/env python3
"""AMemGym — external on-policy memory benchmark. STATUS: QUEUED.

AMemGym (arXiv:2603.01966, AGI-Eval) is the first *interactive, on-policy*
conversational-memory benchmark. Unlike static chat-log benches, the assistant
talks to an LLM-simulated user whose latent state evolves across periods; the
paper's central finding is "reuse bias" — off-policy (static) evaluation ranks
memory policies differently from on-policy, so a static run actively
MISREPRESENTS a memory system's real behaviour.

Data (public, CC-BY-4.0): huggingface.co/datasets/AGI-Eval/AMemGym → v1.base.json
  20 persona instances · each: user_profile, state_schema (evolving latent
  variables), 11 evolution `periods`, 10 `qas`. Code: AGI-Eval-Official/amemgym.

Why this harness is QUEUED, not VERIFIED
----------------------------------------
A faithful AMemGym result requires the on-policy loop — an LLM assistant driving
Mazemaker's write/read policy while an LLM user-simulator advances the latent
state period-by-period — and the paper's Normalized Memory Score (which factors
out the model's reasoning upper bound) plus the write/read/utilization failure
decomposition. Running Mazemaker over the static transcript would produce exactly
the off-policy number AMemGym was built to discredit. We do not publish a number
until the on-policy harness exists and emits a result JSON — same VERIFIED/QUEUED
discipline as the rest of benchmarks/external/.

This module currently: (1) loads + validates the public data, (2) reports the
corpus shape, (3) locks the methodology below. It intentionally emits NO score.

Planned on-policy harness (the remaining work):
  - user-simulator: LLM conditioned on user_profile + the period's state delta.
  - assistant: LLM policy that calls Mazemaker remember()/recall() as its memory.
  - run 11 periods per persona on-policy; then score the 10 qas.
  - metrics: Normalized Memory Score + write/read/utilization failure rates,
    reported vs a long-context-only and a vanilla-RAG baseline (same policy LLM).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "amemgym" / "v1.base.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--data", default=str(DATA))
    args = p.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"[amemgym] data not found at {path}\n"
              f"  download: huggingface.co/datasets/AGI-Eval/AMemGym → v1.base/data.json",
              file=sys.stderr)
        return 2

    raw = path.read_bytes()
    data = json.loads(raw)
    n_personas = len(data)
    n_periods = sum(len(x.get("periods", [])) for x in data)
    n_qas = sum(len(x.get("qas", [])) for x in data)
    state_vars = sum(len(x.get("state_schema", {})) for x in data)

    print("=" * 64)
    print("AMemGym — on-policy conversational memory benchmark")
    print(f"  data sha256 : {hashlib.sha256(raw).hexdigest()[:12]}…")
    print(f"  personas    : {n_personas}")
    print(f"  evolution periods (total): {n_periods}")
    print(f"  QA pairs (total)         : {n_qas}")
    print(f"  latent state variables   : {state_vars}")
    print(f"  STATUS      : QUEUED — methodology locked; on-policy simulator")
    print(f"                loop not yet wired. NO number emitted by design.")
    print(f"                (A static run would exhibit the 'reuse bias' the")
    print(f"                 benchmark exists to expose.)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
