#!/usr/bin/env python3
"""BEAM-10M judge orchestrator.

Consumes a ``beam_10m_generations_*.json`` produced by ``beam_10m.py``
and emits a ``beam_10m_judged_*.json`` with per-rubric-item scores
aggregated to per-question / per-ability / per-model / overall.

The judge protocol is the official BEAM rubric-by-rubric template
(``unified_llm_judge_base_prompt`` in BEAM/src/prompts.py): for every
``(question, llm_response, rubric_item)`` triple the judge model emits
``{"score": 1.0|0.5|0.0, "reason": "..."}``. The question score is the
mean of its rubric-item scores; ability/model/overall scores are means
of the contributing question scores.

This script writes the judge-input JSONL to a staging path and prints
the path for a downstream Claude sub-agent to consume. The judge is
NOT run inline -- the operator (or an outer harness) hands the JSONL
to a sub-agent in batches of ~50 items and writes the returned scores
back. This keeps the python harness pure data-plumbing and lets the
LLM judging happen through whichever sub-agent / API the operator
prefers (default: Claude Sonnet 4.6 sub-agent; alternative: BEAM's
gpt-4.1-mini for direct comparability).

Usage
-----
    # Stage: explode generations -> per-rubric-item judge jobs
    python beam_10m_judge.py stage path/to/beam_10m_generations_*.json
        -> writes beam_10m_judge_jobs_<ts>.jsonl + a sidecar manifest

    # Aggregate: after a sub-agent fills in scores per job
    python beam_10m_judge.py aggregate path/to/beam_10m_judge_scored_*.jsonl
        -> writes beam_10m_judged_<ts>.json with the final scores + matrix
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Stage: explode generations into per-rubric-item judge jobs
# ---------------------------------------------------------------------------

def stage(gen_path: Path) -> Path:
    payload = json.loads(gen_path.read_text())
    gens_by_model: dict[str, list[dict[str, Any]]] = payload.get(
        "generations_by_model", {})

    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jobs_path = RESULTS_DIR / f"beam_10m_judge_jobs_{ts}.jsonl"
    manifest_path = RESULTS_DIR / f"beam_10m_judge_manifest_{ts}.json"

    job_id = 0
    counts: dict[str, int] = {}
    with jobs_path.open("w") as fh:
        for model, rows in gens_by_model.items():
            for row in rows:
                rubric = row.get("rubric") or []
                if not rubric:
                    continue
                for r_idx, rubric_item in enumerate(rubric):
                    job = {
                        "job_id": job_id,
                        "model": model,
                        "conv": row["conv"],
                        "ability": row["ability"],
                        "q_idx": row["q_idx"],
                        "question": row["question"],
                        "rubric_item_idx": r_idx,
                        "rubric_item": rubric_item,
                        "llm_response": row.get("llm_response", ""),
                    }
                    fh.write(json.dumps(job) + "\n")
                    counts[model] = counts.get(model, 0) + 1
                    job_id += 1

    manifest_path.write_text(json.dumps({
        "schema": "beam_10m_judge_jobs.v1",
        "staged_at": ts,
        "source_generations": str(gen_path),
        "jobs_path": str(jobs_path),
        "total_jobs": job_id,
        "jobs_per_model": counts,
        "judge_prompt_source": "BEAM/src/prompts.py:unified_llm_judge_base_prompt",
        "judge_protocol": (
            "For each job, judge fills out {score: 1.0|0.5|0.0, reason: <str>}. "
            "Score 1.0 = full compliance with rubric item; 0.5 = partial; "
            "0.0 = no compliance OR response non-responsive to question."),
    }, indent=2))

    print(f"[beam-judge] staged {job_id} judge jobs from {len(gens_by_model)} models")
    print(f"[beam-judge] jobs: {jobs_path}")
    print(f"[beam-judge] manifest: {manifest_path}")
    return jobs_path


# ---------------------------------------------------------------------------
# Aggregate: post-judging, compute the per-ability/per-model matrix
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def aggregate(scored_path: Path) -> Path:
    """Consume a JSONL where each line is a job augmented with a
    'judge_score' float field (and optional 'judge_reason')."""
    # Group: (model, conv, ability, q_idx) -> list[score]
    per_q: dict[tuple[str, int, str, int], list[float]] = {}
    per_q_meta: dict[tuple[str, int, str, int], dict[str, Any]] = {}

    n = 0
    with scored_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            key = (job["model"], job["conv"], job["ability"], job["q_idx"])
            score = job.get("judge_score")
            if score is None:
                continue
            per_q.setdefault(key, []).append(float(score))
            per_q_meta.setdefault(key, {
                "question": job.get("question", ""),
                "model": job["model"],
                "conv": job["conv"],
                "ability": job["ability"],
                "q_idx": job["q_idx"],
            })
            n += 1

    # Question scores = mean of rubric-item scores
    question_scores: list[dict[str, Any]] = []
    for key, scores in per_q.items():
        meta = per_q_meta[key]
        meta["question_score"] = round(_mean(scores), 4)
        meta["n_rubric_items"] = len(scores)
        question_scores.append(meta)

    # Per-(model, ability)
    matrix: dict[str, dict[str, list[float]]] = {}
    for qs in question_scores:
        matrix.setdefault(qs["model"], {}).setdefault(
            qs["ability"], []).append(qs["question_score"])

    summary: dict[str, dict[str, Any]] = {}
    for model, by_ab in matrix.items():
        by_ability = {ab: round(_mean(v), 4) for ab, v in by_ab.items()}
        all_q = [s for v in by_ab.values() for s in v]
        summary[model] = {
            "overall": round(_mean(all_q), 4),
            "n_questions": len(all_q),
            "by_ability": by_ability,
        }

    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"beam_10m_judged_{ts}.json"
    out_path.write_text(json.dumps({
        "schema": "beam_10m_judged.v1",
        "aggregated_at": ts,
        "source_scored": str(scored_path),
        "n_judge_observations": n,
        "summary_by_model": summary,
        "per_question": question_scores,
    }, indent=2))

    print(f"\n[beam-judge] aggregated {n} judge observations -> {out_path}")
    print("\n=== Per-model BEAM-10M scores ===")
    for model, s in summary.items():
        print(f"  {model}: overall = {s['overall']:.4f} (n={s['n_questions']})")
        for ab, v in sorted(s["by_ability"].items()):
            print(f"      {ab:<28} {v:.4f}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="BEAM-10M judge orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_stage = sub.add_parser("stage", help="Explode generations into per-rubric judge jobs (JSONL)")
    p_stage.add_argument("generations_json", type=Path)

    p_agg = sub.add_parser("aggregate", help="Aggregate filled-in scored JSONL into the final matrix")
    p_agg.add_argument("scored_jsonl", type=Path)

    args = p.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.cmd == "stage":
        stage(args.generations_json)
    elif args.cmd == "aggregate":
        aggregate(args.scored_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
