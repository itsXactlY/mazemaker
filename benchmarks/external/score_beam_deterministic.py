#!/usr/bin/env python3
"""Deterministic scorer for BEAM-10M probing_questions.

Per question (and per rubric item):
  - For abstention questions: check whether the answer is a refusal
    (synonym set). Score 1.0 per rubric item if refusal, 0.0 otherwise.
  - For other abilities: each rubric item carries key tokens (extracted
    by beam_to_verified.py). All key tokens of an item must appear in
    the answer (substring, case-insensitive). Score per item = 0/1.

The per-question score is the mean over its rubric items (matches the
BEAM official aggregation).

Usage:
  python score_beam_deterministic.py \\
    --responses /path/to/mazemaker_responses_convN.jsonl \\
    --questions /tmp/beam_verified/conv-N/verified_questions.json \\
    --out       /path/to/beam_det_scores_convN.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ABSTAIN_TOKENS = {
    "no information", "not information", "not mentioned", "not specified",
    "not provided", "no record", "no specific", "no concrete",
    "no grounding", "not described", "not present", "no details",
    "no data", "no evidence", "doesn't say", "does not say",
    "does not mention", "cannot answer", "i cannot", "unknown",
    "abstain", "none of the snippets", "no retrieved",
    "based on the provided chat, there is no",
    "based on the chat, there is no",
    "the provided chat does not",
}

ABSTAIN_ABILITIES = {"abstention"}


def normalize(s: str) -> str:
    s = (s or "").lower()
    # Preserve percent expressions as Npct so numeric+pct rubric tokens match
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1pct", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_answer(response: str) -> str:
    """Pull Phase-2 answer; fall back to full response."""
    if not response:
        return ""
    if "=== ANSWER ===" in response:
        return response.split("=== ANSWER ===", 1)[1].strip()
    return response.strip()


def is_abstain(answer_norm: str) -> bool:
    if not answer_norm:
        return True
    for tok in ABSTAIN_TOKENS:
        if tok in answer_norm:
            return True
    return False


def score_rubric_item(answer: str, key_tokens: list[str], ability: str) -> tuple[float, str]:
    a_norm = normalize(answer)
    if ability in ABSTAIN_ABILITIES:
        if is_abstain(a_norm):
            return 1.0, "abstain_correct"
        return 0.0, "expected_refusal_got_hallucination"
    if not key_tokens:
        # No tokens to check — treat as soft pass if answer non-empty
        return (1.0, "no_tokens_pass") if a_norm else (0.0, "empty_answer")
    # All key tokens must appear
    missing = [t for t in key_tokens if t not in a_norm]
    if not missing:
        return 1.0, "all_tokens_present"
    if len(missing) == len(key_tokens):
        return 0.0, "no_tokens_present"
    # Partial credit: fraction of tokens present
    present = len(key_tokens) - len(missing)
    return present / len(key_tokens), f"partial:{present}/{len(key_tokens)}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    qs = {q["qid"]: q for q in json.loads(args.questions.read_text())["questions"]}
    rows = []
    by_ability: dict[str, list[float]] = defaultdict(list)
    total: list[float] = []
    by_q: dict[str, list[float]] = defaultdict(list)

    with args.responses.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            qid = r.get("qid")
            q = qs.get(qid)
            if not q: continue
            answer = extract_answer(r.get("llm_response") or "")
            rubric_items = q.get("rubric", [])
            key_tokens_per_item = q.get("rubric_key_tokens", [])
            per_item_scores = []
            per_item_details = []
            for i, ri in enumerate(rubric_items):
                toks = key_tokens_per_item[i] if i < len(key_tokens_per_item) else []
                s, reason = score_rubric_item(answer, toks, q["ability"])
                per_item_scores.append(s)
                per_item_details.append({"rubric_item": ri[:120], "key_tokens": toks, "score": s, "reason": reason})
            q_score = sum(per_item_scores) / len(per_item_scores) if per_item_scores else 0.0
            row = {
                "qid": qid,
                "conv": r.get("conv"),
                "ability": q["ability"],
                "gold": q.get("gold", "")[:140],
                "answer_extracted": answer[:240],
                "n_rubric_items": len(rubric_items),
                "rubric_scores": per_item_details,
                "q_score": q_score,
                "model": r.get("model"),
            }
            rows.append(row)
            total.append(q_score)
            by_ability[q["ability"]].append(q_score)
            by_q[qid].append(q_score)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(total)
    overall = sum(total) / n if n else 0.0

    if not args.quiet:
        print(f"\n=== BEAM deterministic: {args.responses.name} ===")
        print(f"  n_questions={n}  overall={overall:.4f}")
        for ab in sorted(by_ability):
            arr = by_ability[ab]
            print(f"    {ab:<24} n={len(arr):>3}  mean={sum(arr)/len(arr):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
