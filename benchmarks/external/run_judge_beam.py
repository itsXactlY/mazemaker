#!/usr/bin/env python3
"""BEAM-10M judge driver — runs the OFFICIAL BEAM rubric prompt
(BEAM/src/prompts.py:unified_llm_judge_base_prompt) via OpenAI API
(gpt-5-nano default) against staged judge jobs.

Comparable methodology to BEAM-published Hindsight 64.1%. Same prompt,
same scoring scale (1.0 / 0.5 / 0.0), same per-rubric-item granularity.

Usage:
  python run_judge_beam.py \\
    --jobs <judge_jobs.jsonl> \\
    --out  <judge_jobs_scored.jsonl> \\
    [--model gpt-5-nano-2025-08-07] [--parallel 6]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

KEY_FILE = Path.home() / ".benchkey"
LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_JUDGE_MODEL = os.environ.get("MM_JUDGE_MODEL", "gpt-5-nano-2025-08-07")

JUDGE_PROMPT = """You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
- **0.5 (Partial Compliance)**: Partially complies.
- **0.0 (No Compliance)**: Fails to comply.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""

_BLOCK_RE = re.compile(r"\{[^{}]*\"score\"[^{}]*\}", re.DOTALL)
_REASONING_MODELS = ("gpt-5", "o1", "o3", "o4")


def load_key() -> str:
    if not KEY_FILE.exists():
        raise SystemExit(f"benchmark key not found at {KEY_FILE}")
    k = KEY_FILE.read_text().strip()
    if not k.startswith("sk-"):
        raise SystemExit(f"key in {KEY_FILE} does not look like OpenAI")
    return k


def build_prompt(row: dict) -> str:
    return (JUDGE_PROMPT
            .replace("<question>", row["question"])
            .replace("<rubric_item>", row["rubric_item"])
            .replace("<llm_response>", row.get("llm_response", "") or ""))


def call_judge(api_key: str, prompt: str, model: str,
               timeout: int = 120, max_attempts: int = 4) -> tuple[str, dict]:
    is_reasoning = any(model.startswith(p) for p in _REASONING_MODELS)
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_reasoning:
        body["max_completion_tokens"] = 4096
        body["reasoning_effort"] = "minimal"
    else:
        body["max_tokens"] = 512
        body["temperature"] = 0.0
    payload = json.dumps(body).encode()
    last_err = None
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(
                f"{LLM_BASE_URL}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
            text = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {}) or {}
            return text, usage
        except urllib.error.HTTPError as e:
            body_err = e.read().decode("utf-8", errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {e.code}: {body_err}")
            else:
                raise RuntimeError(f"HTTP {e.code}: {body_err}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
        if attempt < max_attempts - 1:
            time.sleep(2.0 * (2 ** attempt))
    raise RuntimeError(f"judge call failed after {max_attempts} attempts: {last_err}")


def parse_score(raw: str) -> tuple[float | None, str]:
    # Try direct JSON parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "score" in obj:
            return float(obj["score"]), obj.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find JSON-shaped block
    for m in _BLOCK_RE.findall(raw):
        try:
            obj = json.loads(m)
            return float(obj["score"]), obj.get("reason", "")
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
    # Fallback: look for "score": N
    m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw)
    if m:
        return float(m.group(1)), raw[:200]
    return None, raw[:200]


_WRITE_LOCK = threading.Lock()


def judge_one(api_key: str, model: str, row: dict) -> dict:
    t0 = time.perf_counter()
    try:
        prompt = build_prompt(row)
        raw, usage = call_judge(api_key, prompt, model)
        score, reason = parse_score(raw)
        row_out = {
            **row,
            "judge_score": score,
            "judge_reason": reason,
            "judge_model": model,
            "judge_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        return row_out
    except Exception as e:
        return {**row, "judge_score": None, "judge_error": str(e)[:300],
                "judge_latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--parallel", type=int, default=6)
    args = p.parse_args()

    api_key = load_key()
    done_ids: set[int] = set()
    if args.out.exists():
        with args.out.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("judge_score") is not None:
                        done_ids.add(int(r["job_id"]))
                except Exception:
                    continue

    jobs: list[dict] = []
    with args.jobs.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if int(r["job_id"]) in done_ids:
                continue
            jobs.append(r)

    print(f"[judge] key={api_key[:14]}...  model={args.model}", flush=True)
    print(f"[judge] {args.jobs.name}: total={len(jobs)+len(done_ids)} done={len(done_ids)} pending={len(jobs)}", flush=True)

    if not jobs:
        print(f"[judge] nothing to do")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = err = 0
    with args.out.open("a") as out_fh, concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(judge_one, api_key, args.model, j) for j in jobs]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            r = fut.result()
            if r.get("judge_score") is None:
                err += 1
            else:
                ok += 1
            with _WRITE_LOCK:
                out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_fh.flush()
            if i % 25 == 0 or i == len(futures):
                print(f"  [{i}/{len(futures)}] ok={ok} err={err}", flush=True)
    print(f"[judge] done {args.out.name}: ok={ok} err={err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
