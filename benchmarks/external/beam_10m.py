#!/usr/bin/env python3
"""BEAM-10M external benchmark harness for Mazemaker.

Runs BEAM-10M (Tavakoli et al., ICLR 2026) -- 10 conversations at ~10M
tokens each, 20 probing questions per conversation across 10 memory
abilities = 200 total questions.

Two-phase design:
  1. ingest + generate   (this script)
       -> per-conv isolated SQLite at /tmp/beam-10m/conv-<i>/memory.db,
          full retrieval pipeline (hybrid + ColBERT@1.5), ollama
          generation for each (question, model) pair.
       -> writes a `beam_10m_generations_*.json` artefact with the
          question, retrieved memories, gold answer, rubric, and the
          model's free-text response.
  2. judge   (beam_10m_judge.py, separate)
       -> consumes the generations JSON, scores each (response,
          rubric_item) via Claude sub-agents using BEAM's official
          unified_llm_judge_base_prompt template, aggregates per
          ability / per model / overall.

Why two phases: generation is GPU- and ollama-bound; judging is API/
sub-agent-bound. Decoupling lets us re-judge without re-generating,
and lets the judge run in parallel against a finished artefact.

Usage
-----
    # full sweep (~80-180 min wall-time depending on models)
    python beam_10m.py --convs 1-10 \\
        --models gemma3:270m,qwen2.5:0.5b,gemma3:1b,gpt-oss:120b-cloud

    # smoke (1 conv, 2 questions per ability, 1 model)
    python beam_10m.py --convs 1 --limit-questions 1 --models gemma3:1b

    # reuse existing per-conv DBs (skip ingest):
    python beam_10m.py --skip-ingest --convs 1-10 --models gemma3:1b
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PY_DIR = ROOT / "python"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from memory_client import Mazemaker  # noqa: E402

BEAM_REPO = Path(os.environ.get("BEAM_REPO", "/tmp/BEAM"))
CHATS_DIR = BEAM_REPO / "chats" / "10M"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
INGEST_BASE = Path(os.environ.get("BEAM_INGEST_BASE", "/tmp/beam-10m"))

ABILITIES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_chat_turns(conv_idx: int) -> list[dict[str, Any]]:
    """Flatten chats/10M/<i>/chat.json into a list of turn dicts.

    The on-disk shape is:
        chat.json = [ {"plan-0": [batches]}, {"plan-1": [batches]}, ... ]
        batch     = {"batch_number": int, "time_anchor": str, "turns": [ [turn, ...], ... ]}
        turn      = {"role": "user"|"assistant", "id": int, "content": str, "index": str,
                     "time_anchor": str, "question_type": str}
    """
    path = CHATS_DIR / str(conv_idx) / "chat.json"
    if not path.exists():
        raise FileNotFoundError(f"BEAM conv {conv_idx} chat not found: {path}")
    chat = json.loads(path.read_text())
    flat: list[dict[str, Any]] = []
    for plan_obj in chat:
        for plan_key, batches in plan_obj.items():
            plan_idx = int(plan_key.split("-", 1)[1])
            for batch in batches:
                batch_num = int(batch.get("batch_number", 0))
                batch_anchor = batch.get("time_anchor", "") or ""
                for grp_idx, group in enumerate(batch.get("turns", [])):
                    for t_idx, turn in enumerate(group):
                        content = (turn.get("content") or "").strip()
                        if not content:
                            continue
                        flat.append({
                            "plan": plan_idx,
                            "batch": batch_num,
                            "group": grp_idx,
                            "tidx": t_idx,
                            "role": turn.get("role", "?"),
                            "time_anchor": turn.get("time_anchor") or batch_anchor,
                            "content": content,
                            "id": turn.get("id"),
                            "index": turn.get("index", ""),
                            "question_type": turn.get("question_type", ""),
                        })
    return flat


def load_probing_questions(conv_idx: int) -> dict[str, list[dict[str, Any]]]:
    path = CHATS_DIR / str(conv_idx) / "probing_questions" / "probing_questions.json"
    if not path.exists():
        raise FileNotFoundError(f"BEAM conv {conv_idx} probing questions not found: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Mazemaker bring-up
# ---------------------------------------------------------------------------

def _build_engine(db_path: Path, backend: str) -> Mazemaker:
    # Use the centralised quality config — every Mazemaker feature ON
    # (C++ kNN, advanced multi-channel, rerank, HNSW auto, DAE, ColBERT).
    sys.path.insert(0, str(
        Path(__file__).resolve().parent.parent /
        "neural_memory_benchmark" / "mm_10m_eval" / "runners"
    ))
    from engine_config import build_quality_engine
    return build_quality_engine(db_path, backend=backend)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_conv(conv_idx: int, args) -> Path:
    db = INGEST_BASE / f"conv-{conv_idx}" / "memory.db"
    if args.skip_ingest and db.exists():
        n_existing = 0
        try:
            import sqlite3
            con = sqlite3.connect(str(db))
            n_existing = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            con.close()
        except Exception:
            pass
        print(f"[beam] conv {conv_idx}: skip-ingest, reusing {db} "
              f"({n_existing} memories)", flush=True)
        return db

    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()

    turns = load_chat_turns(conv_idx)
    if args.max_turns_per_conv > 0:
        turns = turns[: args.max_turns_per_conv]

    nm = _build_engine(db, args.backend)
    t0 = time.perf_counter()
    n_written = 0
    last_log = t0
    for i, t in enumerate(turns):
        anchor = t["time_anchor"]
        prefix = f"[{anchor}] " if anchor else ""
        text = f"{prefix}{t['role']}: {t['content']}"
        label = f"p{t['plan']}.b{t['batch']}.g{t['group']}.t{t['tidx']}"
        try:
            nm.remember(text, label=label,
                        detect_conflicts=False, auto_connect=False)
            n_written += 1
        except Exception as e:
            print(f"[beam] conv {conv_idx}: WARN ingest turn "
                  f"{i} failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)

        now = time.perf_counter()
        if now - last_log >= 30.0:
            elapsed = now - t0
            rate = (i + 1) / elapsed
            eta_s = (len(turns) - i - 1) / max(rate, 0.1)
            print(f"[beam] conv {conv_idx}: ingested "
                  f"{i + 1}/{len(turns)}  ({rate:.1f}/s, ETA {eta_s / 60:.1f} min)",
                  flush=True)
            last_log = now

    elapsed = time.perf_counter() - t0
    print(f"[beam] conv {conv_idx}: ingest done in {elapsed / 60:.1f} min "
          f"({n_written}/{len(turns)} turns) -> {db}", flush=True)
    try:
        nm.close()
    except Exception:
        pass
    return db


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
GEN_PROMPT = """You are a helpful assistant. The text below is your conversation history with the user, retrieved from your long-term memory. Use it as authoritative evidence of what was discussed and answer the user's question.

# CONVERSATION HISTORY (relevant excerpts, most-relevant first)
{context}

# USER QUESTION
{question}

# YOUR ANSWER (follow the question's shape EXACTLY)

Match your answer to what the question asks:
- "List N items in order" -> exactly N items, ordered, one per line
- "Reconstruct the sequence" -> ordered list covering every item present in the history
- "How many X across Y" -> the single number, with the contributing values
- "What is X" / "What rate / version / count" -> the single fact, in one short sentence
- "Summarize Y over period Z" -> 3-6 sentences covering distinct subtopics raised in that period
- "Have I done X?" -> direct yes/no with the contradicting evidence if any
- "What are my preferences for X?" -> the preferences as expressed in the history

Ground every claim in the history. If a specific fact the question asks for is GENUINELY absent from the history (not merely scattered, but truly not discussed), say so. Otherwise commit to an answer drawn from the excerpts — do not hedge, do not refuse, do not add generic best-practice advice. Do the synthesis the question requires."""


def _format_memory(idx: int, result: dict[str, Any]) -> str:
    label = result.get("label", "") or ""
    content = (result.get("content", "") or "").strip()
    # Strip the role: prefix we added during ingest if it's noisy, but
    # leave the time anchor in place since it matters for temporal Qs.
    return f"[{idx}] ({label}) {content}"


def _ollama_generate(model: str, prompt: str, timeout: int = 600,
                     num_ctx: int = 32768) -> tuple[str, dict[str, Any]]:
    import urllib.request
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip(), {
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "total_duration_ns": data.get("total_duration"),
    }


def generate_answers_for_conv(conv_idx: int,
                              db_path: Path,
                              models: list[str],
                              args) -> dict[str, list[dict[str, Any]]]:
    probing = load_probing_questions(conv_idx)
    nm = _build_engine(db_path, args.backend)

    out: dict[str, list[dict[str, Any]]] = {m: [] for m in models}

    # When --emit-prompts-only, we skip the model call entirely and emit
    # one entry per question (model column = "<prompts-only>") with the
    # full assembled prompt. The judge / external dispatcher then
    # generates the responses out-of-band (e.g., via a Claude sub-agent).
    prompts_only = bool(getattr(args, "emit_prompts_only", False))
    if prompts_only:
        out = {"<prompts-only>": []}

    for ability in ABILITIES:
        questions = probing.get(ability, [])
        if args.limit_questions > 0:
            questions = questions[: args.limit_questions]

        for q_idx, q in enumerate(questions):
            question = q.get("question", "")
            gold = q.get("answer", "")
            rubric = q.get("rubric", []) or []
            key_facts = q.get("key_facts_tested", []) or []
            difficulty = q.get("difficulty", "")
            qtype = q.get("question_type", "")

            t0 = time.perf_counter()
            try:
                results = nm.recall(
                    question, k=args.k,
                    hybrid=True,
                    enable_colbert=True,
                    colbert_weight=1.5,
                    enable_dae=True,
                    dae_weight=1.0,
                )
            except Exception as e:
                results = []
                print(f"[beam] conv {conv_idx} {ability}#{q_idx}: "
                      f"WARN recall failed: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
            recall_ms = (time.perf_counter() - t0) * 1000.0

            top = results[: args.k]
            context = "\n\n".join(_format_memory(i + 1, r) for i, r in enumerate(top))
            prompt = GEN_PROMPT.format(context=context, question=question)
            top_labels = [r.get("label", "") for r in top]

            if prompts_only:
                out["<prompts-only>"].append({
                    "conv": conv_idx,
                    "ability": ability,
                    "q_idx": q_idx,
                    "question": question,
                    "gold": gold,
                    "rubric": rubric,
                    "key_facts_tested": key_facts,
                    "difficulty": difficulty,
                    "question_type": qtype,
                    "n_retrieved": len(results),
                    "top_labels": top_labels,
                    "recall_ms": round(recall_ms, 1),
                    "prompt": prompt,
                })
                print(f"[beam] conv {conv_idx} {ability} {q_idx + 1}/"
                      f"{len(questions)}: prompt-emit recall={recall_ms:.0f}ms",
                      flush=True)
                continue

            for model in models:
                t1 = time.perf_counter()
                try:
                    llm_response, stats = _ollama_generate(model, prompt)
                    err = None
                except Exception as e:
                    llm_response = ""
                    stats = {}
                    err = f"{type(e).__name__}: {e}"
                gen_ms = (time.perf_counter() - t1) * 1000.0

                out[model].append({
                    "conv": conv_idx,
                    "ability": ability,
                    "q_idx": q_idx,
                    "question": question,
                    "gold": gold,
                    "rubric": rubric,
                    "key_facts_tested": key_facts,
                    "difficulty": difficulty,
                    "question_type": qtype,
                    "n_retrieved": len(results),
                    "top_labels": top_labels,
                    "recall_ms": round(recall_ms, 1),
                    "gen_ms": round(gen_ms, 1),
                    "llm_response": llm_response,
                    "ollama_stats": stats,
                    "error": err,
                })
            done_q = q_idx + 1
            n_q = len(questions)
            print(f"[beam] conv {conv_idx} {ability} {done_q}/{n_q}: "
                  f"recall={recall_ms:.0f}ms, models={len(models)}, "
                  f"total {sum(len(out[m]) for m in models)} entries",
                  flush=True)

    try:
        nm.close()
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def _parse_conv_arg(spec: str) -> list[int]:
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description="BEAM-10M external harness")
    p.add_argument("--convs", default="1-10",
                   help="Conversation indices, e.g. '1-10' or '1,2,3' (default: 1-10).")
    p.add_argument("--max-turns-per-conv", type=int, default=0,
                   help="Cap turns per conv for smoke runs (default 0 = all).")
    p.add_argument("--limit-questions", type=int, default=0,
                   help="Cap questions per ability (BEAM-10M ships 2 per ability; default 0 = all).")
    p.add_argument("--models", default="gemma3:1b",
                   help="Comma-separated ollama model names.")
    p.add_argument("--backend", default="auto",
                   help="Embedding backend (default 'auto' uses shared socket).")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Reuse existing per-conv DBs at $BEAM_INGEST_BASE/conv-<i>/memory.db.")
    p.add_argument("--ingest-only", action="store_true",
                   help="Run ingest only, skip generation.")
    p.add_argument("-k", "--k", type=int, default=10,
                   help="Top-k memories fed to the LLM (default 10).")
    p.add_argument("--emit-prompts-only", action="store_true",
                   help="Skip ollama generation; emit a prompts-only JSONL "
                        "for downstream dispatch (e.g., Claude sub-agent).")
    p.add_argument("--tag", default="",
                   help="Optional tag prefix on the result filename.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    conv_indices = _parse_conv_arg(args.convs)
    if not conv_indices:
        print("[beam] no convs selected", file=sys.stderr)
        return 1
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models and not args.ingest_only and not args.emit_prompts_only:
        print("[beam] no models selected", file=sys.stderr)
        return 1
    if args.emit_prompts_only:
        models = ["<prompts-only>"]

    print(f"[beam] convs: {conv_indices}", flush=True)
    print(f"[beam] models: {models}", flush=True)
    print(f"[beam] dataset: {CHATS_DIR}", flush=True)
    print(f"[beam] ingest dir: {INGEST_BASE}", flush=True)

    # Phase 1: ingest
    db_paths: dict[int, Path] = {}
    for c in conv_indices:
        try:
            db_paths[c] = ingest_conv(c, args)
        except Exception as e:
            print(f"[beam] conv {c}: ingest failed: "
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)

    if args.ingest_only:
        print(f"[beam] ingest-only done; {len(db_paths)} convs ready.", flush=True)
        return 0

    # Phase 2: generation
    all_results: dict[str, list[dict[str, Any]]] = {m: [] for m in models}
    for c in conv_indices:
        db = db_paths.get(c)
        if db is None:
            continue
        try:
            per_conv = generate_answers_for_conv(c, db, models, args)
        except Exception as e:
            print(f"[beam] conv {c}: generation failed: "
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
            continue
        for m in models:
            all_results[m].extend(per_conv[m])

    # Phase 3: write generations JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag_part = f"{args.tag}_" if args.tag else ""
    name = f"beam_10m_generations_{tag_part}{ts}.json"
    path = RESULTS_DIR / name

    payload = {
        "schema": "beam_10m_generations.v1",
        "timestamp": ts,
        "git_sha": _git_sha(),
        "config": {
            "convs": conv_indices,
            "models": models,
            "backend": args.backend,
            "max_turns_per_conv": args.max_turns_per_conv,
            "limit_questions": args.limit_questions,
            "recall_mode": "hybrid",
            "colbert_enabled": True,
            "colbert_weight": 1.5,
            "k": args.k,
        },
        "dataset": {
            "name": "BEAM-10M",
            "paper": "https://arxiv.org/abs/2510.27246",
            "repo": "https://github.com/mohammadtavakoli78/BEAM",
            "chats_dir": str(CHATS_DIR),
        },
        "totals": {m: len(all_results[m]) for m in models},
        "generations_by_model": all_results,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"\n[beam] wrote generations -> {path}", flush=True)
    print(f"[beam] entries per model: {payload['totals']}", flush=True)
    print(f"[beam] next: judge phase via beam_10m_judge.py {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
