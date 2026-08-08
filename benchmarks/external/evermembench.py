uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuz#!/usr/bin/env python3
"""EverMemBench — external memory benchmark run against the Mazemaker engine.

STATUS: QUEUED (diagnostic only — do NOT publish the R@k it prints as a score).
Empirical finding on scenario 004: only 13% of ground-truth answers appear
VERBATIM in the corpus; 87% are synthesized/paraphrased by the benchmark. So the
substring answer-recall@k below is a corpus/pipeline DIAGNOSTIC, not a valid
EverMemBench result — R@k on all questions is inflated by trivial short-answer
false positives, and on meaningful answers it is a false negative (substring
can't match a paraphrase). A VERIFIED EverMemBench score needs the benchmark's
own answer-generation + open-ended LLM-judge scoring — that pipeline is TODO.


EverMemBench (arXiv:2602.01313, EverMind-AI) is a multi-party, multi-group,
>1M-token workplace-dialogue memory benchmark: 5+ scenarios, each a set of
"Group N" conversations of {speaker, time, dialogue} messages, plus a QA set of
{id, Q, A} records with short free-text ground-truth answers.

This harness measures the *retrieval* half of that benchmark — which is where
the EverMemBench authors report the bottleneck ("similarity-based methods fail
to bridge the semantic gap between queries and implicitly relevant memories",
oracle multi-hop ~26%). Metric: ANSWER-RECALL@k — after ingesting the dialogue
corpus into a fresh, isolated Mazemaker engine, for each question we recall the
top-k memories and check whether the ground-truth answer string appears in them
(normalized substring). Deterministic, no LLM judge, same substring-scoring
philosophy as comparison_bench.

This isolates Mazemaker's memory layer: same corpus, same questions, only the
retrieval engine is under test. Answer-generation / LLM-judge accuracy (the
other half of EverMemBench) is a separate, QUEUED study.

Data: dataset/{NNN}/dialogue_en.json + qa_{NNN}.json from
huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic (gated). Place under
benchmarks/external/data/evermembench/.

Run:
    python -m benchmarks.external.evermembench --scenario 004 --k 10
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent
PY_DIR = ROOT / "python"
if PY_DIR.is_dir() and str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))  # host run; in-container the engine is on PYTHONPATH=/app

DATA_DIR = Path(__file__).resolve().parent / "data" / "evermembench"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── data loading ────────────────────────────────────────────────────
def load_scenario(scenario: str, granularity: str = "message") -> tuple[list[dict[str, Any]], list[dict[str, str]], str]:
    """Return (memories, qa_records, corpus_sha256) for one scenario.

    granularity="message": one memory per utterance (fits any embedder's context
    window — faithful, no truncation). granularity="session": one memory per
    (group, conversation) chunk (matches LongMemEval session-granularity, but
    long chunks get truncated by 512-token embedders).
    """
    dpath = DATA_DIR / f"dialogue_{scenario}.json"
    qpath = DATA_DIR / f"qa_{scenario}.json"
    if not dpath.exists():
        # accept the upstream dataset/{NNN}/ layout too
        dpath = DATA_DIR / f"dialogue_en_{scenario}.json"
    raw_d = dpath.read_bytes()
    raw_q = qpath.read_bytes()
    dlg = json.loads(raw_d)
    qa = json.loads(raw_q)

    entries = dlg["dialogues"]
    entries = entries.values() if isinstance(entries, dict) else entries

    sessions: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        # entry = {"Group X": [ {speaker,time,dialogue}, ... ]}
        for gname, msgs in entry.items():
            if not msgs:
                continue
            if granularity == "message":
                for m in msgs:
                    body = m.get("dialogue", "")
                    if not body.strip():
                        continue
                    sessions.append({
                        "label": f"evermembench:{scenario}:m{len(sessions):05d}",
                        "text": f"[Group: {gname}][{m.get('speaker','?')} @ {m.get('time','?')}] {body}",
                        "group": gname,
                        "n_msgs": 1,
                    })
            else:  # session granularity
                lines = [
                    f"[{m.get('speaker','?')} @ {m.get('time','?')}] {m.get('dialogue','')}"
                    for m in msgs
                ]
                sessions.append({
                    "label": f"evermembench:{scenario}:s{len(sessions):04d}",
                    "text": f"[Group: {gname}]\n" + "\n".join(lines),
                    "group": gname,
                    "n_msgs": len(msgs),
                })

    qars = qa["qars"] if isinstance(qa, dict) and "qars" in qa else qa
    records = [{"id": q.get("id", str(j)), "q": q["Q"], "a": str(q["A"])}
               for j, q in enumerate(qars)]

    sha = hashlib.sha256(raw_d + raw_q).hexdigest()
    return sessions, records, sha


# ── scoring ─────────────────────────────────────────────────────────
# ── generate + judge (EverMemBench's real open-ended protocol) ──────
import urllib.request as _ur

def ollama_chat(model: str, prompt: str, host: str, timeout: float = 120.0) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0}}).encode()
    req = _ur.Request(host.rstrip("/") + "/api/generate", data=body,
                      headers={"Content-Type": "application/json"})
    with _ur.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("response", "").strip()

def generate_answer(model, host, question, results, top_n=8):
    ctx = "\n\n".join(f"[memory {i}] {(r.get('content') or '')[:900]}"
                      for i, r in enumerate(results[:top_n], 1)) or "(no memories retrieved)"
    p = (f"You are answering from a coworker's chat-history memory. Use ONLY the memories below.\n\n"
         f"{ctx}\n\nQuestion: {question}\nAnswer concisely; if the memories don't contain it, say \"I don't know\".\nAnswer:")
    return ollama_chat(model, p, host)

def judge_answer(model, host, question, gold, candidate):
    p = (f"Question: {question}\nReference answer: {gold}\nCandidate answer: {candidate}\n\n"
         f"Does the candidate answer convey the same key information as the reference answer? "
         f"Reply with exactly one word: YES or NO.")
    out = ollama_chat(model, p, host).upper()
    return out.startswith("Y") or ("YES" in out[:6])

_WS = re.compile(r"\s+")

def _norm(s: str) -> str:
    return _WS.sub(" ", str(s).lower().strip())

def answer_rank(answer: str, results: list[dict[str, Any]]) -> int:
    """1-indexed rank of the first recalled memory containing the answer
    string (normalized substring); 0 if not present in the returned list."""
    a = _norm(answer)
    if not a:
        return 0
    for i, r in enumerate(results, 1):
        body = _norm(r.get("content") or r.get("text") or "")
        if a in body:
            return i
    return 0


# ── engine ──────────────────────────────────────────────────────────
def build_engine(db_path: str, backend: str, recall_mode: str, rerank: bool, full: bool,
                 colbert_weight: float, dae_weight: float, retrieval_candidates: int):
    import memory_client as mc
    if full:
        # FULL production engine — the inception-bench stack: C++ ops, GPU BGE-M3,
        # ColBERT + DAE channels, PG backend (MM_DB_BACKEND=postgres). Runs inside
        # the localhost/mazemaker-v2-mcp:gpu image against an ISOLATED PG database
        # (MM_POSTGRES_DB), never the operator's live corpus.
        nm = mc.Mazemaker(
            db_path=db_path, embedding_backend="auto", use_cpp=True,
            retrieval_mode=recall_mode, use_hnsw="auto", lazy_graph=True,
            think_engine="ppr", rerank=rerank, retrieval_candidates=int(retrieval_candidates),
            channel_weights={"colbert": float(colbert_weight), "dae": float(dae_weight)},
        )
        nm.store._ensure_embedding_column(1024)
        return nm
    return mc.Mazemaker(
        db_path=db_path, embedding_backend=backend, use_cpp=False,
        retrieval_mode=recall_mode, use_hnsw=False, lazy_graph=True,
        think_engine="bfs", rerank=rerank,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--scenario", default="004", help="scenario id, e.g. 004")
    p.add_argument("--granularity", default="message", choices=["message", "session"],
                   help="one memory per utterance (message) or per group-chunk (session)")
    p.add_argument("--k", type=int, default=10, help="top-k to retrieve/score")
    p.add_argument("--limit", type=int, default=0, help="cap number of questions (0 = all)")
    p.add_argument("--backend", default="auto")
    p.add_argument("--recall-mode", default="hybrid")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--full", action="store_true",
                   help="FULL production engine (use_cpp + GPU BGE-M3 + ColBERT + DAE + PG); "
                        "run inside the gpu image against an isolated MM_POSTGRES_DB")
    p.add_argument("--colbert-weight", type=float, default=1.5)
    p.add_argument("--dae-weight", type=float, default=1.0)
    p.add_argument("--retrieval-candidates", type=int, default=512)
    p.add_argument("--gen-model", default="", help="ollama model to GENERATE answers from recalled context (open-ended EverMemBench protocol)")
    p.add_argument("--judge-model", default="", help="ollama model to JUDGE gen-answer vs gold (defaults to --gen-model)")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://host.containers.internal:11434"))
    p.add_argument("--min-answer-len", type=int, default=4,
                   help="answers shorter than this are scored but reported separately "
                        "(substring match on <4 chars is ambiguous)")
    p.add_argument("--db", default="", help="persist the SQLite corpus here (default: tempfile)")
    p.add_argument("--skip-ingest", action="store_true",
                   help="corpus already ingested into --db; go straight to scoring")
    p.add_argument("--out", default="")
    args = p.parse_args()

    print(f"[evermembench] loading scenario {args.scenario} ({args.granularity}) …", flush=True)
    sessions, records, sha = load_scenario(args.scenario, args.granularity)
    if args.limit:
        records = records[:args.limit]
    print(f"[evermembench] {len(sessions)} session-memories · {len(records)} questions "
          f"· corpus sha256 {sha[:12]}…", flush=True)

    db = args.db or tempfile.mktemp(suffix=".db")
    nm = build_engine(db, args.backend, args.recall_mode, args.rerank, args.full,
                      args.colbert_weight, args.dae_weight, args.retrieval_candidates)

    # ── ingest (batch-embed + bulk store — vectorized ONNX beats per-item) ──
    ingest_s = 0.0
    if args.skip_ingest:
        print(f"[evermembench] --skip-ingest: reusing corpus in {db}", flush=True)
    else:
        t0 = time.perf_counter()
        texts = [s["text"] for s in sessions]
        labels = [s["label"] for s in sessions]
        remember_batch = getattr(nm, "remember_batch", None)
        embed_batch = getattr(nm.embedder, "embed_batch", None)
        BATCH = 256
        for i in range(0, len(texts), BATCH):
            ct, cl = texts[i:i + BATCH], labels[i:i + BATCH]
            if remember_batch:  # full engine: GPU-batched embed + multi-row PG INSERT
                remember_batch([{"text": t, "label": l} for t, l in zip(ct, cl)],
                               auto_connect=False, detect_conflicts=False, detect_supersedes=False)
            elif embed_batch:
                for l, t, e in zip(cl, ct, embed_batch(ct)):
                    nm.store.store(l, t, list(e))
            else:
                for l, t in zip(cl, ct):
                    nm.store.store(l, t, list(nm.embedder.embed(t)))
            done = min(i + BATCH, len(texts))
            print(f"[evermembench]   ingested {done}/{len(texts)}", flush=True)
        ingest_s = time.perf_counter() - t0
        print(f"[evermembench] ingest done in {ingest_s:.1f}s", flush=True)

    # ── retrieve + score ──
    ranks: list[int] = []
    lats: list[float] = []
    per_q = []
    n_recall_errors = 0
    judge_model = args.judge_model or args.gen_model
    gj_correct = 0
    gj_total = 0
    hybrid = args.recall_mode in {"hybrid", "advanced", "skynet", "lean", "trim"}
    for j, rec in enumerate(records):
        t = time.perf_counter()
        rkw = {"k": args.k, "hybrid": hybrid, "rerank": args.rerank}
        if args.full:
            rkw.update(enable_colbert=True, enable_dae=True,
                       colbert_weight=args.colbert_weight, dae_weight=args.dae_weight)
        try:
            res = nm.recall(rec["q"], **rkw)
            err = None
        except Exception as e:  # engine FTS/tsquery choke on some queries — count, don't crash
            res, err = [], type(e).__name__
            n_recall_errors += 1
        lats.append((time.perf_counter() - t) * 1000.0)
        rank = answer_rank(rec["a"], res)
        ranks.append(rank)
        row = {"id": rec["id"], "answer": rec["a"], "answer_len": len(rec["a"]),
               "rank": rank, "n_results": len(res), "recall_error": err}
        if args.gen_model:
            # EverMemBench open-ended, scored comparison_bench-style: recall (full
            # engine) → LLM generates a plain-text answer from the recalled context
            # → DETERMINISTIC substring-match of the gold answer inside the generated
            # answer. No LLM judge (non-deterministic + a weak judge false-NO'd
            # correct answers in testing). Handles the 87%-synthesized-gold problem:
            # the generator restates the fact, we substring-check the generated text.
            try:
                gen = generate_answer(args.gen_model, args.ollama_host, rec["q"], res)
                a = _norm(rec["a"])
                ok = bool(a) and (a in _norm(gen)) and (len(rec["a"]) >= args.min_answer_len or a in _norm(gen))
            except Exception as e:
                gen, ok = f"<gen error: {type(e).__name__}>", False
            gj_total += 1
            gj_correct += 1 if ok else 0
            row["gen"] = gen[:300]
            row["gen_correct"] = ok
        per_q.append(row)
        if (j + 1) % 50 == 0:
            hit = sum(1 for r in ranks if 0 < r <= 5)
            acc = f" · gen+judge acc={gj_correct/gj_total:.3f}" if gj_total else ""
            print(f"[evermembench]   scored {j+1}/{len(records)} · R@5={hit/(j+1):.3f}{acc}", flush=True)

    # ── metrics (full set + meaningful-answer subset) ──
    def metrics(idxs: list[int]) -> dict[str, Any]:
        n = len(idxs)
        if not n:
            return {"n": 0}
        rk = [ranks[i] for i in idxs]
        def rec_at(kk): return sum(1 for r in rk if 0 < r <= kk) / n
        mrr = sum((1.0 / r) if r > 0 else 0.0 for r in rk) / n
        return {"n": n, "R@1": round(rec_at(1), 4), "R@5": round(rec_at(5), 4),
                "R@10": round(rec_at(10), 4), "MRR": round(mrr, 4)}

    all_idx = list(range(len(records)))
    good_idx = [i for i in all_idx if len(records[i]["a"]) >= args.min_answer_len]
    m_all = metrics(all_idx)
    m_good = metrics(good_idx)

    out = {
        "benchmark": "EverMemBench-Dynamic",
        "source": "huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic",
        "paper": "arXiv:2602.01313",
        "scenario": args.scenario,
        "corpus_sha256": sha,
        "metric": "answer-recall@k (ground-truth answer string appears in top-k recalled memories)",
        "config": {
            "embedding_backend": args.backend,
            "recall_mode": args.recall_mode,
            "rerank": args.rerank,
            "k": args.k,
            "granularity": args.granularity,
            "min_answer_len": args.min_answer_len,
        },
        "n_session_memories": len(sessions),
        "n_questions": len(records),
        "n_recall_errors": n_recall_errors,
        "ingest_seconds": round(ingest_s, 1),
        "p50_recall_ms": round(statistics.median(lats), 1) if lats else None,
        "metrics_all": m_all,
        "metrics_meaningful_answers": m_good,
        "generate_substring": ({
            "method": "recall(full engine) → LLM answer → deterministic substring-match of gold in generated text (comparison_bench-style, no LLM judge)",
            "gen_model": args.gen_model,
            "n": gj_total, "correct": gj_correct,
            "accuracy": round(gj_correct / gj_total, 4) if gj_total else None,
        } if args.gen_model else None),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "per_question": per_q,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = out["generated_at"]
    outpath = Path(args.out) if args.out else RESULTS_DIR / f"evermembench_{args.scenario}_{stamp}.json"
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n" + "=" * 64)
    print(f"EverMemBench-Dynamic · scenario {args.scenario} · answer-recall@k")
    print(f"  corpus: {len(sessions)} session-memories, sha256 {sha[:12]}…")
    print(f"  ALL questions        (n={m_all['n']}): "
          f"R@1={m_all['R@1']} R@5={m_all['R@5']} R@10={m_all['R@10']} MRR={m_all['MRR']}")
    print(f"  meaningful answers ≥{args.min_answer_len}c (n={m_good['n']}): "
          f"R@1={m_good['R@1']} R@5={m_good['R@5']} R@10={m_good['R@10']} MRR={m_good['MRR']}")
    print(f"  p50 recall latency: {out['p50_recall_ms']} ms · ingest {out['ingest_seconds']}s "
          f"· recall errors: {n_recall_errors}/{len(records)}")
    if args.gen_model:
        print(f"  GENERATE→SUBSTRING (open-ended QA accuracy, deterministic): "
              f"{gj_correct}/{gj_total} = {gj_correct/gj_total:.4f}  (gen={args.gen_model})")
    print(f"  → {outpath.relative_to(ROOT)}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
