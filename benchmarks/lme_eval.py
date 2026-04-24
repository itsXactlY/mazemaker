#!/usr/bin/env python3
"""LongMemEval-style benchmark harness for Neural Memory.

Default mode is a deterministic synthetic smoke corpus so CI can run without
network/data downloads. Pass --dataset path/to.jsonl for real LongMemEval-like
records.

Accepted dataset formats per row/object:
  {"question": "...", "answer": "needle string", "context": "document text"}
  {"query": "...", "gold": "needle string", "memory": "document text"}
  {"query": "...", "answer": "needle string", "memories": ["doc1", ...]}

Metrics: R@1/R@5/R@10, MRR, p50/p95 latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY_DIR = ROOT / "python"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from memory_client import NeuralMemory  # noqa: E402


def synthetic_records() -> list[dict[str, Any]]:
    facts = [
        ("Ada stores the launch key under the blue ceramic owl.", "Where is Ada's launch key?", "blue ceramic owl"),
        ("Bruno's backup server is called pinecone-seven.", "What is Bruno's backup server called?", "pinecone-seven"),
        ("The migration window for Atlas is 03:40 UTC on Sunday.", "When is the Atlas migration window?", "03:40 UTC"),
        ("Mira prefers FastEmbed over sentence-transformers for cold start speed.", "What does Mira prefer for cold start speed?", "FastEmbed"),
        ("The red notebook says project Zephyr uses port 7443.", "Which port does project Zephyr use?", "7443"),
        ("Kai's dog Lou reacts badly to chicken treats.", "Which treats are bad for Lou?", "chicken treats"),
        ("The demo API key was rotated after incident ORCHID-19.", "What incident caused the demo API key rotation?", "ORCHID-19"),
        ("The telemetry dashboard lives behind the cloudflared tunnel.", "Where does the telemetry dashboard live?", "cloudflared tunnel"),
        ("Session summaries should be stored, but raw turn dumps should stay opt-in.", "What should stay opt-in?", "raw turn dumps"),
        ("The dream engine insight phase now uses Louvain community detection.", "What does the dream engine insight phase use?", "Louvain community detection"),
        ("PULSE findings need content-hash dedup before neural ingestion.", "What dedup is needed before neural ingestion?", "content-hash dedup"),
        ("The C++ bridge is optional; Python fallback must remain production-safe.", "What must remain production-safe if C++ is absent?", "Python fallback"),
        ("HNSW should activate automatically only when the corpus is large enough.", "When should HNSW activate automatically?", "corpus is large enough"),
        ("The weekly rollup should write a WEEKLY.md brief after Insight.", "What should the weekly rollup write?", "WEEKLY.md"),
        ("The primary memory database is ~/.neural_memory/memory.db.", "What is the primary memory database?", "~/.neural_memory/memory.db"),
    ]
    return [{"context": c, "query": q, "answer": a} for c, q, a in facts]


def load_dataset(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return synthetic_records()
    p = Path(path)
    raw = p.read_text()
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        obj = json.loads(raw)
        rows = obj if isinstance(obj, list) else obj.get("data") or obj.get("records") or []
    records = []
    for row in rows:
        query = row.get("query") or row.get("question") or row.get("input")
        answer = row.get("answer") or row.get("gold") or row.get("target") or row.get("needle")
        contexts = row.get("memories") or row.get("contexts") or row.get("documents")
        if contexts is None:
            ctx = row.get("context") or row.get("memory") or row.get("document") or row.get("text")
            contexts = [ctx] if ctx else []
        if query and answer and contexts:
            records.append({"query": query, "answer": answer, "contexts": [str(c) for c in contexts]})
    return records


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def answer_hit(results: list[dict[str, Any]], answer: str, k: int) -> bool:
    needle = (answer or "").lower()
    if not needle:
        return False
    for r in results[:k]:
        hay = f"{r.get('label','')}\n{r.get('content','')}".lower()
        if needle in hay:
            return True
    return False


def reciprocal_rank(results: list[dict[str, Any]], answer: str) -> float:
    needle = (answer or "").lower()
    for i, r in enumerate(results, start=1):
        hay = f"{r.get('label','')}\n{r.get('content','')}".lower()
        if needle and needle in hay:
            return 1.0 / i
    return 0.0


def run(records: list[dict[str, Any]], args) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nm-lme-") as td:
        db = str(Path(td) / "bench.db")
        nm = NeuralMemory(
            db_path=db,
            embedding_backend=args.backend,
            use_cpp=False,
            retrieval_mode="hybrid" if args.hybrid else "semantic",
            use_hnsw=args.use_hnsw,
            lazy_graph=args.lazy_graph,
            think_engine=args.engine,
            rerank=args.rerank,
        )
        try:
            for i, rec in enumerate(records):
                contexts = rec.get("contexts") or [rec.get("context")]
                for j, ctx in enumerate(contexts):
                    if ctx:
                        nm.remember(str(ctx), label=f"record:{i}:{j}", auto_connect=False, detect_conflicts=False)

            latencies = []
            r1 = r5 = r10 = 0
            rr = []
            for rec in records:
                start = time.perf_counter()
                results = nm.recall(str(rec["query"]), k=10, hybrid=args.hybrid, rerank=args.rerank)
                latencies.append((time.perf_counter() - start) * 1000.0)
                answer = str(rec["answer"])
                r1 += int(answer_hit(results, answer, 1))
                r5 += int(answer_hit(results, answer, 5))
                r10 += int(answer_hit(results, answer, 10))
                rr.append(reciprocal_rank(results, answer))
            n = max(1, len(records))
            return {
                "records": len(records),
                "backend": args.backend,
                "hybrid": args.hybrid,
                "rerank": args.rerank,
                "use_hnsw": args.use_hnsw,
                "lazy_graph": args.lazy_graph,
                "engine": args.engine,
                "R@1": round(r1 / n, 4),
                "R@5": round(r5 / n, 4),
                "R@10": round(r10 / n, 4),
                "MRR": round(statistics.mean(rr) if rr else 0.0, 4),
                "p50_ms": round(percentile(latencies, 0.50), 3),
                "p95_ms": round(percentile(latencies, 0.95), 3),
            }
        finally:
            nm.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Neural Memory on LongMemEval-like data")
    parser.add_argument("--dataset", help="JSON/JSONL dataset path. Omit for synthetic smoke corpus")
    parser.add_argument("--backend", default="hash", help="Embedding backend (hash for fast smoke, auto for real run)")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval channels")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranker")
    parser.add_argument("--use-hnsw", default="auto", help="HNSW mode: auto/true/false")
    parser.add_argument("--lazy-graph", action="store_true", help="Hydrate graph on demand")
    parser.add_argument("--engine", default="bfs", choices=["bfs", "ppr"], help="think/PPR engine")
    args = parser.parse_args()
    records = load_dataset(args.dataset)
    if not records:
        raise SystemExit("No benchmark records loaded")
    metrics = run(records, args)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
