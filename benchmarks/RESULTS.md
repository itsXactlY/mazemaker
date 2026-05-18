# Mazemaker Memory Benchmark — Results

> Pure memory benchmark. No LLM judge. No rubric ambiguity.
> One file, one command, fully reproducible.
> `python benchmarks/mazemaker_memory_bench.py`

**Date:** 2026-05-15 (audit-hardened)
**Engine:** Mazemaker semantic memory (BGE-M3 1024d + HNSW + ColBERT + DAE + cross-encoder rerank)
**Scorer:** deterministic label-match + word-boundary substring check on top-K recall output
**Seed:** 42 (fully deterministic)
**Hardening applied:** explicit S5 queries, word-boundary S4 match, hashable-identity assertion (UUID/ULID/str all OK), S12 fail-fast on dream-phase exceptions, micro-avg added alongside macro, no-results crash guard, S13 lexically-neutral query/values.

## S13 — THE MULTITURN MONUMENT

Single A→B→C question, single corpus, binary pass/fail.

**Setup** (1B-scalable design, ran at 500k):
- 500,000 distractor facts (~6M tokens) bulk-ingested via `remember_batch`
- 3 turns of the same attribute, inserted at quartile positions through the stream:
  - `multiturn_A` — *"Phoenix deployment tier set to **bronze**."*
  - `multiturn_B` — *"Phoenix deployment tier upgraded to **silver**."*
  - `multiturn_C` — *"Phoenix deployment tier finalized at **obsidian**."*
- Query: **`Phoenix deployment tier`** — contains no temporal hints ("current", "now", "latest", "final" all explicitly excluded)
- Values are semantic peers (tier names), not numerical magnitudes a similarity model can sort

**Why this design:** lexical overlap rescues any semantic engine if the query
shares words with one specific turn ("current"+"99.9%"). To test true temporal
memory, the three turns must look lexically symmetric — only their CHRONOLOGY
distinguishes them.

**Pass criterion (BOTH must hold):**
1. `nm.recall("Phoenix deployment tier", k=10)[0].label == "multiturn_C"`
2. `"obsidian" in nm.recall(...)[0].content`

Results will follow when the 500k run completes.

## Headline — 12 supporting scenarios (≤ 100k corpus)

| Scenario | Corpus size | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| S1 exact-recall                 |    100 | 0.110 | 0.870 | **1.000** |   52 |   56 |
| S2 paraphrase-recall            |    100 | 0.090 | 0.400 | 0.690 |   53 |   67 |
| S3 multi-fact                   |    100 | **1.000** | **1.000** | **1.000** |   30 |   30 |
| S4 update-tracking (chronology) |     15 | **1.000** | **1.000** | **1.000** |   15 |   17 |
| S5 conflict-fuse (supersession) |     10 | 0.800 | **1.000** | **1.000** |   15 |   16 |
| S6 distractor-resist            |    105 | **1.000** | **1.000** | **1.000** |   27 |   30 |
| S7 needle-haystack 1k           |  1,005 | **1.000** | **1.000** | **1.000** |  104 |  111 |
| S8 negation                     |     55 | **1.000** | **1.000** | **1.000** |   24 |   28 |
| S9 graph-traversal              |     54 | 0.000 | 0.000 | 0.000 |   — |   — |
| **S10 latency @ 10k corpus**    | **10,000** | 0.695 | **1.000** | **1.000** | **945** | **1,079** |
| **S11 needle @ 100k corpus**    | **100,010** | **1.000** | **1.000** | **1.000** | **9,618** | **13,581** |
| S12 dream-ablation (ABI test)   |  1,005 | **1.000** | **1.000** | **1.000** |   — |   — |
| **S13 multiturn @ 500k**        |  500,003 | *pending* | | | | |

## Reproducing

```bash
cd ~/projects/mazemaker/benchmarks
python mazemaker_memory_bench.py
# All 13 scenarios. Approximate wall time on RTX-class CUDA + 32G RAM:
#   S1-S9 :  5 min
#   S10   :  4 min  (10k corpus + 200 queries)
#   S11   : 11 min  (100k corpus + 10 needle queries)
#   S12   :  3 min  (5-phase dream cycle on 1k corpus)
#   S13   : 25 min  (500k corpus + 1 multiturn query)
# Per-scenario DBs land in /tmp/mazemaker_memory_bench/SN.db.
```

## Notes on the design choices

- **`remember()` identity contract:** asserts `mid is not None` and
  `hash(mid)` succeeds. Engine can return int, UUID, ULID, str, or
  any other hashable handle. Not locked to int.

- **S4 word-boundary match:** anchors on `"is now <value><unit>"` so
  `92` doesn't accidentally match `920`. Same protection on S5's
  expected-new-value check.

- **S12 is the only internal-ABI test.** Documented in the source.
  Don't cite for cross-engine comparisons.

- **Macro vs micro mean:** macro weights each scenario at 1. Micro
  weights by `n_queries` per scenario. S13 contributes 1 query;
  S1 contributes 100; macro is the engine's "breadth" score, micro
  is the "every question" score. Both are printed.

- **No silent fallback paths.** `build_quality_engine` signature
  changes raise an assertion. Dream-phase method renames raise an
  assertion. `recall()` returning 0 results on S13 raises an
  assertion. If the engine API drifts, the bench fails loudly —
  numbers can't be silently invalidated.

## What this is NOT

- Not a NLP benchmark. No LLM in the loop at any stage.
- Not a judge-calibration exercise. Score is `label in top-K`.
- Not a marketing aggregate. Macro/micro are reported separately so
  no scenario can hide behind a roll-up.

## What this IS

The definitive memory-engine measurement for Mazemaker. Every number
is mechanically reproducible from the source by running one command.
The engine stores stuff. The engine recalls stuff. The bench verifies
stuff came back. Nothing else.
