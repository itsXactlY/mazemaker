# Codex / GPT-5.5 audit task

Repo: `/home/alca/projects/neural-memory-adapter` (branch `V3.1`)
Subject of audit: `benchmarks/neural_memory_benchmark/`

This is a benchmark suite for a *semantic memory* system (neural-memory-adapter, a Hermes Agent plugin). The system claims to differ from a generic vector DB via:

- knowledge graph + spreading activation (BFS / PPR `think()`)
- HNSW ANN + multi-channel retrieval (semantic / hybrid / advanced / skynet)
- MMR diversification + score_floor cutoff
- LSTM + kNN re-ranker on the recall hot path
- "Dream Engine" autonomous consolidation (NREM strengthen, REM bridges, Insight community detection / Louvain + derived facts)
- Conflict detection + supersession (`[SUPERSEDED]` markers)
- Cross-session continuity for an agent harness

The benchmark was just rewritten to add five v2 suites alongside the existing nine:

- `suites/baseline.py` — raw cosine baseline (numpy) vs neural-memory `semantic` and `skynet`. Same embedder both sides so the comparison is retrieval-only.
- `suites/diversity.py` — MMR × score_floor sweep on a paraphrase ground-truth set.
- `suites/lstm_knn.py` — on/off ablation of the C++ LSTM+kNN re-ranker.
- `suites/continuity.py` — store target facts in "session 1", add up to 5000 distractor noise memories, query in "session N". Reports recall drop curve.
- `suites/conflict_quality.py` — store original + replacement, measure rank-1 winner rate (real supersession quality, not just "did the marker appear?").
- `suites/dream.py` — *rewritten*. Previous version called `strengthen_connection(mid, mid)` (self-loops), had a `pass` stub for Insight, and read non-existent dict keys. v2 calls `DreamEngine._phase_*` directly and measures pre/post graph deltas + recall@k delta.

Ground-truth dataset is `dataset_v2.py` — `ParaphraseGenerator` produces (statement, query) pairs where:
- The anchor entity is a coined nonsense token (e.g. `zriev38`) that cannot collide with embedding-model pretraining or with other memories.
- The query and statement share **only** the anchor; the answer tokens appear only in the statement.
- Measured average lexical leakage (Jaccard, anchor stripped) is ≈ 0.05.

## Your task

Audit the benchmark for whether it truly showcases the system's unique features without pre-polluting answers. Specifically:

1. **Honesty of ground truth.** Is the paraphrase generator actually disjoint-vocab in practice, or are there hidden lexical leaks (templates that share verbs, answer pools that overlap with query templates, etc.)? Read `dataset_v2.py` and check.

2. **Suite coverage of the unique features.** For each claimed differentiator (graph / dream / MMR / LSTM+kNN / conflict / continuity), is there a suite that actually exercises it end-to-end and produces a metric that would shift if the feature were removed? Identify gaps.

3. **Comparison fairness.** The baseline is raw cosine using the SAME embedder. Is that a fair-enough fight, or does anything tip the scales toward neural-memory beyond the retrieval algorithm itself? Look for: incidental advantages from the C++ side, hidden caching that benefits one side, asymmetric warm-up.

4. **Smoke-test results so far.** Sample run output:
   - baseline: raw R@5=1.0 / MRR=1.0 vs semantic R@5=0.46 / MRR=0.28 vs skynet R@5=0.86 / MRR=0.70
   - diversity sweep: mmr=0 → R@5=0.46 entropy=2.13; mmr=0.7 → R@5=0.36 entropy=3.37 (real trade-off); but score_floor ≥ 0.2 drops everything (the relevance score is RRF-derived ~0.05, so 0.2 is an absurd cutoff — this is a calibration bug worth flagging)
   Are these results telling a coherent story or is something off?

5. **Suggested additions.** What other tests would a peer reviewer of "is this benchmark truly unique" demand to see? Be concrete; don't suggest "consider adding X" — say "add `suites/Y.py` that does Z because the current pipeline doesn't measure W."

## Constraints

- **Read the actual code**, don't summarise from filenames.
- Don't propose tests that re-measure things already measured.
- A good audit is not flattering. If the benchmark is shallower than it looks, say so.
- Output: a single markdown report under 800 words. Sections: `## Honest', `## Gaps`, `## Concrete additions`, `## Verdict (truly unique? y/n + why)`.

Files to read first (in this order):
- `benchmarks/neural_memory_benchmark/dataset_v2.py`
- `benchmarks/neural_memory_benchmark/suites/baseline.py`
- `benchmarks/neural_memory_benchmark/suites/diversity.py`
- `benchmarks/neural_memory_benchmark/suites/dream.py`
- `benchmarks/neural_memory_benchmark/suites/continuity.py`
- `benchmarks/neural_memory_benchmark/suites/conflict_quality.py`
- `benchmarks/neural_memory_benchmark/suites/lstm_knn.py`
- `benchmarks/neural_memory_benchmark/benchmark.py`
- spot-check `python/memory_client.py` near the `recall()` and `_mmr_rerank()` methods if you need to verify the relevance-score scale claim.
