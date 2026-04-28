# Codex / GPT-5.5 final verdict (v4)

You audited this benchmark twice (v2 and v3). Your v3 verdict was **No** with 5 specific blocks. All 5 were addressed in commit `8deebe1` by 5 parallel Opus sub-agents (one block each). This is the third and final review.

Repo: `/home/alca/projects/neural-memory-adapter`

## Your v3 blocks and what changed

1. **`graph_reasoning` failed its own negative control** (think()=0.0, shuffled edges scored higher than the real chain). Fix: ingest with `auto_connect=False`, then explicitly `nm.store.add_connection(A→B)` and `(B→C)` per chain — exactly `2*n` edges, no A↔C shortcut. Assertions added: edge count == `2*n_chains`, A↔C row count == 0 per chain. Shuffle control now deletes via `source_id IN chain OR target_id IN chain` (canonicalisation can put ids on either side), asserts residual==0, then adds `2*n` random pairings refusing any pair coinciding with a real chain edge.

2. **`dream_derived_fact` scoring saturated pre-dream** (collective top-k metric pegged at 0.92). Fix: kept legacy metric flagged `legacy_collective_metric_inflated=True`. Added two strict metrics — `single_doc_both_tokens_rate` (one memory carries BOTH tokens; pre-dream MUST be 0 by template construction since each premise has only one token) and `derived_fact_hit_rate` (any `derived:*` label in top-k; pre-dream MUST be 0 since none exist). Tightened to `k_strict=3`. Injected 300 distractor paraphrase memories.

3. **`continuity_controls` lost to raw cosine across all tiers because queries shared the unique anchor token with the target**. Fix: new generator `generate_concept_continuity_pairs` produces queries using *concept paraphrases* that NEVER mention the anchor. Per noise tier > 0, suite injects 1-2 *near-distractors* per target — noise memories with the concept vocabulary from the query but a fresh unrelated anchor. Raw cosine now MUST drop with noise by design.

4. **`channel_ablation` had wrong default weights**. Fix: resolves defaults from the live NeuralMemory (`baseline_mem._sqlite_memory._channel_weights`). Real defaults are `{semantic:1.0, bm25:0.9, entity:1.0, temporal:0.35, ppr:0.55, salience:0.25}` — codex caught that bm25/entity/temporal were wrong. Added missing-key assertion + extra-key warning.

5. **`hnsw_exactness` didn't isolate HNSW vs exact properly**. Fix: both arms now use `use_cpp=False, rerank=False`. HNSW arm uses `use_hnsw=True` (NOT "auto") so it can't fall back at sub-1000-memory tiers. After a probe recall (to materialise the lazy `_ensure_hnsw` path), asserts `nm._hnsw_index is None` for exact and `is not None` for HNSW. Sub-threshold tiers now record `hnsw_did_not_activate=True` instead of bogus overlap==1.0. Each arm prints its `retrieval_path`.

## Your task

Read each fix and verify it landed properly. Specifically:

- Read `benchmarks/neural_memory_benchmark/suites/graph_reasoning.py` — confirm explicit `add_connection` calls happened, the A↔C-absent assertion is there, and the shuffle control's deletion + residual-check is correct.
- Read `benchmarks/neural_memory_benchmark/suites/dream_derived_fact.py` — confirm the two strict metrics are well-defined, distractors are injected before premises, and `k_strict` is 3.
- Read `benchmarks/neural_memory_benchmark/suites/continuity_controls.py` and the new `generate_concept_continuity_pairs` in `dataset_v2.py` — confirm queries don't mention anchors and distractors share concept vocabulary with queries.
- Read `benchmarks/neural_memory_benchmark/suites/channel_ablation.py` — confirm defaults are read from the live instance, not hardcoded.
- Read `benchmarks/neural_memory_benchmark/suites/hnsw_exactness.py` — confirm `use_cpp=False`, `rerank=False`, `use_hnsw=True/False` (not "auto"), and the activation-assertion logic.

Spot-check `python/memory_client.py` if any fix relies on internal state you want to verify.

## Output

A single markdown report under 500 words. Sections:

`## Per-block verdict` — y/n on each of the 5 fixes (mark fixed / partial / not-fixed) with the specific line(s) of evidence.

`## Anything new broken` — has the fix introduced a new confound or regression that wasn't there before? Be specific.

`## Final verdict` — would a peer reviewer now accept this benchmark as proof that neural-memory-adapter does something a vanilla vector store cannot? **y / n / qualified-y** (qualified-y = yes if accompanied by named caveats; list the caveats). One paragraph max.

Constraints: read the actual code, not summaries. If a fix only LOOKS correct in source but a smoke run would reveal a bug (e.g. a raised exception path), flag it. Do not run the benchmark suite (RAM-tight host) — source-level verdict is enough.
