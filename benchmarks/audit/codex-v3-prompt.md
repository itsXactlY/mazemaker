# Codex / GPT-5.5 verification pass (v3)

You audited this benchmark already (see `/tmp/codex-audit-result.md`). The v3 round acted on every concrete item you listed. Verify the fixes landed and that the new suites do what they claim, then give a short y/n verdict on whether the benchmark is now truly unique.

Repo: `/home/alca/projects/mazemaker-adapter`

## What the v3 round changed

1. `dataset_v2.py`
   * Topic-word leakage you flagged ("team", "incident", "latency", etc) — query templates rewritten. Verify by reading the file and re-running the per-topic overlap scan.
   * `_GLOBAL_ANCHORS` registry + lock added so cross-instance anchor uniqueness is now actually enforced. Verify continuity-suite generators no longer collide.

2. `suites/lstm_knn.py`
   * Now imports `Memory` from `neural_memory.py` (the wrapper that owns `_lstm_knn_ready`), not `Mazemaker` from `memory_client.py`. Adds a 3x warmup pass to seed the AccessLogger.

3. `suites/conflict_quality.py`
   * Adds a `detect_conflicts=False` control arm. Reports `supersession_lift` = winner_rate(with) − winner_rate(control).

4. `suites/baseline.py`
   * Pre-warms the shared embedder for both raw cosine and mazemaker before timing.

5. New v3 suites — read each and verify it does what its docstring claims:
   * `suites/graph_reasoning.py`: A→B→C chains, query about A, answer on C. Compares raw cosine, semantic, skynet, recall_multihop, think(). Negative control = shuffled chain edges.
   * `suites/channel_ablation.py`: zero-out one channel of skynet at a time.
   * `suites/hnsw_exactness.py`: HNSW vs exact at 1k/10k tiers, top-k overlap.
   * `suites/dream_derived_fact.py`: store split premises P1/P2 about X, query for the conjunction pre-dream and post-dream.
   * `suites/continuity_controls.py`: cross-session continuity with raw-cosine + recency baselines.

## Smoke-test findings (paraphrase mode, seed=42)

* **graph_reasoning** — only `think()` actually solves hop-2:
    - raw_cosine: R@10=0.0, MRR=0.0
    - nm_semantic: 0.07
    - nm_skynet: 0.03
    - nm_multihop: 0.03
    - nm_think: **0.43**
    - Shuffled-edges control (multihop): 0.23 — multihop with broken chain edges did *better* than with the real chain. Suggests auto_connect's cosine-threshold-based edges weren't actually following the explicit "see also Y" references in the chain text.

* **dream_derived_fact** — first proof of dream-quality lift:
    - pre-dream semantic both-tokens rate: 0.24 / multihop: 1.00
    - post-dream semantic: 0.48 (+24 pts) / multihop: 1.00
    - 6 derived:cluster memories materialised, 50 new connections in 0.12s

* **conflict_quality** with control:
    - control_no_supersession: winner@1=0.00, loser>winner=0.63
    - with_supersession: winner@1=0.17, loser>winner=0.20
    - supersession_lift: +0.17 winner@1, +0.43 loser_drop. Without supersession, the stale fact ALWAYS outranks the new one.

* **continuity_controls** — mazemaker vs raw cosine vs recency:
    - tier 0:    nm=0.46, raw=1.00, recency=0.10
    - tier 200:  nm=0.76, raw=1.00, recency=0.00
    - tier 1200: nm=0.82, raw=1.00, recency=0.00
    - tier 6200: failed (`unpack requires a buffer of 4 bytes` — real corruption bug at scale)
    - Raw cosine wins outright across every tier. nm-recall does grow with noise (an oddity worth investigating) but never catches raw cosine.

* **dataset leakage** — average Jaccard 0.024 → 0.001 (24×); nonzero-leakage queries 18-24% → 0.8-1% (25×). 0 cross-instance anchor collisions (was 8 / 5200).

## What I want from you

A short final assessment (max 500 words). Sections:

`## Verified` — which of the v2 audit fixes did v3 actually deliver, by what evidence.

`## New blocks` — anything still preventing this benchmark from being publishable as proof of the system's unique features. Include any *new* leakage / fairness issues introduced by the v3 changes.

`## Verdict` — y/n with one paragraph. The criterion: would a peer reviewer accept "this benchmark proves mazemaker-adapter does something a vanilla vector store can't"?

Read at minimum:
- `benchmarks/neural_memory_benchmark/dataset_v2.py`
- `benchmarks/neural_memory_benchmark/suites/graph_reasoning.py`
- `benchmarks/neural_memory_benchmark/suites/dream_derived_fact.py`
- `benchmarks/neural_memory_benchmark/suites/continuity_controls.py`
- `benchmarks/neural_memory_benchmark/suites/conflict_quality.py`
- `benchmarks/neural_memory_benchmark/suites/lstm_knn.py`

Spot-check anything else if a finding seems too good or too bad to be true.
