## Verified

- `dataset_v2.py` delivered the leakage fix: my 5,200-sample scan showed overall avg/max/nonzero token overlap of `0.0`, and every topic template had zero non-anchor overlap.
- `_GLOBAL_ANCHORS` + lock landed; continuity-style target/noise generation produced `0` collisions across `6,250` anchors.
- `lstm_knn.py` now imports `Memory` from `neural_memory.py` and has the `3x` warmup pass.
- `conflict_quality.py` has the `detect_conflicts=False` control. Runtime smoke: winner@1 `0.3333` with supersession vs `0.0333` control, lift `+0.3000`; stale-above-winner drop `+0.5333`.
- `baseline.py` prewarms the embedder cache before timing.
- New suites compile and are wired into the runner.

## New blocks

- `graph_reasoning` does not prove graph reasoning. In my run, `think()` got `0.0`, multihop matched raw at `0.0667`, and shuffled edges did better at `0.1`. The suite relies on cosine `auto_connect`, not explicit A→B→C edges, so the docstring claim is not achieved.
- `dream_derived_fact` does not isolate derived facts. It scores whether top-k collectively contain both tokens, so pre-dream multihop already scored `0.92`; post-dream semantic regressed `0.08 → 0.04`, multihop stayed flat, and only `1` derived fact appeared.
- `continuity_controls` is useful, but it is still anchor retrieval. Raw cosine dominated: `1.0 / 0.98 / 0.9 / 0.7` vs neural-memory `0.76` at every tier in my run.
- `channel_ablation.py` has a fairness bug: its hard-coded “default” ablation weights do not match actual `NeuralMemory` defaults for BM25/entity/temporal, so every ablation changes more than one variable.
- `hnsw_exactness.py` may not measure exact-vs-HNSW cleanly: the “exact” arm leaves C++ retrieval enabled, and the HNSW arm does not assert that HNSW actually activated.

## Verdict

No. v3 materially improves honesty and adds the right kinds of controls, especially leakage removal, global anchor uniqueness, and conflict-control attribution. But a peer reviewer would not accept this as proof that neural-memory-adapter does something a vanilla vector store cannot: raw cosine still wins continuity, graph reasoning fails its own negative control, dream recall lift is not reliable, and two new ablation/exactness suites have confounded controls.