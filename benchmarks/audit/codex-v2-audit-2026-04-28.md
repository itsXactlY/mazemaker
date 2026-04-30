## Honest

`dataset_v2.py` is mostly honest about answer leakage: I did not see answer-pool tokens reused in query templates. But “query and statement share only the anchor” is false. Hidden topic-token overlap remains: `team`, `incident`, `latency`, `production`, `backend`, and `maintenance` appear on both sides. Generated leakage is low but real: about `0.024-0.031` average, max `0.2`, nonzero in roughly `18-24%` of samples.

Bigger issue: anchor uniqueness is per `ParaphraseGenerator`, not global. `continuity.py` uses one generator for targets and a separate generator for noise; with default seed/noise tiers I found 8 target-anchor collisions after 5,200 noise facts. That can pollute continuity truth.

Also, the task is still largely “find the one document with this unique anchor.” Raw cosine getting `R@5=1.0 / MRR=1.0` proves the dataset is not a hard semantic-memory test. It avoids answer-token leakage, but it does not force graph reasoning, consolidation, supersession, or sequence learning.

## Gaps

Graph/spreading activation is shallow. `graph.py` counts visited/activated nodes and latency; it does not ask a query whose correct answer is only reachable through BFS/PPR. `dream.py` now calls real phases and measures graph deltas, but recall is still same-memory anchor recall, so derived bridges/facts are not proven useful.

MMR is genuinely exercised by `diversity.py`; the entropy/recall trade-off is real. The score floor is miscalibrated: `memory_client.recall()` uses RRF-ish relevance around `0.02-0.06`, so floors `0.2+` erase valid results.

LSTM+kNN is not exercised. `lstm_knn.py` imports `Mazemaker` from `memory_client.py`, but `_lstm_knn_ready` and `_enhance_recall()` live in `python/neural_memory.py`’s `Memory` wrapper. This suite will usually report “not loaded” rather than ablate the feature. Even if fixed, IID paraphrase queries do not create a sequence pattern for LSTM to learn.

Conflict quality is much better than marker-counting: it measures whether the replacement outranks the original. But it lacks a `detect_conflicts=False` control, so rank-1 winner rate cannot be attributed cleanly to supersession versus recency/semantic similarity.

Continuity measures old-fact retrieval under noise, but not an agent harness, not session-conditioned behavior, and not against raw cosine or recency baselines. The anchor-collision bug weakens it.

Baseline fairness is only partly fair. It uses the same embedding provider class, but raw cosine runs first, warming the shared model and writing embedding cache entries that mazemaker can reuse. That biases setup/latency toward mazemaker. Quality is less affected, but semantic mode can also use C++/HNSW paths unless explicitly disabled, so it is not a pure “same vectors, different ranking” control.

The smoke results tell a coherent but unfavorable story: raw cosine crushes semantic memory, skynet recovers some via multi-channel retrieval but still loses to a trivial baseline, MMR trades recall for diversity, and score floors are broken for the current score scale.

## Concrete additions

Add `suites/graph_reasoning.py`: build A→B→C chains and bridge communities where the query mentions A but the answer lives only on C. Compare `recall()`, `think(engine="bfs")`, `think(engine="ppr")`, and graph-disabled recall.

Add `suites/channel_ablation.py`: run semantic, BM25, entity, temporal, PPR, and skynet with one channel removed at a time. This is needed because current skynet results do not explain which channel helps or hurts.

Add `suites/hnsw_exactness.py`: at `1k/10k/50k` memories, compare `use_hnsw=False` exhaustive semantic search vs `use_hnsw=True` ANN on identical embeddings for recall loss and latency gain.

Add `suites/lstm_sequence.py`: use `python.neural_memory.Memory`, create repeated access/query motifs, warm the access log, then compare `_lstm_knn_ready=True` vs forced false on next-step recall.

Add `suites/dream_derived_fact.py`: seed clusters where REM/Insight should create bridges or `derived:*` facts, then query facts absent before dreaming and score whether new material improves answer retrieval.

Add `suites/continuity_controls.py`: reuse a global anchor registry, then compare mazemaker against raw cosine, recency-only, and shuffled-session controls.

## Verdict (truly unique? y/n + why)

No. The rewrite is more honest than the old benchmark, especially MMR, conflict quality, and real DreamEngine invocation. But the core ground truth is still anchor-key retrieval, raw cosine wins outright, LSTM+kNN is not actually tested, graph/dream metrics are mostly structural rather than task-quality metrics, and HNSW/multi-channel gains are not isolated. This benchmark can expose regressions and calibration bugs; it does not yet prove the system’s unique semantic-memory claims.