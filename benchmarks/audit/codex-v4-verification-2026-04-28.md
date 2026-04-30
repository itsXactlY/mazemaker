## Per-block verdict

1. **graph_reasoning: fixed.** Ingest uses `auto_connect=False` at lines 238-240, explicit `add_connection(A,B)` and `(B,C)` at lines 248-250, edge count assertion is lines 257-268, no A-C assertion is lines 270-280. Shuffle deletes by `source_id OR target_id` at lines 357-362, checks residual zero at lines 368-378, then adds `2*n` random non-real-chain edges at lines 390-409.

2. **dream_derived_fact: fixed.** Strict single-result metric is lines 132-146; strict `derived:*` hit metric is lines 149-161; `_measure(..., k_strict=3)` is lines 164-165 and constructor default is line 227. Distractors are injected before premises at lines 251-267.

3. **continuity_controls / dataset_v2: fixed.** The suite uses `generate_concept_continuity_pairs` at lines 179-184. Generator queries are selected from templates without `{anchor}` at dataset lines 402-406, 419-423, 436-440, 453-457, 470-474, 487-490 and assigned unformatted at lines 534-550. Distractors are concept templates with fresh unrelated anchors at continuity lines 216-238.

4. **channel_ablation: fixed.** Baseline is built with `channel_weights=None` at lines 90-95, then defaults are read from `baseline_mem._sqlite_memory._channel_weights` at lines 98-106. The source defaults match in `memory_client.py` lines 765-772.

5. **hnsw_exactness: fixed.** Both arms pass `use_cpp=False`, `rerank=False`, and boolean `use_hnsw` at lines 84-90; exact/HNSW arms call `False` and `True` at lines 212-214. Probe recall materializes lazy HNSW at lines 110-116; exact raises if `_hnsw_index` exists at lines 131-137; HNSW nonactivation is flagged and skipped at lines 118-129 and 231-240. `memory_client._ensure_hnsw` confirms `True` bypasses the `"auto"` sub-1000 cutoff at lines 1018-1031.

## Anything new broken

No source-level blocker. Two caveats: `graph_reasoning` wraps the shuffled-control assertions in a broad `except` and records `"error"` instead of failing the suite at lines 430-431; reviewers must treat that as an invalid run. `dream_derived_fact`’s `derived_fact_hit_rate` accepts any `derived:*` in top-k, not necessarily one containing this query’s anchor/tokens; pair it with `single_doc_both_tokens`.

## Final verdict

**qualified-y.** I would accept this as evidence if the actual reported results show graph lift plus shuffle collapse, strict post-dream lift with pre-dream zero, and HNSW/ablation controls behaving as claimed. Caveats: this is synthetic evidence, the graph benchmark proves traversal over explicitly supplied edges rather than automatic edge discovery, and the dream `derived:*` metric needs the single-doc token metric beside it to establish query-specific synthesis.