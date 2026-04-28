## Caveats

1. **Synthetic data only: closed.** `RealTextGenerator` uses actual `.md`/`.py` chunks; verified 200 generated memories, 35 source files, avg leakage 0.0714. This is no longer template-only.

2. **Latency is real: partial.** Synthetic lean fixes the worst case: skynet p50 394ms vs lean 96ms, 4.12x faster. Realistic run shows no material p50 win: 56.39ms vs 56.08ms, because the measured corpus is small.

3. **Weak channels remain: partial.** Salience remains harmful on real text: removing it improves R@5 0.64 -> 0.66. But real text reverses the synthetic story for bm25/temporal: `no_bm25` costs -0.02 recall and `no_temporal` costs -0.08, so they are not globally dead.

4. **`score_floor` mis-calibration: partial.** `score_percentile` works in `memory_client.NeuralMemory.recall`: smoke test confirmed 0.5 -> 5/10, 0.7 -> 3/10, 0.9 -> 1/10, while `score_floor=0.2` returns 0. Legacy `score_floor` remains sharp-edged, and the public `Memory.recall` facade does not expose `score_percentile`.

## New caveats v6 introduced

Real-vs-synthetic channel behavior now diverges materially. Lean drops bm25 and temporal based on synthetic ablation, but real ablation shows semantic, ppr, temporal, entity, and bm25 all contribute; only salience is consistently bad.

The real retrieval result artifacts record `n=50` and `n_memories=50` for baseline/lean/channel_ablation, despite the generator being capable of 200 chunks. Treat the real-text evidence as a 50-query slice, not a scale result.

Dream lift is real but weak on realistic data: derived fact hit rate rises 0.00 -> 0.04, far below synthetic +0.32.

## Final verdict

**qualified-yes-with-4-caveats.** The v6 benchmark is substantially stronger and still supports YES for mechanism: real text exists, graph reasoning survives, lean is a useful opt-in latency preset, supersession works, and `score_percentile` fixes the calibrated filtering path internally. Residual caveats: small real-text retrieval sample, corpus-dependent channel weights, weak realistic dream lift, and missing `score_percentile` exposure through `Memory.recall`.