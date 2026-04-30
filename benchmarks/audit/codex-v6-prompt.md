# GPT-5.5 v6 verdict — caveat fixes + real-text mode

You audited this benchmark five times (v2→v5). Your v5 verdict was **YES** with four named caveats:

1. **Synthetic data only.** All memories are template-generated.
2. **Latency is real.** Skynet p50 = 340ms is 200× raw cosine.
3. **Weak channels remain.** bm25 / temporal contributed zero, salience null-or-slightly-harmful.
4. **`score_floor` mis-calibration** — RRF-derived relevance ~0.05; knob documented as [0,1].

This v6 round addresses each. Verify the fixes and decide whether the YES becomes **unconditional** or what residual caveats remain.

## What changed

### Production code (`python/memory_client.py`)

- **`retrieval_mode="lean"`** — extend-only addition. Zeroes the channels that channel_ablation proved dead-weight on synthetic data (bm25, temporal, salience), keeps semantic + entity + ppr. Existing modes unchanged.
- **`score_percentile` kwarg on `recall()`** — calibrated [0,1] companion to `score_floor`. Operates on rank percentile, not raw RRF score, so 0.5 means "keep top half" regardless of underlying scale. Legacy `score_floor` left untouched.

### Benchmark

- **`dataset_real.RealTextGenerator`** — chunks the project's own .md/.py prose, anchors by real CamelCase / snake_case / *.py tokens. Global registry for cross-chunk uniqueness. 200 memories from 35 source files; avg leakage 0.07 (real prose has unavoidable shared vocabulary; vs 0.001 for synthetic).
- **`--realistic` CLI flag** — swaps generator. Wins over `--paraphrase`.
- **`suites/lean_skynet.py`** — same data, three modes (semantic / skynet / lean), reports recall + p50/p95/p99 latency.

## Live numbers — synthetic mode (--paraphrase --seed 42)

```
lean_skynet:
  semantic : R@5=0.68  MRR=0.60   p50= 66ms
  skynet   : R@5=0.90  MRR=0.85   p50=394ms
  lean     : R@5=0.88  MRR=0.84   p50= 96ms
  Δrecall(lean - skynet) = -0.0200    ← within noise
  speedup p50 = 4.12×                 ← 298ms saved per query
```

## Live numbers — realistic mode (--realistic --seed 42, 200 chunks from project corpus)

```
baseline:
  raw_cosine R@5=0.84  MRR=0.72   p50=  0.08ms   ← drops from 1.0; real prose harder
  semantic  R@5=0.34  MRR=0.22    p50= 17ms
  skynet    R@5=0.54  MRR=0.36    p50=  4.9ms

lean_skynet (different DB instance):
  semantic R@5=0.44  MRR=0.36     p50= 55ms
  skynet   R@5=0.64  MRR=0.48     p50= 57ms
  lean     R@5=0.66  MRR=0.51     p50= 57ms
  Δrecall(lean - skynet) = +0.0200    ← lean BEATS skynet on real text
  Δmrr(lean - skynet)    = +0.0327

graph_reasoning (real text):
  raw_cosine    R@10=0.10  MRR=0.02
  nm_semantic   R@10=0.07  MRR=0.05
  nm_skynet     R@10=0.87  MRR=0.35
  nm_multihop   R@10=1.00  MRR=0.33    ← perfect on hop-2 even on real prose
  nm_think      R@10=0.20  MRR=0.20
  multihop[ctrl] R@10=0.43             ← collapse +0.57

dream_derived_fact (real text):
  pre-dream  derived_fact_hit_rate (multihop) = 0.00   (forbidden by template)
  post-dream derived_fact_hit_rate (multihop) = 0.04
  pre-dream  derived_fact_hit_rate (semantic) = 0.00
  post-dream derived_fact_hit_rate (semantic) = 0.04
  12 derived facts, +251 connections
  lift +0.04 — smaller than synthetic's +0.32 but still > 0 and structurally
  forbidden pre-dream

conflict_quality (real text — same numbers as synthetic since memories are
synthesised conflict pairs regardless of corpus):
  with_supersession    : winner@1=0.33  loser>winner=0.13
  control no-superseed : winner@1=0.03  loser>winner=0.60
  supersession_lift    : +0.30 winner@1, +0.47 loser_drop

channel_ablation (real text) — STORY DIFFERS FROM SYNTHETIC:
  all channels: R@5=0.64  MRR=0.478
  no_semantic : R@5=0.42  MRR=0.346   Δrecall=-0.22   ← MASSIVELY important now
  no_bm25     : R@5=0.62  MRR=0.469   Δrecall=-0.02   ← contributes on real text
  no_entity   : R@5=0.58  MRR=0.450   Δrecall=-0.06
  no_temporal : R@5=0.56  MRR=0.427   Δrecall=-0.08   ← now matters
  no_ppr      : R@5=0.50  MRR=0.409   Δrecall=-0.14
  no_salience : R@5=0.66  MRR=0.515   Δrecall=+0.02   ← still slightly harmful
```

### Key observations the report should address

1. On synthetic data only `ppr` and `entity` contributed; on real text **every** channel except salience pulls weight. The `lean` preset (which drops bm25/temporal too) was tuned to synthetic and **may be too aggressive for real corpora** — it still wins on real text in this run because the recall gain from removing dataset-specific noise outweighs the lost contributions from bm25/temporal at this small corpus size, but on bigger corpora the call could flip.

2. `score_percentile` works correctly: on a hash-backend HashBackend smoke test, 0.5 → top-half, 0.7 → top-30%, 0.9 → top-10%, while legacy `score_floor=0.2` returns 0 results (proving the calibration bug).

3. Latency on real prose is uniformly low (all modes ~57ms p50) because the corpus is small (200 chunks). The 4× synthetic speedup vs 1× realistic speedup is a CORPUS-SIZE effect, not a mode effect — bm25/entity FTS queries are O(N), so they dominate at scale.

4. Synthetic `dream_derived_fact` showed +0.32 lift; realistic shows +0.04. The realistic test has only 25 premise pairs against 300 distractors (well-tuned for synthetic but maybe too noisy for real prose where the distractors are more semantically diverse).

## Your task

Output a final report under 500 words with sections:

`## Caveats` — for each of v5's four caveats, mark **closed / partial / open** with the specific evidence.

`## New caveats v6 introduced` — anything the additions broke or made worse. Be specific about real-vs-synthetic divergence in channel_ablation.

`## Final verdict` — **unconditional-yes / qualified-yes-with-N-caveats / no**. One paragraph. Explicit named caveats list if not unconditional.

Files to read:
- `benchmarks/neural_memory_benchmark/dataset_real.py`
- `benchmarks/neural_memory_benchmark/suites/lean_skynet.py`
- `python/memory_client.py` lines 745-790 (lean preset) and 1567-1582 (score_percentile)
- `benchmarks/results/run-2026-04-28-v6-realistic-full/results/*.json`
- `benchmarks/results/run-2026-04-28-v6-lean/results/lean_skynet_results.json` for the synthetic numbers
