# Benchmarks

Every number Mazemaker publishes, the harness that produced it, and the
audit story behind the methodology. Read this if you're about to cite a
Mazemaker number externally, or if you're trying to reproduce one.

> **One-sentence summary.** Mazemaker ships two complementary benchmarks:
> the **Inception Bench** (deterministic, no LLM in the scoring loop) for
> headline retrieval claims, and the **internal audit suite** (eight rounds
> of GPT-5.5 review, full transcript committed) for capability lifts.

---

## Table of contents

1. [The three result tiers](#the-three-result-tiers)
2. [Headline numbers](#headline-numbers)
3. [Capability lifts](#capability-lifts)
4. [Public-benchmark numbers (LongMemEval-S)](#public-benchmark-numbers-longmemeval-s)
5. [Comparison Bench](#comparison-bench)
6. [The audit story (eight rounds)](#the-audit-story-eight-rounds)
7. [What the benchmark gave back to production](#what-the-benchmark-gave-back-to-production)
8. [Harness mismatch invariant](#harness-mismatch-invariant)
9. [Bench-noise discipline](#bench-noise-discipline)
10. [Reproducing the numbers](#reproducing-the-numbers)

---

## The three result tiers

| Tier                          | Harness                                          | Use for                       |
|-------------------------------|--------------------------------------------------|-------------------------------|
| **Inception Bench**           | `benchmarks/mazemaker_memory_bench.py`           | Headline capability claims    |
| **LongMemEval-oracle 500q**   | `benchmarks/mazemaker_godbench.py --variant oracle` | Full-corpus retrieval results |
| **LongMemEval-S 500q (v8 audit)** | `benchmarks/external/longmemeval_s.py`        | Per-question ephemeral; the public Wu et al. number |

> **Don't compare across tiers without reading
> [the harness mismatch invariant](#harness-mismatch-invariant) first.**
> The three measure different things by design.

---

## Headline numbers

### Inception Bench (deterministic, no LLM)

12 scenarios. Score is `label in [r["label"] for r in top_k]`.

| Scenario               | Corpus   | R@1   | R@5   | R@10  | p50 ms |
|------------------------|---------:|------:|------:|------:|-------:|
| S3 multi-fact          |     100  | 1.000 | 1.000 | 1.000 |   30   |
| S4 update-tracking     |      15  | 1.000 | 1.000 | 1.000 |   15   |
| S5 conflict-fuse       |      10  | 0.800 | 1.000 | 1.000 |   15   |
| S6 distractor-resist   |     105  | 1.000 | 1.000 | 1.000 |   27   |
| S7 needle 1 k          |   1,005  | 1.000 | 1.000 | 1.000 |  104   |
| S8 negation            |      55  | 1.000 | 1.000 | 1.000 |   24   |
| S10 latency 10 k       |  10,000  | 0.695 | 1.000 | 1.000 |  945   |
| **S11 needle 100 k**   | 100,010  | **1.000** | **1.000** | **1.000** | 9,618 |
| S12 dream-ablation     |   1,005  | 1.000 | 1.000 | 1.000 |   —    |
| **Macro mean**         |          | **0.888** | **0.924** | **0.972** |        |

**The big number.** 100,010 facts stored, 10 hand-crafted needles hidden
uniformly through the haystack. Query each by paraphrase. All 10 surface
at rank-1. Pure `nm.recall(query, k=10)`. No LLM.

See [`inception-bench.md`](inception-bench.md) for the methodology.

### LongMemEval-oracle 500q — full-corpus harness (iter97 champion)

500 questions, ~24 k–30 k memories per corpus state, single PG schema,
every question competes against the full corpus.

| Metric                            | iter00 anchor | iter97        | Δ                  |
|-----------------------------------|:-------------:|:-------------:|:------------------:|
| Aggregate R@5                     | 0.7298        | **0.8340**    | **+10.42 pp**      |
| R@10                              | 0.7383        | **0.9000**    | **+16.17 pp**      |
| MRR                               | 0.5777        | **0.7124**    | **+13.47 pp**      |
| single-session-preference R@5     | 0.2333        | **0.6667**    | **+43.34 pp (+186 %)** |
| single-session-user R@10          | 0.7344        | **1.0000**    | **+26.56 pp**      |
| temporal-reasoning R@5            | 0.6063        | **0.7323**    | **+12.60 pp**      |
| multi-session R@5                 | 0.8099        | **0.8595**    | **+4.96 pp**       |
| p50 latency                       | —             | 1,728 ms      |                    |
| p95 latency                       | —             | 3,261 ms      |                    |

iter100 took the aggregate further (R@5 = 0.8426). See
[`changelog-beta.md`](changelog-beta.md#the-threshold) for the full
threshold-crossing narrative.

### LongMemEval-S 500q — per-question ephemeral harness (v8 audit)

External benchmark from Wu et al. (ICLR 2025). Each question gets a
fresh engine with only that question's 50–200 haystack sessions ingested.
470 gradeable questions of 500.

| Metric      | hybrid baseline | hybrid + ColBERT@1.5 | Δ          |
|-------------|----------------:|---------------------:|:----------:|
| **R@1**     | 0.8064          | **0.8574**           | **+5.10 pp**|
| **R@5**     | 0.9596          | **0.9787**           | **+1.91 pp**|
| **R@10**    | 0.9830          | **0.9894**           | +0.64 pp   |
| **MRR**     | 0.8733          | **0.9114**           | **+3.81 pp**|
| p50 latency | 41.1 ms         | 56.9 ms              | +15.8 ms   |

ColBERT@1.5 lifts **three of six question types to perfect R@5**
(knowledge-update, multi-session, single-session-assistant) and gives
the largest single-category swing on **single-session-user (+7.8 pp R@5,
+10.4 pp MRR)**.

---

## Capability lifts

The internal suite proves the cognition layers are doing real work by
running each capability with the relevant mechanism *disabled* as a
control arm. If the control passes, the mechanism is decoration.

| Capability                                                          | Vanilla cosine          | Mazemaker         | Lift               |
|---------------------------------------------------------------------|-------------------------|-------------------|--------------------|
| Hop-2 graph reasoning (answer reachable only via A→B→C edges)      | **0.00** R@10            | **1.00** R@10     | **+1.00**          |
| Real edges vs shuffled control (proves traversal, not embedding)    | n/a                     | 1.00 → 0.27       | **+0.73 collapse** |
| Post-dream synthesis (facts inferable only after consolidation)     | structurally **0.00**   | **0.43** at scale | **+0.43 lift**     |
| Conflict supersession (winner@1 with `detect_conflicts=False`)      | 0.03 control            | **0.33**          | **+0.30**          |
| Cross-session continuity under concept-mode distractors             | **0.06**                | **0.62**          | **+0.56**          |
| Lean retrieval mode (real prose, n=200) vs default skynet           | n/a                     | **0.60** vs 0.42  | **+0.18 R@5**      |

Negative controls (shuffled edges, supersession off, pre-dream zero)
**must fail when the relevant mechanism is disabled** — and they do.
That's the evidence the algorithm is doing something a vector DB cannot.

---

## Public benchmark numbers (LongMemEval-S)

The v8 audit numbers above are reproducible end-to-end. The canonical
result JSONs are checked in.

```bash
# Run LongMemEval-S 500q with ColBERT enabled
python benchmarks/external/longmemeval_s.py \
  --enable-colbert --colbert-weight 1.5 \
  --questions 500 \
  --output benchmarks/external/results/lme_s_500q_colbert.json
```

Output JSON includes per-question-type breakdowns, per-question hits,
and full timings.

---

## Comparison Bench

Head-to-head against the ten small/medium open-source models that an
external memory-benchmark vendor publishes as scoring `0/N` because the
models couldn't follow the required JSON output schema. We score
plain-text answers via substring match. No JSON gating.

| Run                  | Aggregate            | Errors | Notes                            |
|----------------------|---------------------:|-------:|----------------------------------|
| no-ColBERT           | 186/200 = **93.0%**  | 2      | hybrid + rerank + advanced       |
| **ColBERT@1.5 (fixed)** | **188/200 = 94.0%** | **0**  | reproducibility-fix verified     |

`gemma3:270m` — Google's smallest production-deployed LLM (270 M
parameters, runs on a Raspberry Pi) — scores **18/20 = 90 %** in both
conditions.

```bash
# One-line reproduction
bash <(curl -fsSL https://mazemaker.dev/bench.sh)
```

Harness + canonical JSONs in [`benchmarks/external/`](../benchmarks/external/README.md).

---

## The audit story (eight rounds)

A peer-review-grade benchmark for this kind of system **didn't exist**.
Existing semantic-memory evaluations measure either retrieval (BEIR,
MS MARCO) or QA (NaturalQuestions) — none of them test graph
traversal, dream consolidation, or supersession.

So we built one and had it independently audited by **GPT-5.5** via the
codex CLI. It pushed back hard. Eight rounds:

| Round | Verdict                                | Headline reason                                                                  |
|-------|----------------------------------------|----------------------------------------------------------------------------------|
| v2    | **no**                                 | Lexical leakage in queries; broken dream suite; no baseline                      |
| v3    | **no**                                 | Topic-word leakage; cross-instance anchor collisions; wrong-class import         |
| v4    | qualified-y                            | Source-level fixes pending verification                                          |
| v5    | **YES + 4 caveats**                    | Every condition empirically satisfied                                            |
| v6    | qualified-y w/ 4 caveats               | Real-text mode + lean preset shipped; 4 follow-ups                               |
| v7    | qualified-y w/ 1 caveat                | n=200 real-prose: lean **beats** default skynet by +0.18 R@5                     |
| **v8** | **UNCONDITIONAL YES — no residual caveat** | Dream lift +0.43 at scale; the +0.04 at v7 was a sample-size artifact      |

Every prompt and every verdict, from "no, this is just lexical retrieval"
to "unconditional yes — accept it as evidence", is committed verbatim
under [`benchmarks/audit/`](../benchmarks/audit/). Open
`codex-v2-audit-2026-04-28.md` and `codex-v8-verdict-2026-04-28.md`
side by side to see the journey end-to-end.

---

## What the benchmark gave back to production

Running the benchmark wasn't just measurement — it surfaced real
engineering wins. Each one is now a documented, opt-in knob:

- **`retrieval_mode: lean`** — channel ablation proved BM25 / temporal /
  salience are dead-weight (or actively *harmful*) on real prose. Lean
  drops them. 4× faster than skynet on synthetic; +0.18 R@5 better than
  skynet on real prose.
- **`recall_score_percentile`** — replaces the legacy `score_floor`
  which operated on a badly-scaled internal score (~0..0.05). The new
  percentile knob is calibrated [0,1] by rank, so `0.5` keeps top half
  regardless of corpus or model.
- **PPR is the load-bearing channel for ranking** (-0.13 MRR if removed);
  semantic is the load-bearing channel for recall (-0.26 if removed).
- **`MM_COLBERT_ENABLED=1` is the precision-mode opt-in** — pre-computes
  a per-memory top-32 token cache (~64 KB/row, ~14.7 GB across a 230 k
  corpus). On LongMemEval-S 500q lifts R@1 +5.10 pp / MRR +3.81 pp.
  Default-off so existing latency budgets stay intact.

```bash
# Reproduce the full v8 audit suite
python -m benchmarks.mazemaker_benchmark.runner \
  --realistic --suite baseline --suite lean_skynet \
  --suite graph_reasoning --suite dream_derived_fact \
  --suite conflict_quality --suite continuity_controls \
  --suite channel_ablation \
  --output-dir benchmarks/results/my-run --seed 42

# Single-suite quick check (graph reasoning is the headline)
python -m benchmarks.mazemaker_benchmark.runner \
  --paraphrase --suite graph_reasoning
```

A full run takes ~12 min on a workstation.

---

## Harness mismatch invariant

`mazemaker_godbench.py` and `longmemeval_s.py` measure different things.
Do **not** compare numbers across them without this caveat.

| Harness                                    | Corpus shape                                         | R@5 typical |
|--------------------------------------------|------------------------------------------------------|-------------|
| `mazemaker_godbench.py --variant oracle`   | One ~25 k-memory haystack per question (full corpus) | 0.74–0.84   |
| `mazemaker_godbench.py --variant s`        | All 334,158 sessions in one schema                   | ~0.1        |
| `benchmarks/external/longmemeval_s.py`     | Per-question ephemeral 50–200 sessions               | 0.96+       |

- **godbench oracle** = production-shaped, brutal. The full corpus
  competes for every recall slot.
- **godbench --variant s** = "if Mazemaker were the operator's long-term
  memory, could it pick the right session out of EVERYTHING?"
- **longmemeval_s.py** = what the LongMemEval paper actually scores; what
  every published competitor is measured against.

If a result drops, look at the harness path FIRST. Don't go hunting for
engine bugs to explain a difference that's harness-by-design.

---

## Bench-noise discipline

> **Read before citing any number.**

- **godbench R@5 has ±0.5 pp run-to-run noise** at n=500. iter26 R@5 =
  0.7319 and iter39 (identical code path) R@5 = 0.7277 — within noise.
- **Per-question-type splits at n=30** have **3.3 pp per-question
  granularity** — a 1-question variance dominates anything below 6.7 pp.
- **Real wins need either** a delta clearly above the noise band, or
  multi-iteration replication. iter34/35/37/38 all hit ssp R@5 = 0.3667
  — that's a real signal because it appeared four times. A single 0.7319
  outlier shouldn't anchor "best" claims.
- **External judge attribution is mandatory.** The same 442 mm_10m_eval
  answers read **0.36 (nano), 0.49 (Opus), 0.53 (Haiku), or 0.64
  (gpt-5.4-mini)** depending on judge. If you see a memory-benchmark
  number without judge attribution, treat it as decoration.

This is why Inception Bench exists — see [`inception-bench.md`](inception-bench.md).

---

## Reproducing the numbers

### Get the 1.6 GB reproducibility bundle

```bash
# Download from ProtonDrive
# URL:    https://drive.proton.me/urls/J2T53B95XC#gtbM3E2mTvjt
# SHA256: 263e249408fa5b057dd8f356581cd5c14b3b5e62ba1b29e61704e54a156754c9

# Verify
sha256sum mazemaker-claims-2026-05-19.tar
# Should match the SHA256 above.

# Extract
tar xf mazemaker-claims-2026-05-19.tar
cd mazemaker-claims-2026-05-19/
```

Inside: pg_dump of the oracle corpus, every iter01–iter100 JSON, the
full 8-round audit transcripts, the bench runner, and a top-level
`README.md` with the exact CLI to reproduce the headline.

### Restore + reproduce iter97

```bash
# Create the bench database
createdb mm_bench_oracle
pg_restore -d mm_bench_oracle longmemeval_oracle.dump      # ~3 s

# Run the bench with the iter97 champion stack
export MM_DB_BACKEND=postgres
export MM_POSTGRES_DB=mm_bench_oracle
export MM_COLBERT_ENABLED=1
export MAZEMAKER_INTENT_BOOST=0.10
export MAZEMAKER_TEMPORAL_WEIGHT=0.7
export MAZEMAKER_SALIENCE_WEIGHT=0.5
export MAZEMAKER_PPR_WEIGHT=0.55

python benchmarks/mazemaker_godbench.py \
  --variant oracle \
  --questions 500 \
  --recall-mode skynet \
  --colbert-weight 2.5 \
  --dae-weight 2.0 \
  --pref-multi-recall \
  --rerank \
  --dream \
  --output benchmarks/external/results/oracle_reproduce.json
```

### Run the Inception Bench

```bash
git clone https://github.com/itsXactlY/mazemaker
cd mazemaker
pip install -r requirements.txt
python benchmarks/mazemaker_memory_bench.py
```

Takes ~15 min including the 100 k needle scenario. Output written to
`benchmarks/RESULTS.md`.

---

## Going deeper

- **Inception Bench methodology** — [`inception-bench.md`](inception-bench.md)
- **What the dream cycle contributes** — [`dream-engine.md`](dream-engine.md)
- **Production lessons + operator rules** — [`production-lessons.md`](production-lessons.md)
- **Beta changelog** — [`changelog-beta.md`](changelog-beta.md)
