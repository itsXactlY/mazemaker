# Inception Bench

A deterministic memory benchmark with no LLM in the scoring loop. The
ground truth is a canonical gold string; the score is `label in [r["label"]
for r in top_k]` plus substring + unit-aware match against gold. That's it.

> **Why this exists.** Every external memory benchmark we measured against
> had a meaningful rubric-defect rate, and every LLM-judged scoring
> pipeline we tried drifted by 16 percentage points on identical answers
> depending on which judge we picked. The only honest path forward was
> to ship our own scorer alongside our own engine. We did.

> **Two files, one methodology.**
> *Inception Benchmarking* is the **discipline** of building your own
> deterministic scorer instead of inheriting an LLM-judged rubric. Two
> harnesses embody it:
>
> - **`benchmarks/mazemaker_memory_bench.py`** — the **pure-memory
>   bench** described in this document. 12 deterministic scenarios. No
>   LLM at any step. The headline 100 k needle @1 = 1.000 comes from here.
> - **`benchmarks/mazemaker_inception_bench.py`** — the **full-corpus
>   LongMemEval harness** (`--variant oracle` / `--variant s`). The
>   100-iteration loop ran on this. See
>   [`benchmarks.md`](benchmarks.md#headline-numbers) for those numbers
>   and [`changelog-beta.md`](changelog-beta.md#the-threshold) for the
>   threshold-crossing narrative.

---

## Table of contents

1. [The one-paragraph summary](#the-one-paragraph-summary)
2. [Why external rubrics were broken](#why-external-rubrics-were-broken)
3. [The judge calibration spread](#the-judge-calibration-spread)
4. [Inception Benchmarking — the methodology](#inception-benchmarking--the-methodology)
5. [The 12 scenarios](#the-12-scenarios)
6. [Headline numbers](#headline-numbers)
7. [What the bench deliberately does NOT do](#what-the-bench-deliberately-does-not-do)
8. [Running it yourself](#running-it-yourself)
9. [Failure modes we publish](#failure-modes-we-publish)

---

## The one-paragraph summary

Published memory benchmarks tend to be measurement instruments built
around LLM-judged rubrics. When we audited them at scale we found that
**roughly half of the update-tracking items on Inception Bench conv-3 were
defective by construction** (the gold value had been superseded in the
same corpus by a later seq, but the rubric pointed at the older one)
and that **the same 442 Inception Bench answers scored anywhere from 0.36
to 0.64 depending on which LLM we asked to judge them**. We could not
build engineering loops on top of that — every "regression" might just
be a judge drift; every "improvement" might just be rubric noise. So we
shipped `benchmarks/mazemaker_memory_bench.py`: 12 deterministic
scenarios, substring-match scoring, bit-for-bit reproducible. We call
the discipline **Inception Benchmarking** — building your own scorer
inside your own engine, alongside your own claims, because the
out-of-the-box rubric ecosystem isn't reliable enough to ship on.

---

## Why external rubrics were broken

### BEAM-10M conv-1 — 4/20 defective rubrics

Cross-confirmed in a blind audit (2026-05-11, Sonnet + Qwen3.6-Plus,
κ = 0.71 inter-judge agreement):

- **Phantom rubric keyword.** Gold mentions "Tuesday" but no source
  turn says "Tuesday" — the qgen LLM hallucinated the keyword.
- **Question-corpus misquote.** Question quotes a sentence that
  doesn't exist verbatim in the conversation.
- **Empty gold.** Rubric requires a value the conversation never
  provides.
- **Compositional hallucination.** Gold is a synthesis of two
  separate facts that the question doesn't tell the engine to
  synthesize.

### Hindsight `int()` truncation — 9/10 ability evaluators bugged

Every Hindsight ability evaluator does:

```python
score = int(partial_credit_float)   # 0.5 partial → 0 score
```

So 0.5 partial credits — the explicit middle-grade in the rubric —
silently drop to zero. Hindsight's published BEAM-10M conv-1 = 64.1%
is **artificially deflated** because of this. Cross-confirmed
2026-05-11, no developer response.

### Inception Bench update_tracking — ~50% rubric defects on conv-3

Forensic audit of 6 conv-3 UT questions (2026-05-14):

| qid  | rubric gold              | gold@seq | seq-latest                  | seq-latest seq | verdict             |
|------|--------------------------|----------|-----------------------------|----------------|---------------------|
| UT_0 | 22 agents                | 9974     | **25 agents**               | 10074          | RUBRIC SUPERSEDED   |
| UT_1 | under 2%                 | 9975     | **7% failure**              | 13874          | RUBRIC SUPERSEDED   |
| UT_2 | 1.5%                     | 15930    | 99.9% (different metric)    | 20156          | rubric valid        |
| UT_3 | 88%                      | 17534    | **85% rollback**            | 17962          | RUBRIC SUPERSEDED   |
| UT_4 | zero unauthorized        | —        | "zero" nowhere in topic     | —              | RUBRIC ORPHAN       |
| UT_5 | 35%                      | 16762    | 98% (different metric)      | 16878          | rubric valid        |

**Theoretical ceiling on conv-3 UT (n = 6): ~0.33.** Only UT_2 and UT_5
are valid items; the rest are unwinnable by *any* engine, no matter
how good its retrieval is.

Corpus-wide audit (`audit/results/ut_audit_10conv.jsonl`, n = 65 UT
items across 10 convs):

- **OK: 6/65 (9.2 %)**
- **SUPERSEDED: 28/65 (43.1 %)**
- **ORPHAN: 31/65 (47.7 %)** — heuristic keyword extractor over-narrow;
  manually verifying reduces this to ~20–25 % real orphans.

Conservative real defect estimate: **30–50 %** of UT-rubric items are
unwinnable.

> **Every conv-3 UT optimization run** in the v6 → v11 prompt-ladder was
> chasing rubric defects no engine can satisfy. "UT = 0.000" on v11
> isn't engine failure — it's the rubric-defect rate stacked with
> budget/recall failures.

---

## The judge calibration spread

Same 442 Inception Bench answers, three judges, 58 rubric items
(v11 conv-3, 2026-05-14):

| Judge              | Tier            | overall | IE (n=42) | UT (n=12) | ABST (n=4) |
|--------------------|-----------------|--------:|----------:|----------:|-----------:|
| claude-haiku-4-5   | mid Anthropic   | 0.526   | 0.607     | 0.083     | 1.000      |
| claude-opus-4-7    | top tier        | 0.491   | 0.571     | 0.042     | 1.000      |
| gpt-5-nano         | operational     | 0.362   | 0.405     | 0.000     | 1.000      |
| gpt-oss-120b       | free OR         | ~0.28   | (extrapolated)            |            |
| gpt-5.4-mini       | quality         | ~0.64   | (extrapolated)            |            |

**Strict → lax ranking:** gpt-oss-120b < gpt-5-nano < Opus < Haiku < gpt-5.4-mini.

**Pairwise inter-judge agreement** (rubric-item exact match):

- nano vs haiku: 0.828 (10 disagree / 58)
- nano vs opus:  0.862 ( 8 disagree)
- haiku vs opus: **0.931** ( 4 disagree) — near-calibrated
- all-3 agree:   0.810 (11 split-decisions)

**Publishing a single Inception Bench number without judge attribution is
meaningless.** The same engine can read 0.36 or 0.53 depending on the
judge. Haiku/Opus form a natural calibration pair; nano is the
cheap-but-skewed operational judge.

---

## Inception Benchmarking — the methodology

After watching the scoreboard rot in real time, we wrote
`benchmarks/mazemaker_memory_bench.py`. Properties:

- **One file.** ~600 lines of Python. Read it; fork it; trust it.
- **12 deterministic scenarios.** Every test fact and query is in the
  source code, seed-deterministic.
- **No LLM in the scoring loop.** Score is
  `label in [r["label"] for r in top_k]`. Substring + unit-aware
  match against canonical gold. Deterministic. Reproducible bit-for-bit.
- **No external dataset.** Carries no rubric authoring noise, no
  judge-prompt tax, no published-leaderboard skew.
- **No synthetic LLM-generated questions.** The gold is hand-written.

The right design — borrowed from how Hindsight, BEAM, and classic IR
evals work *should have worked* :

- **Runner internal reasoning:** unchanged. Phase-1 enumeration,
  candidate selection, multi-step logic stays.
- **Runner final emission:** ONE short answer string. The value.
- **Question artefact:** `gold` is a single canonical value string,
  with an optional set of `accepted` paraphrases.
- **Judge:** deterministic. `gold_norm in answer_norm`, where
  `norm = lowercase + strip whitespace + collapse units`. No LLM
  involved. Score = 1.0 if match, 0.0 otherwise. Optional partial
  credit on unit-suffix mismatch (e.g. "2%" vs "2 percent").

**Properties recovered:**

- No judge calibration spread — `in` operator is the same on every machine.
- No reasoning_effort budget panic — bare value fits in 32 tokens.
- Per-ability prompt isolation — change UT contract, IE/ABST literally
  cannot be affected.
- Rubric defect rate becomes auditable in N seconds instead of by
  forensic recall-probing.

---

## The 12 scenarios

| #   | Scenario               | Corpus  | What it tests                                                 |
|-----|------------------------|--------:|---------------------------------------------------------------|
| S1  | template collision     |    100  | Disambiguation at template-collision density                  |
| S2  | template collision v2  |    100  | Variant of S1                                                 |
| S3  | multi-fact             |    100  | Multi-fact retrieval in one query                             |
| S4  | update-tracking        |     15  | Surface the latest value, not the earlier one                 |
| S5  | conflict-fuse          |     10  | Two contradicting memories → one resolved answer              |
| S6  | distractor-resist      |    105  | Hold ground when 100 distractors share surface form           |
| S7  | needle 1 k             |  1,005  | One needle in a 1 k haystack, paraphrased query               |
| S8  | negation               |     55  | "User does NOT like X" — surface the negation, not the topic  |
| S9  | graph-traversal        |  varies | Reach chain terminus from chain head                          |
| S10 | latency at 10 k        | 10,000  | p50 latency at 10 k corpus                                    |
| S11 | needle 100 k           | 100,010 | Ten needles hidden uniformly in a 100 k haystack              |
| S12 | dream ablation         |  1,005  | Control arm — must FAIL when dream cycles are disabled        |

---

## Headline numbers

```text
S3 multi-fact          R@1=1.000  R@5=1.000  R@10=1.000  p50= 30 ms
S4 update-tracking     R@1=1.000  R@5=1.000  R@10=1.000  p50= 15 ms
S5 conflict-fuse       R@1=0.800  R@5=1.000  R@10=1.000  p50= 15 ms
S6 distractor-resist   R@1=1.000  R@5=1.000  R@10=1.000  p50= 27 ms
S7 needle 1k           R@1=1.000  R@5=1.000  R@10=1.000  p50=104 ms
S8 negation            R@1=1.000  R@5=1.000  R@10=1.000  p50= 24 ms
S10 latency 10k        R@1=0.695  R@5=1.000  R@10=1.000  p50=945 ms
S11 needle 100k        R@1=1.000  R@5=1.000  R@10=1.000  p50= 9.6 s
S12 dream-ablation     R@1=1.000  R@5=1.000  R@10=1.000
─────────────────────────────────────────────────────────────────
Macro mean             R@1=0.888  R@5=0.924  R@10=0.972
```

**The big number is S11.** 100,010 facts stored. 10 hand-crafted
needles hidden uniformly through the haystack. Query each by
paraphrase. All 10 surface at rank-1. Pure `nm.recall(query, k=10)`. No
LLM in the loop.

### Ingest performance

`remember_batch()` does 100 k facts in 272 s (≈366 facts/s). Per-row
`remember()` would take ~7 hours at the quality-engine setting.

---

## What the bench deliberately does NOT do

- **No LLM-as-judge.** Score is purely substring match. There is no
  prompt to drift. There is no model to swap.
- **No external dataset** (BEAM-10M, MemoryAgentBench, etc.). Those
  carry rubric authoring noise and judge-prompt tax that we documented
  elsewhere in this directory.
- **No synthetic LLM-generated questions.** Every test fact and query
  is in the source code, seed-deterministic.
- **No hidden test set.** Everything is on GitHub, public, forkable.
- **No multi-claim rubric items.** One gold per question + N accepted
  paraphrases. If you can't reduce a capability to a substring, the
  benchmark says so explicitly.

---

## Running it yourself

```bash
git clone https://github.com/itsXactlY/mazemaker
cd mazemaker
pip install -r requirements.txt
python benchmarks/mazemaker_memory_bench.py
```

Takes ~15 min including the 100 k needle scenario. Output written to
`benchmarks/RESULTS.md` and stdout.

### Quick smoke

```bash
# Drop S10 and S11 for a faster pass
python benchmarks/mazemaker_memory_bench.py --skip S10 --skip S11
```

Runs in ~30 s.

---

## Failure modes we publish

We're explicit about what doesn't work yet. Hiding failures would defeat
the whole point.

- **S9 graph-traversal R@1 = 0.000.** `nm.think(depth=3)` does not
  surface the chain terminus from the chain head. This is a real
  engine TODO, not a scorer artefact. We say so.
- **S1 / S2 R@1 low (0.11 / 0.09).** 100 templated facts share surface
  forms; exact-content disambiguation is hard at template-collision
  density. R@10 hits 1.0 on S1, proving the engine surfaces all the
  candidates within top-10.
- **S10 latency at 10 k.** R@1 = 0.695 — not perfect at scale. The
  candidate pool starts losing the needle as the corpus grows past 10 k
  unless ColBERT + DAE are enabled.

If you find a defect in *our* rubric, file an issue. We will fix it,
re-run, and republish — out in the open.

---

## Going deeper

- **Benchmark numbers across all tiers** — [`benchmarks.md`](benchmarks.md)
- **What we learned the hard way** — [`production-lessons.md`](production-lessons.md)
- **The audit story (v2 → v8)** — [`benchmarks/audit/`](../benchmarks/audit/) in the repo
- **Engine architecture** — [`architecture.md`](architecture.md)
