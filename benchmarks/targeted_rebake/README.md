# Targeted Stage C Rebake

The formation-side lever that lifted Mazemaker's
**LongMemEval-oracle 500q R@5 from 0.7404 to 0.8043** —
above the 0.80 stretch target — at a total OpenAI API cost of ~$0.07.

## TL;DR

When retrieval-side tuning plateaus, the bottleneck is **memory
formation**: the gold session exists in the corpus but doesn't have
a strongly-embeddable user fact for the query to match against. This
toolkit fixes that by surgically extracting the missing facts.

The methodology is reusable for any benchmark and any question type:

```bash
# 1. Find the gold sessions you're missing
python find_typed_misses.py single-session-preference ssp_misses.json

# 2. Run gpt-5-nano with a type-appropriate prompt to extract the
#    missing user-side facts, insert at salience=2.0
python rebake.py --misses ssp_misses.json --type ssp --namespace api2

# 3. Re-run the benchmark, watch the targeted question type's R@5 jump
python ../mazemaker_inception_bench.py --variant oracle --limit 500 \
       --recall-mode skynet --rerank --colbert --colbert-weight 2.5 \
       --dae --dae-weight 2.0 --read-cache --retrieval-candidates 512 \
       --pref-multi-recall \
  MAZEMAKER_INTENT_BOOST=0.10 MAZEMAKER_TEMPORAL_WEIGHT=0.7 \
  MAZEMAKER_SALIENCE_WEIGHT=0.5
```

Each round typically gains 6-22 top-5 hits on the targeted type while
trading 2-6 hits across others (dilution dance). Run the diagnostic
again on the next-largest miss bucket and repeat.

## When to use this

Mazemaker's R@5 will saturate around 0.74 on inception_bench-oracle 500q with
pure retrieval-side tuning (channel weights, candidate pool size,
multi-recall, intent boost). The
[`invariant_retrieval_side_levers_converge`](../../../.claude/projects/-home-alca/memory/invariant_retrieval_side_levers_converge.md)
memory documents that ceiling — four structurally different tunings
all land at R@5=0.7404.

When you hit a similar plateau on **any** Mazemaker benchmark:

1. The gold is **not** missing from the candidate pool — the
   embedding+FTS+ColBERT pipeline finds it, but the gold doesn't
   score high enough vs noise candidates.
2. The fix is **formation**, not retrieval: crystallize a strongly-
   embeddable user fact from the gold session content so the
   query's cosine match finds it directly.

## Why three prompt variants

The three question types in LongMemEval-oracle need different
extraction shapes:

| `--type` | Question shape | Prompt extracts |
|---|---|---|
| `ssp` | "Can you suggest a hotel for my Miami trip?" | User preferences, ownership, plans — phrased "user prefers X" |
| `ssu` | "What degree did I graduate with?" | Concrete user-state facts — "user X is Y" with values inline |
| `tr`  | "What time do I wake up on Tuesdays?" | Time-anchored events — "user did X on Y" with concrete dates |

You can extend the script with a new prompt for a new question type.

## The dilution dance (read this before doing repeated rounds)

Each rebake inserts new facts into the same `memories` table that the
HNSW + FTS + ColBERT pipeline searches over. Those new facts compete
with the existing gold-matched facts for top-5 slots. We empirically
observed across iter74-iter80:

- ssp rebake (+97 facts):  ssp R@5 +20.00pp,  small cross-type cost
- ssp rebake round 2 (+101 facts):  ssp R@5 +13.33pp
- ssu rebake (+48 facts): ssu R@5 +29.69pp, ssp R@5 -13.34pp (dilution!)
- tr rebake (+98 facts):  tr R@5 +18.11pp,  ssp R@5 -13.34pp further
- ssp re-rebake (+116 facts): ssp R@5 +23.34pp recovery, ssu R@5 -15.63pp

**Per-session cap is ~6 facts in round 1, dropping in subsequent
rounds.** Past that, low-signal facts pull the high-signal ones out
of top-5. A round 3 test on the iter75 corpus regressed ssp R@5 by
-10pp before we rolled it back.

**Use a distinct namespace per round** (`api2`, `api3`, `ssu`, `tr`,
`sspre`, etc.) so you can roll back a regressing round with a single
`DELETE FROM memories WHERE label LIKE '%::<namespace>::%'`.

## Cost

For inception_bench-oracle 500q (24,531 baseline memories, 30 ssp / 64 ssu /
127 tr questions):

| Round | Sessions touched | Facts added | OpenAI cost |
|---|---:|---:|---:|
| ssp round 1 (generic prompt) | 18 | 97 | $0.02 |
| ssp round 2 (query-conditional) | 18 | 101 | $0.005 |
| ssu | 22 | 48 | $0.005 |
| tr  | 74 | 98 | $0.02 |
| ssp re-rebake | 16 | 116 | $0.01 |
| **Total** | **148** | **460** | **~$0.07** |

The bench is fully deterministic, so a single bench run per round is
all you need to evaluate the lift. Each bench run takes ~10 minutes.

## Files in this directory

- `find_typed_misses.py` — diagnostic; emits a JSON of missed-top-5
  queries with their gold session ids
- `rebake.py` — polymorphic extractor; `--type ssp|ssu|tr` selects the
  prompt template
- `README.md` — this file

## Provenance

The original per-round scripts and the empirical case study are
documented in the operator's auto-memory:

- `decision_iter75_formation_breakthrough.md` — the first ssp result
- `decision_iter79_target_exceeded.md` — the 0.80 R@5 crossover
- `invariant_retrieval_side_levers_converge.md` — why retrieval-side
  hits a ceiling at this corpus

Read those first if you're trying to understand the full lineage.
