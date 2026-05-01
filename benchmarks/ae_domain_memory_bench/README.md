# AE-domain memory bench

240-query benchmark calibrated to Angels Electric Service workflows.
Per Sprint 2 Phase 7 execution addendum lines 580-637.

## Categories

| Category | Queries | R@5 threshold | Why it matters |
|---|---|---|---|
| `electrical_contracting` | 40 | 0.78 | NEC code, breaker brands, conduit specs, panel work |
| `spanish_whatsapp` | 40 | 0.70 | Crew comms; tests multilingual + sparse-on-jargon |
| `materials_sku` | 40 | 0.75 | SKU resolution, brand entities, BOM lookups |
| `lennar_lots` | 40 | 0.80 | Lot-keyed entity retrieval, builder customer flow |
| `financial_calendar` | 40 | 0.72 | Invoice/job-cost/QBO context |
| `customer_temporal` | 40 | 0.82 | Bi-temporal "before X" / "valid as of" queries |

Global target: R@5 >= 0.760 across all 240. Per addendum line 633.

## Running

```bash
# Diagnostic mode — no ground-truth labels needed; reports what
# each query surfaces from the live store.
python3 run_ae_domain_bench.py --db ~/.neural_memory/memory.db

# Run only one category
python3 run_ae_domain_bench.py --category lennar_lots

# Scored mode — requires ground_truth_ids filled in queries.py
python3 run_ae_domain_bench.py --mode scored --out report.json
```

Diagnostic mode output is the input you use to **label** ground truth.
Once labels exist, switch to scored mode for pass/fail.

## Labeling workflow

1. Run diagnostic mode against the live DB.
2. For each query in the JSON output, inspect the `dense_top_k_ids` and
   `sparse_top_k_ids` lists.
3. Pick the memory IDs that are "right" for the question — the ground
   truth supporting memories.
4. Edit `queries.py` and fill `ground_truth_ids: [...]` on the query dict.
5. Re-run scored mode. Aim for the per-category R@5 thresholds.

## Exit codes

- `0` — diagnostic completed, OR scored mode passed all category thresholds
- `1` — bad CLI args / unknown category
- `2` — scored mode but at least one category missed threshold

Use exit code `2` to gate CI: a Phase 7 commit that drops a category
below threshold should fail the build.

## Composition with existing benches

This bench is **additive**. The pre-Phase-7 LongMemEval 500-record
baseline at `~/.neural_memory/bench-history/full-500-rerank-2026-04-25-214311.json`
(R@5 = 0.580) remains the regression detector for synthetic recall.
The AE-domain bench tests AE-specific workflows that the synthetic LME
bench doesn't cover.

Definition of done for Phase 7 (per addendum lines 567-578):
- LME 500-record R@5 doesn't regress more than 0.020 absolute
- AE-domain R@5 reaches the per-category thresholds above
- Combined coverage passes daily smoke gating
