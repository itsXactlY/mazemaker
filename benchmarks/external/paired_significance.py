#!/usr/bin/env python3
"""Paired significance for the two committed LongMemEval-S runs.

Why this file exists: the README quotes deltas between the baseline run and
the ColBERT run. An unpaired delta on 500 questions is not evidence, so the
README quotes a *paired* test instead. This script recomputes it from the two
result JSONs that are committed in benchmarks/external/results/, and asserts
that its own hit definition reproduces the harness' own metric before it
prints anything.

    python benchmarks/external/paired_significance.py

Exit code 1 if the recomputation disagrees with the harness metric — a silent
mismatch would make every number in the README ballast.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "results" / "longmemeval_s_master-baseline_20260509T214714Z.json"
COLBERT = HERE / "results" / "longmemeval_s_colbert-on-master_20260510T034308Z.json"

KS = (1, 5, 10)


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing result file: {path}")
    return json.loads(path.read_text())


def hit(q: dict, k: int) -> bool:
    """A hit is a gold session inside the top k. Same rule as the harness."""
    rank = q.get("rank_of_gold")
    return isinstance(rank, int) and rank <= k


def mcnemar_exact(flips_up: int, flips_down: int) -> float:
    """Two-sided exact McNemar on the discordant pairs."""
    n = flips_up + flips_down
    if n == 0:
        return 1.0
    smaller = min(flips_up, flips_down)
    tail = sum(comb(n, i) for i in range(smaller + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> int:
    base, col = load(BASE), load(COLBERT)

    if base["git_sha"] != col["git_sha"]:
        raise SystemExit(f"different git shas: {base['git_sha']} vs {col['git_sha']}")
    if base["dataset"]["sha256"] != col["dataset"]["sha256"]:
        raise SystemExit("different dataset sha256 — not the same 500 questions")
    print(f"dataset sha256 : {base['dataset']['sha256'][:16]}…  (identical in both runs)")
    print(f"git sha        : {base['git_sha'][:12]}  (identical in both runs)")
    diffs = {
        k: (base["system_config"].get(k), col["system_config"].get(k))
        for k in set(base["system_config"]) | set(col["system_config"])
        if base["system_config"].get(k) != col["system_config"].get(k)
    }
    print(f"config delta   : {diffs}")
    print()

    a = {q["qid"]: q for q in base["per_question"] if not q.get("is_abstention")}
    b = {q["qid"]: q for q in col["per_question"] if not q.get("is_abstention")}
    paired = sorted(set(a) & set(b))
    print(f"paired questions: {len(paired)} "
          f"({base['metrics']['n_total']} total, "
          f"{base['metrics']['n_total'] - len(paired)} abstention-flagged or unmatched)")

    # Prove our hit() is the harness' metric before quoting a single delta.
    for run, per_q, metrics in (("baseline", a, base["metrics"]),
                                ("colbert", b, col["metrics"])):
        for k in KS:
            ours = sum(1 for q in per_q.values() if hit(q, k)) / len(per_q)
            theirs = metrics[f"recall@{k}"]
            # the harness stores 4 decimals, so compare at that resolution
            if round(ours, 4) != round(theirs, 4):
                raise SystemExit(
                    f"{run} R@{k}: recomputed {ours:.6f} != harness {theirs:.6f}"
                )
    print("hit() definition reproduces the harness metric on both runs — ok")
    print()

    hdr = f"{'metric':6} {'miss->hit':>10} {'hit->miss':>10} {'net':>5} {'p (exact)':>12}"
    print(hdr)
    print("-" * len(hdr))
    for k in KS:
        up = sum(1 for q in paired if not hit(a[q], k) and hit(b[q], k))
        down = sum(1 for q in paired if hit(a[q], k) and not hit(b[q], k))
        p = mcnemar_exact(up, down)
        verdict = "" if p < 0.05 else "   <- not significant"
        print(f"R@{k:<4} {up:>10} {down:>10} {up - down:>+5} {p:>12.3g}{verdict}")

    print()
    for label, metrics in (("baseline", base["metrics"]), ("colbert", col["metrics"])):
        print(f"{label:9s} R@1 {metrics['recall@1']:.4f}  R@5 {metrics['recall@5']:.4f}  "
              f"R@10 {metrics['recall@10']:.4f}  MRR {metrics['MRR']:.4f}  "
              f"p50 {metrics['p50_ms']:.1f} ms  p95 {metrics['p95_ms']:.1f} ms")
    print()
    print("worst category, both runs (R@1):")
    for t, m in sorted(base["metrics_by_question_type"].items(),
                       key=lambda kv: kv[1]["recall@1"]):
        c = col["metrics_by_question_type"].get(t, {})
        print(f"  {t:28s} n={m['n_gradeable']:3d}  {m['recall@1']:.4f} -> "
              f"{c.get('recall@1', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
