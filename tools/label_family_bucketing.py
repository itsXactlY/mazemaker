#!/usr/bin/env python3
"""label_family_bucketing.py — parse AE-domain bench corpus, bucket
UNLABELED queries by anchor-family, emit JSONL ready for batch labeling.

Per Tito 2026-05-02 directive (Item F + Multi-correct GT expansion from
save-state Future Work). Codex gpt-5.4 subagent identified the labeling
strategy in ~/.neural_memory/codex-subagent-runs/nm-builder/20260502-035227-*.md:
labeling FAMILIES is more efficient than 1-by-1 because:
- Anchor families (Sarah/contact, panel labels, GFCI, QBO reauth, permit
  paperwork, Spanish missing materials) recur across multiple queries
- Multi-correct GT means same memory IDs satisfy multiple related queries
- Family-level human review is ~10× faster than per-query review

This script does Step 1 of the methodology: parse + bucket. Steps 2-6
(run retrieval, seed candidates, emit per-family JSONL, human review,
write back GT) are downstream — see the codex finding in
~/.neural_memory/codex-subagent-runs/nm-builder/ for the full plan.

Output: ~/.neural_memory/label-families/<TIMESTAMP>-buckets.jsonl
Each line is one family with member queries + anchor terms + existing labels.

Usage:
    python3 tools/label_family_bucketing.py
    python3 tools/label_family_bucketing.py --print-summary  # human-readable

No substrate access required — pure static analysis of queries.py.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


REPO = Path(__file__).resolve().parent.parent
QUERIES_FILE = REPO / "benchmarks" / "ae_domain_memory_bench" / "queries.py"
OUT_DIR = Path.home() / ".neural_memory" / "label-families"


# Anchor families per codex finding 2026-05-02. Each family has:
#   - regex patterns (lowercase) that signal membership
#   - a short description
#   - "seed_ids" from existing labeled queries (multi-correct candidates
#     that likely apply to other family members too)
ANCHOR_FAMILIES: Dict[str, dict] = {
    "panel_labels": {
        "patterns": [r"panel.*label", r"label.*panel", r"panel.*schedule",
                     r"missing.*label", r"permit.*sticker"],
        "description": "Panel labeling, schedule notes, missing-label inspections",
        "seed_ids": [274, 286],  # from ELC-027, MAT-025, LOT-005
    },
    "sarah_contact": {
        "patterns": [r"sarah", r"before sarah", r"contact.*update",
                     r"contact.*before", r"main contact", r"customer.*contact"],
        "description": "Sarah / lot contact transitions, customer contact updates",
        "seed_ids": [264, 268, 280, 282],  # from LOT-006, LOT-023
    },
    "lennar_lot_12": {
        "patterns": [r"lennar.*lot 12", r"lot 12", r"lennar.*invoice",
                     r"lot.*invoice"],
        "description": "Lennar lot 12 + invoice chain",
        "seed_ids": [4914, 5179, 5180, 5531],  # from LOT-001, LOT-039
    },
    "spanish_missing_materials": {
        "patterns": [r"material.*no.*lleg", r"falt.*material",
                     r"materia.*falt", r"crew.*material",
                     r"comprar.*breaker", r"cable.*doce", r"cable.*numero"],
        "description": "Spanish WhatsApp: missing/needed materials, breakers, wire",
        "seed_ids": [6671, 6674, 6700, 7363, 7365, 7377],  # SPA-003, SPA-010, MAT-021, SPA-035
    },
    "qbo_reauth": {
        "patterns": [r"qbo.*reauth", r"qbo.*token", r"qbo.*refresh",
                     r"intuit.*reauth", r"quickbook.*reauth", r"qbo.*sync.*fail"],
        "description": "QBO token refresh / reauth events",
        "seed_ids": [],  # no existing labels yet — codex flagged FIN-005, FIN-033 candidates
    },
    "permit_paperwork": {
        "patterns": [r"permit.*paperwork", r"permit.*timing", r"permit.*office",
                     r"recoger.*permit", r"permit.*approv"],
        "description": "Permit paperwork, scheduling, approval",
        "seed_ids": [],  # codex flagged LOT-003, LOT-035, SPA-034 candidates
    },
    "gfci": {
        "patterns": [r"gfci"],
        "description": "GFCI receptacles + breakers (cross-language)",
        "seed_ids": [277, 288, 4666],  # ELC-001, MAT-006
    },
    "ev_charger": {
        "patterns": [r"ev charger", r"ev.*install", r"electric vehicle"],
        "description": "EV charger installation",
        "seed_ids": [158],
    },
    "bonding_grounding": {
        "patterns": [r"bonding.*bushing", r"grounding.*electrode",
                     r"grounding.*rod", r"neutral.*ground"],
        "description": "Bonding bushings + grounding work",
        "seed_ids": [5961],
    },
    "junction_box": {
        "patterns": [r"junction.*box"],
        "description": "Junction box sizing/placement",
        "seed_ids": [5947, 5966],
    },
}


def parse_queries(path: Path) -> List[dict]:
    """Parse queries.py via AST, return list of query dicts."""
    tree = ast.parse(path.read_text())
    queries: List[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "_q":
            continue
        # _q signature: (qid, category, query, channels, temporal_mode='...',
        #                minimum_rank=5, ground_truth_ids=None)
        if len(node.args) < 4:
            continue
        try:
            qid = node.args[0].value
            category = node.args[1].value
            query = node.args[2].value
        except AttributeError:
            continue
        gt_ids: List[int] = []
        for kw in node.keywords:
            if kw.arg == "ground_truth_ids" and isinstance(kw.value, (ast.List, ast.Tuple)):
                gt_ids = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
        queries.append({
            "id": qid,
            "category": category,
            "query": query,
            "ground_truth_ids": gt_ids,
            "labeled": bool(gt_ids),
        })
    return queries


def assign_family(query_text: str) -> Optional[str]:
    """Match query text against family patterns, return first family or None."""
    qlow = query_text.lower()
    for family_name, family in ANCHOR_FAMILIES.items():
        for pattern in family["patterns"]:
            if re.search(pattern, qlow):
                return family_name
    return None


def bucket(queries: List[dict]) -> Dict[str, List[dict]]:
    """Group queries by anchor family. Unmatched → 'unfamilied'."""
    buckets: Dict[str, List[dict]] = {fname: [] for fname in ANCHOR_FAMILIES}
    buckets["unfamilied"] = []
    for q in queries:
        family = assign_family(q["query"])
        buckets[family if family else "unfamilied"].append(q)
    return buckets


def emit_jsonl(buckets: Dict[str, List[dict]], out_path: Path) -> None:
    """One JSONL line per family. Empty families skipped."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for family_name, members in buckets.items():
            if not members:
                continue
            family_meta = ANCHOR_FAMILIES.get(family_name, {})
            unlabeled_members = [m for m in members if not m["labeled"]]
            labeled_members = [m for m in members if m["labeled"]]
            record = {
                "family": family_name,
                "description": family_meta.get("description", "(unfamilied — no anchor match)"),
                "seed_ground_truth_ids": family_meta.get("seed_ids", []),
                "labeled_count": len(labeled_members),
                "unlabeled_count": len(unlabeled_members),
                "labeled_query_ids": [m["id"] for m in labeled_members],
                "unlabeled_queries": [
                    {"id": m["id"], "category": m["category"], "query": m["query"]}
                    for m in unlabeled_members
                ],
            }
            f.write(json.dumps(record) + "\n")


def print_summary(buckets: Dict[str, List[dict]]) -> None:
    """Human-readable summary of bucketing."""
    total = sum(len(m) for m in buckets.values())
    total_unlabeled = sum(1 for m in buckets.values() for q in m if not q["labeled"])
    print(f"Total queries: {total}")
    print(f"Unlabeled: {total_unlabeled}")
    print()
    print(f"{'family':<28} {'labeled':>8} {'unlabeled':>10} {'seed_ids'}")
    print("-" * 80)
    for family_name in list(ANCHOR_FAMILIES) + ["unfamilied"]:
        members = buckets[family_name]
        labeled = sum(1 for q in members if q["labeled"])
        unlabeled = len(members) - labeled
        seed_ids = ANCHOR_FAMILIES.get(family_name, {}).get("seed_ids", [])
        print(f"{family_name:<28} {labeled:>8} {unlabeled:>10} {seed_ids}")
    print()
    print("Top unlabeled candidates per priority family (codex finding 2026-05-02):")
    priority_families = ["sarah_contact", "panel_labels", "lennar_lot_12",
                         "spanish_missing_materials", "qbo_reauth", "permit_paperwork", "gfci"]
    for fname in priority_families:
        unlabeled = [q for q in buckets[fname] if not q["labeled"]]
        if unlabeled:
            print(f"\n  [{fname}]")
            for q in unlabeled[:5]:
                print(f"    {q['id']} ({q['category']}): {q['query'][:80]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-summary", action="store_true",
                    help="Print human-readable bucket summary instead of writing JSONL")
    args = ap.parse_args()

    queries = parse_queries(QUERIES_FILE)
    buckets = bucket(queries)

    if args.print_summary:
        print_summary(buckets)
        return 0

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"{ts}-buckets.jsonl"
    emit_jsonl(buckets, out)
    print(f"→ wrote {out}")
    print_summary(buckets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
