#!/usr/bin/env python3
"""Parse a inception_bench result JSON, emit one TSV row.

Schema: {benchmark, schema, timestamp, config, baseline, post_dream?}
baseline = {phase, n_total, n_gradeable, recall@1, recall@5, recall@10, MRR,
            p50_ms, p95_ms, metrics_by_question_type:{<qtype>:{n,recall@1,recall@5,recall@10,MRR}}}
"""
import json
import sys
from pathlib import Path

QTYPES = ("knowledge-update", "multi-session", "single-session-assistant",
          "single-session-preference", "single-session-user", "temporal-reasoning")

def main():
    path = Path(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else path.stem
    iter_n = sys.argv[3] if len(sys.argv) > 3 else "?"
    d = json.loads(path.read_text())
    # Prefer post_dream phase if present, else baseline.
    block = d.get("post_dream") or d.get("baseline") or {}
    phase = block.get("phase", "baseline")
    mbqt = block.get("metrics_by_question_type") or {}
    qrow = [f"{(mbqt.get(q) or {}).get('recall@5', 0):.4f}" for q in QTYPES]
    cols = [iter_n, tag, phase,
            f"{block.get('recall@1', 0):.4f}",
            f"{block.get('recall@5', 0):.4f}",
            f"{block.get('recall@10', 0):.4f}",
            f"{block.get('MRR', 0):.4f}",
            f"{block.get('p50_ms', 0):.1f}"] + qrow
    print("\t".join(cols))

if __name__ == "__main__":
    main()
