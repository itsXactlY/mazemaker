#!/usr/bin/env python3
"""DAE evaluation orchestrator.

Runs LongMemEval-S three times via subprocess (baseline, +ColBERT@1.5,
+ColBERT+DAE), parses the per-run JSON, and emits a diff markdown
that states the locked verdict thresholds in the header.

Operator runs this on the dev box; the full --limit 0 sweep takes
~7.5h.  Smoke test with --limit 5 finishes in ~2 minutes.

Verdict thresholds (locked, written into every diff markdown):
  YES        — DAE delta vs ColBERT >= +1.0pp R@5 OR >= +1.5pp MRR
  NO         — DAE delta < +0.5pp R@5 AND < +0.5pp MRR
  GRAY ZONE  — anything in between; calls for further ablation
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
HARNESS = HERE / "longmemeval_s.py"


def _log(msg: str) -> None:
    print(f"[lme-dae-eval] {msg}", file=sys.stderr, flush=True)


def _run_config(name: str, flags: list[str], tag: str, limit: int) -> Path:
    full_tag = f"{tag}_{name}"
    cmd = [sys.executable, str(HARNESS), *flags,
           "--tag", full_tag, "--limit", str(limit)]
    _log(f"running {name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(HERE))
    if proc.returncode != 0:
        raise RuntimeError(f"config {name!r} subprocess exited {proc.returncode}")
    matches = sorted(
        RESULTS_DIR.glob(f"longmemeval_s_{full_tag}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise RuntimeError(f"no result JSON found for tag {full_tag!r}")
    return matches[-1]


def _metrics(path: Path) -> dict:
    payload = json.loads(path.read_text())
    m = payload.get("metrics", {})
    return {
        "R@1": float(m.get("recall@1", 0.0)),
        "R@5": float(m.get("recall@5", 0.0)),
        "R@10": float(m.get("recall@10", 0.0)),
        "MRR": float(m.get("MRR", 0.0)),
        "p50_ms": float(m.get("p50_ms", 0.0)),
        "_path": str(path),
    }


def _fmt(x: float, places: int = 4) -> str:
    return f"{x:.{places}f}"


def _verdict(d_r5: float, d_mrr: float) -> str:
    if d_r5 >= 0.010 or d_mrr >= 0.015:
        return ("**YES** — DAE clears the locked threshold "
                "(Delta R@5 >= +1.0pp OR Delta MRR >= +1.5pp vs ColBERT). "
                "Ship the channel wiring.")
    if d_r5 < 0.005 and d_mrr < 0.005:
        return ("**NO** — DAE delta below locked floor "
                "(Delta R@5 < +0.5pp AND Delta MRR < +0.5pp vs ColBERT). "
                "Keep the channel unwired; the second pass is not paying for itself.")
    return ("**GRAY ZONE** — DAE delta between floor and threshold. "
            "Run further ablation (sweep self_weight / neighbour_k) "
            "before deciding.")


def _markdown(results: dict, args, ts: str) -> str:
    base = results["baseline"]
    cb = results["colbert"]
    dae = results["dae"]
    d_r5 = dae["R@5"] - cb["R@5"]
    d_mrr = dae["MRR"] - cb["MRR"]
    d_r1 = dae["R@1"] - cb["R@1"]
    d_r10 = dae["R@10"] - cb["R@10"]
    d_p50 = dae["p50_ms"] - cb["p50_ms"]
    verdict = _verdict(d_r5, d_mrr)

    lines = [
        f"# LongMemEval-S DAE evaluation — {ts}",
        "",
        f"- limit: `{args.limit}` (0 = all 500)",
        f"- DAE knobs: self_weight=`{args.self_weight}`, "
        f"neighbour_k=`{args.neighbour_k}`, dae_weight=`{args.dae_weight}`",
        f"- ColBERT weight (configs ii + iii): `1.5`",
        "",
        "## Locked verdict thresholds (vs ColBERT)",
        "",
        "- **YES**: Delta R@5 >= +1.0pp **OR** Delta MRR >= +1.5pp",
        "- **NO**: Delta R@5 < +0.5pp **AND** Delta MRR < +0.5pp",
        "- **GRAY ZONE**: anything in between -> run further ablation",
        "",
        "## Metrics",
        "",
        "| Config | R@1 | R@5 | R@10 | MRR | p50 (ms) |",
        "|---|---|---|---|---|---|",
        f"| (i) baseline (hybrid) | {_fmt(base['R@1'])} | {_fmt(base['R@5'])} "
        f"| {_fmt(base['R@10'])} | {_fmt(base['MRR'])} | {_fmt(base['p50_ms'], 3)} |",
        f"| (ii) +ColBERT@1.5 | {_fmt(cb['R@1'])} | {_fmt(cb['R@5'])} "
        f"| {_fmt(cb['R@10'])} | {_fmt(cb['MRR'])} | {_fmt(cb['p50_ms'], 3)} |",
        f"| (iii) +ColBERT+DAE | {_fmt(dae['R@1'])} | {_fmt(dae['R@5'])} "
        f"| {_fmt(dae['R@10'])} | {_fmt(dae['MRR'])} | {_fmt(dae['p50_ms'], 3)} |",
        f"| Delta DAE vs ColBERT | {_fmt(d_r1)} | {_fmt(d_r5)} "
        f"| {_fmt(d_r10)} | {_fmt(d_mrr)} | {_fmt(d_p50, 3)} |",
        "",
        f"## Verdict",
        "",
        verdict,
        "",
        "## Source result files",
        "",
        f"- baseline: `{base['_path']}`",
        f"- colbert: `{cb['_path']}`",
        f"- dae: `{dae['_path']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="DAE eval orchestrator")
    p.add_argument("--self-weight", type=float, default=0.4)
    p.add_argument("--neighbour-k", type=int, default=20)
    p.add_argument("--dae-weight", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tag", default="dae_eval")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = {
        "baseline": ["--recall-mode", "hybrid"],
        "colbert": ["--recall-mode", "skynet",
                    "--enable-colbert", "--colbert-weight", "1.5"],
        "dae": ["--recall-mode", "skynet",
                "--enable-colbert", "--colbert-weight", "1.5",
                "--enable-dae",
                "--dae-weight", str(args.dae_weight),
                "--dae-self-weight", str(args.self_weight),
                "--dae-neighbour-k", str(args.neighbour_k)],
    }

    results: dict[str, dict] = {}
    for name, flags in configs.items():
        path = _run_config(name, flags, args.tag, args.limit)
        results[name] = _metrics(path)
        _log(f"{name} -> R@1={results[name]['R@1']:.4f} "
             f"R@5={results[name]['R@5']:.4f} "
             f"MRR={results[name]['MRR']:.4f}")

    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md = _markdown(results, args, ts)
    out = RESULTS_DIR / f"lme_dae_eval_{ts}.md"
    out.write_text(md)
    _log(f"wrote {out}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
