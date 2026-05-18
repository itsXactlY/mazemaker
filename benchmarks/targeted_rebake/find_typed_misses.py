#!/usr/bin/env python3
"""Identify benchmark queries whose gold session id isn't in top-5 with
the current Mazemaker champion stack.

Output is a JSON array of {qid, question, gold_sids, rank}, consumed by
the rebake scripts in the same directory.

Usage:
    python find_typed_misses.py <question_type> <output.json>

    # Example: find all single-session-preference queries that miss top-5
    python find_typed_misses.py single-session-preference ssp_misses.json

The script runs against the LongMemEval-oracle cache schema by default
(longmemeval_oracle_bgem3_1024). Set MM_POSTGRES_SCHEMA to override.

Environment knobs match the iter79 champion stack (loaded automatically
if not set; override to test different stacks):
    MAZEMAKER_INTENT_BOOST=0.10
    MAZEMAKER_TEMPORAL_WEIGHT=0.7
    MAZEMAKER_SALIENCE_WEIGHT=0.5
    MAZEMAKER_CANONICAL_PRIOR=0
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "benchmarks"))


def _set_default_env() -> None:
    defaults = {
        "MM_DB_BACKEND": "postgres",
        "MM_POSTGRES_DB": "mm10m_bench",
        "MM_POSTGRES_SCHEMA": "longmemeval_oracle_bgem3_1024",
        "MAZEMAKER_INTENT_BOOST": "0.10",
        "MAZEMAKER_CANONICAL_PRIOR": "0",
        "MAZEMAKER_TEMPORAL_WEIGHT": "0.7",
        "MAZEMAKER_SALIENCE_WEIGHT": "0.5",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    qtype = sys.argv[1]
    out_path = sys.argv[2]

    _set_default_env()

    from memory_client import Mazemaker
    from mazemaker_godbench import _preference_query, _rrf_fuse, rank_of_gold

    nm = Mazemaker(
        db_path="/dev/null",
        embedding_backend="auto",
        lazy_graph=True,
        retrieval_mode="skynet",
        rerank=True,
        channel_weights={"colbert": 2.5, "dae": 2.0},
        retrieval_candidates=512,
    )

    import psycopg
    raw_db = os.environ.get("MM_BENCH_RAW_DB", "mm_bench_raw")
    raw_schema = os.environ.get("MM_BENCH_RAW_SCHEMA", "longmemeval_oracle")
    pg_user = os.environ.get("MM_POSTGRES_USER", "mazemaker")
    pg_pass = os.environ.get("MM_POSTGRES_PASSWORD", "")
    pg_host = os.environ.get("MM_POSTGRES_HOST", "localhost")
    with psycopg.connect(dbname=raw_db, user=pg_user, password=pg_pass, host=pg_host) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT question_id, question, answer, answer_session_ids "
                f"FROM {raw_schema}.questions WHERE question_type=%s",
                (qtype,),
            )
            rows = cur.fetchall()

    print(f"{qtype}: {len(rows)} questions", flush=True)

    misses: list[dict] = []
    for qid, qtext, ans, sids in rows:
        sids_set = set(sids or [])
        q_pref = _preference_query(qtext)
        try:
            rlists = [
                nm.recall(qtext, k=10, hybrid=True, rerank=True,
                          enable_colbert=True, enable_dae=True)
            ]
        except Exception as e:
            print(f"  recall orig error on {qid}: {e}", flush=True)
            continue
        if q_pref:
            try:
                rlists.append(
                    nm.recall(q_pref, k=10, hybrid=True, rerank=True,
                              enable_colbert=True, enable_dae=True)
                )
            except Exception:
                # FTS tsquery sometimes barfs on certain preference rewrites;
                # skip the second pass when that happens.
                pass
        results = _rrf_fuse(rlists, k=10)
        rk = rank_of_gold(results, sids_set)
        status = "HIT" if rk and rk <= 5 else "MISS"
        if status == "MISS":
            misses.append({
                "qid": qid,
                "question": qtext,
                "answer": ans,
                "gold_sids": list(sids_set),
                "rank": rk,
            })
        print(f"[{status}] qid={qid:<14} rank={rk}", flush=True)

    with open(out_path, "w") as f:
        json.dump(misses, f, indent=2)
    print(f"\nMISSES: {len(misses)}/{len(rows)}", flush=True)
    print(f"saved to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
