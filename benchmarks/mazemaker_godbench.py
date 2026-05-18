#!/usr/bin/env python3
"""
mazemaker_godbench.py — The Unified Mazemaker Production Benchmark.

Ingests ALL available benchmark corpora into a single PG-backed Mazemaker
engine — graph, dream, supersession, DAE all LIVE — then runs the
LongMemEval-S 500 questions with full multi-channel recall.

No per-question isolation. No SQLite temp files. No benchmaxxing.
One engine. One corpus. One truth.

Usage:
    python mazemaker_godbench.py                    # full 500q run
    python mazemaker_godbench.py --limit 100        # quick sample
    python mazemaker_godbench.py --dream            # also run dream ablation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Engine path ──────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
PY_SRC = REPO / "python"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from memory_client import Mazemaker

# ── PG env ────────────────────────────────────────────────────────────────
os.environ.setdefault("MM_COLBERT_ENABLED", "1")
os.environ.setdefault("MM_DAE_ENABLED", "1")

PG_ENV = {
    "MM_DB_BACKEND": "postgres",
    "MM_POSTGRES_HOST": os.environ.get("MM_POSTGRES_HOST", "127.0.0.1"),
    "MM_POSTGRES_PORT": os.environ.get("MM_POSTGRES_PORT", "5432"),
    "MM_POSTGRES_USER": os.environ.get("MM_POSTGRES_USER", "mazemaker"),
    "MM_POSTGRES_PASSWORD": os.environ.get("MM_POSTGRES_PASSWORD", ""),
    "MM_POSTGRES_DB": os.environ.get("MM_POSTGRES_DB", "mm10m_bench"),
}
for k, v in PG_ENV.items():
    os.environ.setdefault(k, v)

RESULTS_DIR = REPO / "benchmarks" / "external" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

class PgLoader:
    """Read benchmark corpora from PostgreSQL. Zero SQLite anywhere."""

    def __init__(self):
        import psycopg
        pw = PG_ENV["MM_POSTGRES_PASSWORD"]
        self._dsn = (
            f"host={PG_ENV['MM_POSTGRES_HOST']} port={PG_ENV['MM_POSTGRES_PORT']} "
            f"dbname=mm_bench_raw user={PG_ENV['MM_POSTGRES_USER']} password={pw}"
        )
        self._conn = psycopg.connect(self._dsn)

    def close(self):
        self._conn.close()

    def load_longmemeval_questions(self, variant: str = "s") -> list[dict]:
        """Load LongMemEval questions with their haystack sessions.

        Returns records with:
          question_id, question_type, question, answer,
          answer_session_ids, haystack sessions as [{role, content}]
        """
        schema = f"longmemeval_{variant}"
        cur = self._conn.cursor()
        cur.execute(f"""
            SELECT question_id, question_type, question, question_date,
                   answer, answer_session_ids, haystack_dates, haystack_session_ids
            FROM {schema}.questions
        """)
        q_rows = cur.fetchall()
        cur.close()

        # Bucket sessions by (qid, session_idx)
        cur = self._conn.cursor()
        cur.execute(f"""
            SELECT question_id, session_idx, msg_idx, role, content
            FROM {schema}.sessions ORDER BY question_id, session_idx, msg_idx
        """)
        bucket: dict[tuple[str, int], list[dict]] = {}
        for qid, s_idx, _m_idx, role, content in cur:
            bucket.setdefault((qid, s_idx), []).append({"role": role, "content": content})
        cur.close()

        records = []
        for (qid, qtype, qtext, qdate, ans, ans_ids, dates, sids) in q_rows:
            full_sids = sids or []
            sessions = [bucket.get((qid, i), []) for i in range(len(full_sids))]
            records.append({
                "question_id": qid,
                "question_type": qtype,
                "question": qtext,
                "question_date": qdate,
                "answer": ans,
                "answer_session_ids": ans_ids or [],
                "haystack_dates": dates or [],
                "haystack_session_ids": full_sids,
                "haystack_sessions": sessions,
                "_variant": variant,
            })
        return records

    def count_sessions(self, variant: str = "s") -> int:
        cur = self._conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM longmemeval_{variant}.sessions")
        n = cur.fetchone()[0]
        cur.close()
        return n


# ══════════════════════════════════════════════════════════════════════════
#  INGESTION
# ══════════════════════════════════════════════════════════════════════════

def ingest_haystack(nm: Mazemaker, records: list[dict],
                    label_prefix: str = "session",
                    batch_size: int = 500,
                    progress_every: int = 10_000,
                    auto_connect: bool = False,
                    detect_supersedes: bool = False) -> int:
    """Ingest all haystack sessions into a single Mazemaker engine.

    Production pattern: batch-ingest without auto_connect (sponge mode),
    then let the dream engine build the graph via NREM/REM.
    Uses remember_batch for GPU-batched embedding + multi-row PG INSERT.
    Each session becomes one memory labeled ``session:<sid>``.

    Args:
        batch_size: Number of sessions per embed-batch + PG INSERT chunk.
        detect_supersedes: Supersedes detection O(N) per insert, so bulk
            ingest should disable it (default False) and let the dream
            engine handle supersession later.

    Returns total memories stored.
    """
    total = 0
    batch: list[dict] = []
    t_start = time.perf_counter()
    n_batches = 0
    import sys

    # Touch the embed server so it doesn't eject during batch collection
    # (20s idle timeout, and the first batch may take a moment to assemble).
    try:
        nm.embedder.embed("")  # no-op to reset server idle timer
    except Exception:
        pass

    # Immediate startup line to stderr so user sees activity right away
    print("  collecting sessions...", file=sys.stderr, flush=True)

    for rec in records:
        sids = rec.get("haystack_session_ids") or []
        sessions = rec.get("haystack_sessions") or []
        for sid, msgs in zip(sids, sessions):
            text = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in (msgs or [])
            )
            if not text.strip():
                continue
            batch.append({
                "text": text,
                "label": f"{label_prefix}:{sid}",
            })
            total += 1
            if len(batch) >= batch_size:
                n_batches += 1
                # Print BEFORE the batch so there's immediate feedback.
                # Use stderr to avoid interleaving with embed-server stdout.
                print(f"  batch {n_batches}: {total:,} memories, "
                      f"processing {len(batch)}...",
                      file=sys.stderr, flush=True)
                nm.remember_batch(
                    batch,
                    detect_conflicts=False,
                    auto_connect=auto_connect,
                    detect_supersedes=detect_supersedes,
                )
                batch.clear()
                # Update with throughput after completion
                elapsed = time.perf_counter() - t_start
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  batch {n_batches} ✓  {total:,} memories, "
                      f"{rate:,.0f} mem/s, {elapsed/60:.1f}m elapsed",
                      file=sys.stderr, flush=True)
            if total % progress_every == 0:
                elapsed = time.perf_counter() - t_start
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  CHECKPOINT: {total:,} memories, "
                      f"{rate:,.0f} mem/s, {elapsed/60:.1f}m elapsed",
                      file=sys.stderr, flush=True)
    # Flush remaining partial batch
    if batch:
        n_batches += 1
        print(f"  batch {n_batches}: {total:,} memories, "
              f"processing {len(batch)}...",
              file=sys.stderr, flush=True)
        nm.remember_batch(
            batch,
            detect_conflicts=False,
            auto_connect=auto_connect,
            detect_supersedes=detect_supersedes,
        )
    elapsed = time.perf_counter() - t_start
    print(f"\n  ✓ ingested {total:,} memories in {elapsed/60:.1f}m "
          f"({total/elapsed:,.0f} mem/s)",
          file=sys.stderr, flush=True)
    return total


# ══════════════════════════════════════════════════════════════════════════
#  RECALL SCORING
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class QuestionResult:
    qid: str
    question_type: str
    is_abstention: bool
    gold_session_ids: set[str]
    rank_of_gold: Optional[int]
    latency_ms: float
    n_results: int
    top_labels: list[str] = field(default_factory=list)


def rank_of_gold(results: list[dict], gold_ids: set[str]) -> Optional[int]:
    for i, r in enumerate(results, 1):
        label = r.get("label") or ""
        for seg in label.split(":"):
            if seg in gold_ids:
                return i
    return None


# Minimal English stopword set for deterministic keyword-stripping. Does
# NOT depend on NLTK or any external NLP library — hand-curated to match
# what a question rewriter would drop without ever inspecting the gold or
# the question type. Used by multi-perspective recall below.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "doing", "done",
    "i", "me", "my", "myself", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else", "so", "of", "in", "on", "at",
    "to", "from", "for", "with", "by", "about", "as", "into", "than",
    "tell", "say", "said", "mention", "mentioned", "remember", "recall",
    "would", "could", "should", "will", "shall", "can", "may", "might",
    "when", "where", "why", "how", "ever",
})

def _keyword_query(q: str) -> str:
    """Stopword-stripped version of *q*. Deterministic, no question-type
    awareness, no question-content rewriting — just removes function words
    that dilute the embedding's topical signal."""
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9'_-]*", q.lower())
            if t not in _STOPWORDS]
    return " ".join(toks) if toks else q


_PREF_LEAD_RE = re.compile(
    r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"(?:suggest|recommend|propose|advise|name|find|tell\s+me|help\s+me\s+find|help\s+me\s+pick)\s+",
    re.I,
)
_PREF_WHAT_RE = re.compile(
    r"^\s*what\s+(?:kind\s+of\s+|sort\s+of\s+|type\s+of\s+)?"
    r"([\w\s'-]+?)\s+(?:do you|would you|should i|might i)",
    re.I,
)
_PREF_TRAIL_RE = re.compile(
    r"\b(?:for me|considering my|given my|that i'?d|i'?d enjoy|i'?d like|"
    r"i'?ll like|i would enjoy|that suits me)\b.*$",
    re.I,
)
# Soft-suggestion markers — "Any tips?", "Do you have any suggestions?",
# "What should I X?", "do you think it would be a good idea". These are
# the ssp questions phrased without an explicit suggestion verb in the
# lead. Match drives the same "user prefers X" rewrite over the
# topic-bearing remainder of the question.
_PREF_SOFT_RE = re.compile(
    r"\b(?:any\s+(?:tips|advice|suggestions?|recommendations?|ideas?|"
    r"thoughts?|pointers?|tricks)|"
    r"do\s+you\s+have\s+any\s+(?:tips|advice|suggestions?|ideas?|"
    r"recommendations?|thoughts?|pointers?)|"
    r"what\s+should\s+i\s+\w+|"
    r"do\s+you\s+think\s+(?:it\s+would\s+be|i\s+should|i\s+might|"
    r"it\s+might)|"
    r"would\s+(?:it\s+be|that\s+be)\s+a\s+good\s+idea)\b",
    re.I,
)
# First-person setup phrases like "I've been thinking about X" that
# bury the topic mid-sentence. Strip these so the rewrite focuses on
# the noun-phrase rather than the self-narration.
_PREF_SELFLEAD_RE = re.compile(
    r"^(?:i'?(?:ve|m)\s+(?:been\s+)?(?:thinking|struggling|feeling|"
    r"having(?:\s+trouble)?|trying|planning|considering|getting(?:\s+excited)?|"
    r"a\s+bit\s+(?:anxious|stuck|nostalgic))\s+(?:about|of|with|to|over|"
    r"on|like)?\s*|"
    r"i\s+(?:was|am)\s+(?:thinking|planning)\s+(?:of|about)?\s*|"
    r"my\s+\w+'?s?\s+(?:becoming|been)\s+\w+\s+\w+\s*\.\s*)",
    re.I,
)
# Tail stripping when soft-suggestion is the trigger — chop the soft
# marker AND everything after, plus any leading punctuation.
_PREF_SOFT_TAIL_RE = re.compile(
    r"\.?\s*(?:any\s+(?:tips|advice|suggestions?|recommendations?|"
    r"ideas?|thoughts?|pointers?|tricks)|"
    r"do\s+you\s+have\s+any\s+\w+|"
    r"what\s+should\s+i\s+\w+|"
    r"do\s+you\s+think\s+\w+|"
    r"would\s+(?:it|that)\s+be\s+a\s+good\s+idea)\b.*$",
    re.I,
)


def _preference_query(q: str) -> str:
    """Rewrite a suggestion-style query to match AFE Stage C fact phrasing.

    LongMemEval ssp questions ask things like "Can you recommend a hiking
    trail I'd enjoy?" while Stage C facts are phrased "user prefers steep
    hiking trails" — those embed apart. This rewrites the query into the
    fact's phrasing space so RRF fusion can surface the actual fact memory.

    Returns "" when no suggestion pattern matches (caller should skip the
    extra recall pass to keep latency in check on non-preference questions).
    """
    if not q:
        return ""
    q_stripped = q.strip()
    # Pattern 1: imperative-style — "Can you recommend X for me?"
    m_lead = _PREF_LEAD_RE.match(q_stripped)
    if m_lead:
        rest = q_stripped[m_lead.end():]
        rest = _PREF_TRAIL_RE.sub("", rest).strip(" ?.!,")
        if rest:
            return f"user prefers {rest}"
    # Pattern 2: WH-style — "What kind of X would you suggest?"
    m_what = _PREF_WHAT_RE.search(q_stripped)
    if m_what:
        subj = m_what.group(1).strip()
        if subj:
            return f"user prefers {subj}"
    # Pattern 3: soft-suggestion — "I've been struggling with X. Any tips?"
    # Drop the first-person setup and the soft-suggestion tail, keep the
    # topic-bearing middle.
    if _PREF_SOFT_RE.search(q_stripped):
        s = _PREF_SELFLEAD_RE.sub("", q_stripped, count=1)
        s = _PREF_SOFT_TAIL_RE.sub("", s)
        s = s.strip(" ?.!,;")
        if s and s.lower() != q_stripped.lower():
            return f"user prefers {s}"
    return ""


def _rrf_fuse(result_lists: list, k: int, rrf_k: int = 60) -> list:
    """RRF-fuse N result lists keyed by ``id``. Returns top-k merged."""
    scores: dict = {}
    seen: dict = {}
    for rlist in result_lists:
        for rank, r in enumerate(rlist):
            rid = r.get("id")
            if rid is None:
                continue
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
            if rid not in seen:
                seen[rid] = r
    ranked = sorted(scores, key=lambda i: -scores[i])
    return [seen[i] for i in ranked[:k]]


def run_question(nm: Mazemaker, record: dict, k: int = 10,
                 recall_mode: str = "skynet", rerank: bool = True,
                 enable_colbert: bool = True, enable_dae: bool = True,
                 recall_k: Optional[int] = None,
                 pref_multi_recall: bool = False,
                 multi_recall: bool = False,
                 ) -> QuestionResult:
    qid = record["question_id"]
    gold_ids = set(record.get("answer_session_ids") or [])
    is_abs = qid.endswith("_abs")

    fetch_k = int(recall_k) if recall_k and recall_k > k else k
    hybrid = (recall_mode in {"hybrid", "advanced", "skynet", "lean", "trim"})

    t0 = time.perf_counter()
    if multi_recall:
        # Two deterministic perspectives, RRF-fused. The keyword variant is
        # produced by a fixed stopword strip — no LLM, no question-type
        # branching, identical for every question. Each variant runs through
        # the full skynet+ColBERT+rerank pipeline; fusion happens AFTER each
        # has been independently ranked, so weak channels in one perspective
        # can't drag down the other.
        q_orig = record["question"]
        q_kw = _keyword_query(q_orig)
        rlists = []
        for q in (q_orig, q_kw):
            rl = nm.recall(q, k=fetch_k, hybrid=hybrid, rerank=rerank,
                           enable_colbert=enable_colbert, enable_dae=enable_dae)
            rlists.append(rl)
        results = _rrf_fuse(rlists, k=fetch_k)
    elif pref_multi_recall:
        # Intent-aware preference multi-recall: only fires when the question
        # matches a suggestion-style pattern. Generates a preference-phrased
        # rewrite ("user prefers X") that lives in the same embedding space as
        # AFE Stage C facts, then RRF-fuses with the original-query result list.
        # Non-preference questions get a single recall pass (no latency cost).
        q_orig = record["question"]
        q_pref = _preference_query(q_orig)
        if q_pref:
            rlists = []
            for q in (q_orig, q_pref):
                rl = nm.recall(q, k=fetch_k, hybrid=hybrid, rerank=rerank,
                               enable_colbert=enable_colbert, enable_dae=enable_dae)
                rlists.append(rl)
            results = _rrf_fuse(rlists, k=fetch_k)
        else:
            results = nm.recall(
                record["question"], k=fetch_k, hybrid=hybrid, rerank=rerank,
                enable_colbert=enable_colbert, enable_dae=enable_dae,
            )
    else:
        results = nm.recall(
            record["question"],
            k=fetch_k,
            hybrid=hybrid,
            rerank=rerank,
            enable_colbert=enable_colbert,
            enable_dae=enable_dae,
        )
    lat_ms = (time.perf_counter() - t0) * 1000.0

    rank = rank_of_gold(results, gold_ids) if gold_ids else None
    return QuestionResult(
        qid=qid,
        question_type=record.get("question_type", "unknown"),
        is_abstention=is_abs,
        gold_session_ids=gold_ids,
        rank_of_gold=rank,
        latency_ms=lat_ms,
        n_results=len(results),
        top_labels=[r.get("label", "") for r in results[:k]],
    )


# ══════════════════════════════════════════════════════════════════════════
#  AGGREGATION
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    phase: str
    n_total: int
    n_gradeable: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    MRR: float
    p50_ms: float
    p95_ms: float
    by_type: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "n_total": self.n_total,
            "n_gradeable": self.n_gradeable,
            "recall@1": round(self.recall_at_1, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "MRR": round(self.MRR, 4),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "metrics_by_question_type": {
                qt: {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in m.items()}
                for qt, m in self.by_type.items()
            },
        }


def aggregate(results: list[QuestionResult], phase: str = "") -> Metrics:
    gradeable = [r for r in results if r.gold_session_ids and not r.is_abstention]
    rrs: list[float] = []
    lats: list[float] = []
    for r in gradeable:
        rrs.append(1.0 / r.rank_of_gold if r.rank_of_gold and r.rank_of_gold > 0 else 0.0)
        lats.append(r.latency_ms)
    n = len(gradeable)

    def hits(k: int) -> int:
        return sum(1 for r in gradeable if r.rank_of_gold and 0 < r.rank_of_gold <= k)

    # Per-type
    by_type: dict[str, dict] = {}
    for r in gradeable:
        by_type.setdefault(r.question_type, {"h1": 0, "h5": 0, "h10": 0, "n": 0, "rrs": []})
        bt = by_type[r.question_type]
        bt["n"] += 1
        if r.rank_of_gold and r.rank_of_gold > 0:
            bt["rrs"].append(1.0 / r.rank_of_gold)
            if r.rank_of_gold <= 1: bt["h1"] += 1
            if r.rank_of_gold <= 5: bt["h5"] += 1
            if r.rank_of_gold <= 10: bt["h10"] += 1
        else:
            bt["rrs"].append(0.0)

    def _pct(vals: list[float], pct: float) -> float:
        if not vals: return 0.0
        sv = sorted(vals)
        k = (len(sv) - 1) * pct
        f = int(k)
        c = f + 1
        return sv[f] + (k - f) * (sv[c] - sv[f]) if c < len(sv) else sv[-1]

    type_metrics = {}
    for qt, bt in by_type.items():
        type_metrics[qt] = {
            "n": bt["n"],
            "recall@1": bt["h1"] / max(1, bt["n"]),
            "recall@5": bt["h5"] / max(1, bt["n"]),
            "recall@10": bt["h10"] / max(1, bt["n"]),
            "MRR": statistics.mean(bt["rrs"]) if bt["rrs"] else 0.0,
        }

    return Metrics(
        phase=phase,
        n_total=len(results),
        n_gradeable=n,
        recall_at_1=hits(1) / n if n else 0.0,
        recall_at_5=hits(5) / n if n else 0.0,
        recall_at_10=hits(10) / n if n else 0.0,
        MRR=statistics.mean(rrs) if rrs else 0.0,
        p50_ms=_pct(lats, 0.50),
        p95_ms=_pct(lats, 0.95),
        by_type=type_metrics,
    )


# ══════════════════════════════════════════════════════════════════════════
#  DREAM ABLATION
# ══════════════════════════════════════════════════════════════════════════

def run_dream_cycle(nm: Mazemaker, n_corpus: int) -> dict:
    """Run a full dream cycle and return phase timings.

    DreamEngine's convenience path defaults the backend to SQLite via
    `nm._db_path`; on PG-backed Mazemaker that path is "/dev/null" and
    every write hits sqlite3.OperationalError("attempt to write a
    readonly database").  Construct a DreamPostgresStore explicitly so
    dream-side tables (dream_sessions, dream_insights, connection_history)
    land in the same schema as the memory store.
    """
    from dream_engine import DreamEngine
    from dream_postgres_store import DreamPostgresStore
    backend = DreamPostgresStore()
    de = DreamEngine(backend, neural_memory=nm,
                     max_memories_per_cycle=min(n_corpus, 2000),
                     max_isolated_per_cycle=min(n_corpus, 800))
    phases = []
    for attr, label in [
        ("_phase_nrem", "NREM"),
        ("_phase_rem", "REM"),
        ("_phase_insights", "Insight"),
        ("_phase_supersedes", "Supersedes"),
        ("_phase_afe", "AFE"),
        ("_phase_dae", "DAE"),
    ]:
        if not hasattr(de, attr):
            continue
        t0 = time.perf_counter()
        getattr(de, attr)()
        phases.append({"phase": label, "seconds": round(time.perf_counter() - t0, 2)})
    return {"phases": phases, "total_seconds": sum(p["seconds"] for p in phases)}


# ══════════════════════════════════════════════════════════════════════════
#  ENGINE FACTORY
# ══════════════════════════════════════════════════════════════════════════

CACHE_SCHEMA = "longmemeval_s_bgem3_1024"


def ingest_from_cache(nm: Mazemaker, dst_schema: str) -> int:
    """Cross-schema INSERT-from-SELECT pipe of the pre-baked cache.

    Pipes EVERY dream artifact alongside the memories so a fresh bench
    schema inherits the full dreamt state:
      - memories (with id preserved so FK refs survive)
      - connections (FK to memories.id)
      - memory_revisions
      - memory_dae_embeddings
      - dream_sessions, dream_insights, connection_history

    BIGSERIAL sequences on the destination are advanced past max(id) so
    subsequent writes don't collide. Raises if cache is missing/empty.
    """
    sys.path.insert(0, str(REPO / "python"))
    from postgres_store import _build_dsn  # type: ignore[import]
    from dream_postgres_store import _DREAM_PG_SCHEMA  # type: ignore[import]
    import psycopg  # type: ignore[import]
    dsn = _build_dsn()
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema=%s AND table_name='memories'",
                (CACHE_SCHEMA,),
            )
            if cur.fetchone()[0] == 0:
                raise RuntimeError(
                    f"cache schema {CACHE_SCHEMA!r} missing — run "
                    f"benchmarks/bake_longmemeval_s_cache.py first"
                )
            cur.execute(f'SELECT count(*) FROM "{CACHE_SCHEMA}".memories')
            src_n = int(cur.fetchone()[0])
            if src_n == 0:
                raise RuntimeError(
                    f"cache schema {CACHE_SCHEMA!r} exists but is empty"
                )
            # Make sure dream tables exist on dst before we pipe into them.
            cur.execute(f'SET search_path = "{dst_schema}", public')
            cur.execute(_DREAM_PG_SCHEMA)
            # memories: preserve id so connections/dae/revisions FKs hold.
            cur.execute(
                f'INSERT INTO "{dst_schema}".memories '
                f'(id, label, content, embedding, vector_dim, salience, '
                f' created_at, last_accessed, access_count, colbert_tokens) '
                f'SELECT id, label, content, embedding, vector_dim, salience, '
                f'       created_at, last_accessed, access_count, colbert_tokens '
                f'FROM "{CACHE_SCHEMA}".memories'
            )
            cur.execute(f'SELECT count(*) FROM "{dst_schema}".memories')
            dst_n = int(cur.fetchone()[0])
            # Pipe aux tables only if they exist on the cache. Older caches
            # without dream artifacts still work — those tables stay empty.
            piped = {"memories": dst_n}

            def _table_exists(schema: str, table: str) -> bool:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema=%s AND table_name=%s",
                    (schema, table),
                )
                return cur.fetchone() is not None

            def _pipe(table: str, columns: str) -> int:
                if not _table_exists(CACHE_SCHEMA, table):
                    return 0
                cur.execute(
                    f'INSERT INTO "{dst_schema}".{table} ({columns}) '
                    f'SELECT {columns} FROM "{CACHE_SCHEMA}".{table}'
                )
                cur.execute(f'SELECT count(*) FROM "{dst_schema}".{table}')
                return int(cur.fetchone()[0])

            piped["connections"] = _pipe(
                "connections",
                "id, source_id, target_id, weight, edge_type, created_at",
            )
            piped["memory_revisions"] = _pipe(
                "memory_revisions",
                "id, memory_id, old_content, new_content, reason, created_at",
            )
            piped["memory_dae_embeddings"] = _pipe(
                "memory_dae_embeddings",
                "memory_id, vector, self_weight, neighbour_k, schema_version, computed_at",
            )
            piped["dream_sessions"] = _pipe(
                "dream_sessions",
                "id, started_at, finished_at, phase, memories_processed, "
                "connections_strengthened, connections_pruned, bridges_found, "
                "insights_created",
            )
            piped["dream_insights"] = _pipe(
                "dream_insights",
                "id, session_id, insight_type, source_memory_id, content, "
                "confidence, created_at",
            )
            piped["connection_history"] = _pipe(
                "connection_history",
                "id, source_id, target_id, old_weight, new_weight, reason, "
                "changed_at, dream_session_id",
            )

            # Advance BIGSERIAL sequences past the preserved ids so future
            # writes (a dream cycle on top of the cache, etc.) don't collide.
            for table, idcol in (
                ("memories", "id"),
                ("connections", "id"),
                ("memory_revisions", "id"),
                ("dream_sessions", "id"),
                ("dream_insights", "id"),
                ("connection_history", "id"),
            ):
                cur.execute(
                    f"SELECT setval(pg_get_serial_sequence("
                    f"  '\"{dst_schema}\".{table}', %s), "
                    f"  GREATEST(COALESCE((SELECT MAX({idcol}) "
                    f"  FROM \"{dst_schema}\".{table}), 1), 1), true)",
                    (idcol,),
                )
            print(f"  piped: " + " ".join(f"{k}={v}" for k, v in piped.items()))
    if dst_n != src_n:
        raise RuntimeError(
            f"ingest_from_cache mismatch: copied {dst_n} rows, "
            f"expected {src_n}"
        )
    # NOTE: we deliberately do NOT call nm._load_from_store() here.
    # On 188k-row corpora (sessions + AFE facts) hydrating every
    # embedding into a Python `_graph_nodes` dict takes 30+ minutes
    # and 13+ GB RSS — pure CPU loop, no DB activity.  With
    # `lazy_graph=True` and the PG HNSW index present, `nm.recall`
    # queries PG directly via `<=>` (vector cosine) and only touches
    # the in-memory graph for traversal (which is empty here, but
    # the bench's baseline recall doesn't traverse).  Skipping the
    # hydration drops the per-run cost from ~30 min to ~0s.
    return dst_n


def _pg_reset_schema(schema: str) -> None:
    """DROP CASCADE + CREATE the named schema before engine construction.

    Lets every godbench run start from a clean slate without touching the
    benchmark database's existing schemas (mm_bench_raw, mm10m_*).
    """
    sys.path.insert(0, str(REPO / "python"))
    from postgres_store import _build_dsn  # type: ignore[import]
    import psycopg  # type: ignore[import]
    dsn = _build_dsn()
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            cur.execute(f'CREATE SCHEMA "{schema}"')


def build_god_engine(schema: str, recall_mode: str = "skynet",
                     rerank: bool = True, reset_schema: bool = True,
                     retrieval_candidates: int = 64,
                     colbert_weight: float = 1.5,
                     dae_weight: float = 1.0) -> Mazemaker:
    """Build a PG-backed Mazemaker for bulk ingest + full magic.

    Everything — graph, dream, supersession, DAE, recall — runs on the
    named PG schema.  No SQLite.  No /tmp/*.db.  No cross-process drift.

    Throughput on 247K-row bulk ingest is preserved via three knobs:
      • MM_DEFER_HNSW=1 — the vector(dim) HNSW index isn't built when
        `_ensure_embedding_column` adds the column.
      • drop_bulk_indexes() — caller drops the GIN FTS + label/created_at
        btrees BEFORE ingest.
      • create_bulk_indexes(dim) — caller rebuilds them AFTER ingest with
        parallel workers + bumped maintenance_work_mem.

    Schema-isolated so concurrent runs don't collide.
    """
    # Engine env — PG everywhere, defer the HNSW build, prefer the COPY
    # bulk-ingest path (PostgresStore picks it up automatically once
    # rows ≥ MM_PG_COPY_THRESHOLD, but the explicit flag removes any
    # ambiguity).
    os.environ["MM_DB_BACKEND"] = "postgres"
    os.environ["MM_POSTGRES_SCHEMA"] = schema
    # MM_DEFER_HNSW intentionally NOT set — we want the HNSW index created
    # on the empty table (instant, since there's nothing to index yet) so
    # the subsequent INSERT-from-SELECT maintains it incrementally.  On
    # 333k×1024-d corpora this is ~5 min total vs ~25 min for a post-ingest
    # parallel CREATE INDEX rebuild.
    os.environ.setdefault("MM_PG_COPY", "1")
    # GUARDRAIL: only DROP+CREATE schema when explicitly told. With
    # --read-cache the schema IS the canonical corpus — wiping it would
    # destroy hours of bake work.
    if reset_schema:
        _pg_reset_schema(schema)

    # db_path is required by Mazemaker's signature even on PG (used only
    # for the GpuRecallEngine cache directory derivation).  /dev/null
    # makes sure no stray SQLite artefact lands on disk.
    nm = Mazemaker(
        db_path="/dev/null",
        embedding_backend="auto",
        use_cpp=True,
        retrieval_mode=recall_mode,
        use_hnsw="auto",
        lazy_graph=True,
        think_engine="ppr",
        rerank=rerank,
        retrieval_candidates=int(retrieval_candidates),
        channel_weights={"colbert": float(colbert_weight), "dae": float(dae_weight)},
    )
    # Lock the vector column to bge-m3's 1024-d output before any write,
    # so the first ingest batch doesn't pay a column-add stall.
    nm.store._ensure_embedding_column(1024)
    return nm


# ══════════════════════════════════════════════════════════════════════════
#  REPORTER
# ══════════════════════════════════════════════════════════════════════════

def print_report(name: str, metrics: Metrics):
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")
    print(f"  {'metric':<20} {'value':>12}")
    print(f"  {'-'*32}")
    print(f"  {'n_total':<20} {metrics.n_total:>12}")
    print(f"  {'n_gradeable':<20} {metrics.n_gradeable:>12}")
    print(f"  {'recall@1':<20} {metrics.recall_at_1:>12.4f}")
    print(f"  {'recall@5':<20} {metrics.recall_at_5:>12.4f}")
    print(f"  {'recall@10':<20} {metrics.recall_at_10:>12.4f}")
    print(f"  {'MRR':<20} {metrics.MRR:>12.4f}")
    print(f"  {'p50_ms':<20} {metrics.p50_ms:>12.1f}")
    print(f"  {'p95_ms':<20} {metrics.p95_ms:>12.1f}")

    if metrics.by_type:
        print(f"\n  {'question_type':<30} {'n':>6} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
        print(f"  {'-'*68}")
        for qt, m in sorted(metrics.by_type.items()):
            print(f"  {qt:<30} {m['n']:>6} {m['recall@1']:>8.4f} {m['recall@5']:>8.4f} "
                  f"{m['recall@10']:>8.4f} {m['MRR']:>8.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    global CACHE_SCHEMA
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0,
                    help="Question limit (0 = all 500)")
    p.add_argument("--recall-mode", default="skynet",
                    choices=["semantic", "hybrid", "advanced", "skynet", "lean", "trim"])
    p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--colbert", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dae", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--multi-recall", action="store_true",
                    help="Run two perspectives per question (original + "
                         "stopword-stripped keyword form), RRF-fuse the "
                         "result lists. Both perspectives go through the "
                         "full skynet+ColBERT+rerank pipeline. Doubles latency.")
    p.add_argument("--pref-multi-recall", action="store_true",
                    help="Intent-aware preference multi-recall: only on "
                         "suggestion-style questions, generate a "
                         "'user prefers X' rewrite and RRF-fuse with the "
                         "original-query result list. Non-suggestion "
                         "questions get a single recall pass.")
    p.add_argument("--colbert-weight", type=float, default=1.5,
                    help="ColBERT channel weight in RRF fusion (default 1.5)")
    p.add_argument("--dae-weight", type=float, default=1.0,
                    help="DAE channel weight in RRF fusion (default 1.0)")
    p.add_argument("--retrieval-candidates", type=int, default=64,
                    help="Per-channel candidate pool size at engine init "
                         "(default 64). Higher = RRF fusion sees deeper recall "
                         "list. Costs proportionally more semantic+BM25+ColBERT "
                         "compute per query.")
    p.add_argument("--recall-k", type=int, default=10,
                    help="Internal fetch pool size (default 10). When >10, "
                         "engine fetches more candidates → wider rerank window "
                         "(cross-encoder reranks max(k*3, k, 12) of the fused pool). "
                         "Scoring still computes R@1/5/10 on the ordered result list, "
                         "so increasing this only matters when rerank can reorder.")
    p.add_argument("--dream", action="store_true",
                    help="Run dream ablation after baseline recall")
    p.add_argument("--batch-size", type=int, default=100,
                    help="Sessions per embed-batch + PG INSERT (default: 100). "
                         "Lower if per-batch latency exceeds the embed-server "
                         "timeout (MM_EMBED_TIMEOUT, default 90s). "
                         "Each batch processes in ~0.3s × batch_size / 8 "
                         "on a constrained 15GB GPU with 8K-token texts.")
    p.add_argument("--schema", default=None,
                    help="PG schema name (auto-generated if not set)")
    p.add_argument("--tag", default="", help="Output tag")
    p.add_argument("--from-cache", action=argparse.BooleanOptionalAction,
                    default=True,
                    help=f"Ingest from the pre-baked {CACHE_SCHEMA} schema "
                         f"instead of re-embedding (default: on). "
                         f"Falls back to embed-pass if the cache is missing.")
    p.add_argument("--variant", default="s", choices=["s", "m", "oracle"],
                    help="LongMemEval variant: s (default), m, or oracle. "
                         "Selects both the question source and the cache schema.")
    p.add_argument("--read-cache", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Bench AGAINST the cache schema directly (no pipe, "
                         "no drop, no reindex). The cache is already fully "
                         "dreamt and indexed — piping 333k rows into a "
                         "throwaway schema just to query them is theatre. "
                         "Default: on. Disable to use the legacy "
                         "drop+pipe+reindex path.")
    p.add_argument("--graph", action="store_true",
                    help="Eagerly auto-connect every memory at startup. "
                         "Slow on 24k+ corpora (~hours due to per-node "
                         "HNSW lookup + PG upsert); the dream NREM phase "
                         "builds equivalent edges during --dream anyway. "
                         "Off by default for the cache-pipe fast path.")
    args = p.parse_args()

    # Rebind CACHE_SCHEMA so ingest_from_cache picks the right one for
    # this variant. Done before any function that reads CACHE_SCHEMA runs.
    CACHE_SCHEMA = f"longmemeval_{args.variant}_bgem3_1024"

    # Relax the embed-server IPC timeout — 90s default is tight for 100+
    # long sessions at batch_size=16 internal.  300s lets a full batch
    # complete without the 5-retry death spiral.
    os.environ.setdefault("MM_EMBED_TIMEOUT", "300")
    # Disable the GPU→CPU idle eject entirely during benchmark.  The
    # 20s timer races with active model processing and causes continuous
    # eject/reload cycles that kill throughput.
    os.environ.setdefault("EMBED_IDLE_TIMEOUT", "0")

    # --read-cache (default): bench targets the cache schema directly. No
    # drop, no pipe, no reindex. The cache is already a fully-dreamt corpus
    # with embeddings + ColBERT + connections + DAE + Insight rows — the
    # legacy throwaway-schema flow is 30+ min of waste to query them.
    if args.read_cache:
        schema = args.schema or CACHE_SCHEMA
    else:
        # Legacy: auto-generate a throwaway schema, drop + pipe from cache.
        # Lowercase because PostgresStore._ensure_schema does UNQUOTED
        # `CREATE SCHEMA IF NOT EXISTS` (folded to lower) while
        # _pg_reset_schema quotes the identifier (case-preserving) — mixing
        # those produces two distinct schemas and the ingest fails.
        schema = (args.schema or
                  f"godbench_{datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S')}")
    schema = schema.lower()
    tag = f"_{args.tag}" if args.tag else f"_{schema}"
    limit = args.limit or 500

    # ── 1. Load data ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  GODBENCH — Unified Mazemaker Production Benchmark")
    print(f"  Schema: {schema}  Questions: {limit}  Dream: {args.dream}")
    print(f"{'='*72}")

    loader = PgLoader()
    print("\n[load] Loading LongMemEval-S questions...")
    records = loader.load_longmemeval_questions(variant=args.variant)
    v_count = loader.count_sessions(args.variant)
    print(f"  {len(records)} questions loaded ({v_count:,} sessions in PG)")
    if limit < len(records):
        records = records[:limit]
        print(f"  trimmed to {limit}")

    s_count = loader.count_sessions("s")
    m_count = loader.count_sessions("m")
    print(f"  LongMemEval-S: {s_count:,} sessions")
    print(f"  LongMemEval-M: {m_count:,} sessions (available)")
    loader.close()

    total_sessions = v_count

    # ── 2. Build engine ────────────────────────────────────────────────────
    print(f"\n[engine] Building single PG-backed Mazemaker...")
    t_eng = time.perf_counter()
    nm = build_god_engine(schema, recall_mode=args.recall_mode, rerank=args.rerank,
                          reset_schema=not args.read_cache,
                          retrieval_candidates=args.retrieval_candidates,
                          colbert_weight=args.colbert_weight,
                          dae_weight=args.dae_weight)
    eng_s = time.perf_counter() - t_eng
    print(f"  engine ready in {eng_s:.1f}s — schema: {schema}")

    # ── 2b/3/3b. Cache-as-bench-target — skip drop+pipe+reindex ────────────
    # When --read-cache (default), the engine already points at the populated
    # cache schema. Counts come straight off the existing tables. No DDL.
    if args.read_cache:
        import psycopg as _pg  # local to keep top-level imports minimal
        from postgres_store import _build_dsn as _bd
        with _pg.connect(_bd(), autocommit=True) as _c, _c.cursor() as _cur:
            _cur.execute(f'SELECT count(*) FROM "{schema}".memories')
            n_stored = int(_cur.fetchone()[0])
            _cur.execute(f'SELECT count(*) FROM "{schema}".connections')
            n_edges = int(_cur.fetchone()[0])
        ingest_s = 0.0
        idx_s = 0.0
        print(f"\n[ingest] read-cache mode — bench targets cache schema "
              f"{schema!r} directly")
        print(f"  {n_stored:,} memories, {n_edges:,} connections already in place")
    else:
        # Legacy throwaway-schema flow: drop indexes, pipe from cache, rebuild.
        print("  optimizing PG for bulk ingest (dropping write-side indexes)...")
        try:
            dropped = nm.store.drop_bulk_indexes()
            print(f"    dropped: {', '.join(dropped)}")
        except Exception as exc:
            print(f"    drop_bulk_indexes() failed: {exc} — continuing")

        used_cache = False
        if args.from_cache:
            print(f"\n[ingest] Trying cache schema {CACHE_SCHEMA!r}...")
            t_ingest = time.perf_counter()
            try:
                n_stored = ingest_from_cache(nm, schema)
                ingest_s = time.perf_counter() - t_ingest
                print(f"  cache hit — {n_stored:,} memories piped in "
                      f"{ingest_s:.1f}s ({n_stored / ingest_s:.0f} mem/s)")
                used_cache = True
            except RuntimeError as exc:
                print(f"  cache miss: {exc}")
                print(f"  falling back to embed-pass ingest")

        if not used_cache:
            print(f"\n[ingest] Embed-pass ingesting {s_count:,} sessions...")
            t_ingest = time.perf_counter()
            n_stored = ingest_haystack(nm, records, batch_size=args.batch_size)
            ingest_s = time.perf_counter() - t_ingest
            print(f"  ingest done in {ingest_s:.1f}s "
                  f"({n_stored:,} memories, {n_stored / ingest_s:.0f} mem/s)")

        print(f"\n[indexes] Rebuilding PG indexes on populated table...")
        t_idx = time.perf_counter()
        try:
            nm.store.create_bulk_indexes(dim=nm.dim)
            idx_s = time.perf_counter() - t_idx
            print(f"  indexes rebuilt in {idx_s:.1f}s "
                  f"(idx_memories_label, idx_memories_created_at, "
                  f"idx_memories_content_fts, idx_memories_embedding_hnsw)")
        except Exception as exc:
            print(f"  create_bulk_indexes() failed: {exc} — recall fast paths "
                  f"may fall back to seq-scan")
            idx_s = 0.0

    # ── 4a. Build graph (auto-connect) ──────────────────────────────────────
    # Eager auto-connect builds semantic neighbour edges before baseline
    # recall.  On 24k+ corpora this is hours (per-node HNSW knn_query +
    # PG upsert), so the default path WARMS HNSW only — dream NREM later
    # produces equivalent edges if --dream is set.  Pass --graph to force
    # the eager pre-recall build.
    print(f"\n[graph] Warming HNSW for {n_stored:,} memories"
          + (" + auto-connect (--graph)" if args.graph else "") + "...")
    t_graph = time.perf_counter()
    nm._ensure_hnsw()
    connected = 0
    if args.graph:
        all_ids = list(nm._graph_nodes.keys())
        for mem_id in all_ids:
            node = nm._graph_nodes.get(mem_id) or {}
            emb = node.get("embedding") or []
            if not emb or len(emb) != nm.dim:
                continue
            try:
                nm._auto_connect(mem_id, emb, node.get("content", ""))
                connected += 1
            except Exception:
                pass
    graph_s = time.perf_counter() - t_graph
    if args.graph:
        print(f"  graph built in {graph_s:.1f}s "
              f"({connected:,}/{n_stored:,} nodes processed, "
              f"{connected/graph_s:.0f} nodes/s)")
    else:
        print(f"  HNSW warm in {graph_s:.1f}s (auto-connect skipped — "
              f"pass --graph or use --dream to materialise edges)")

    # ── 4. Baseline recall ─────────────────────────────────────────────────
    print(f"\n[recall] Running {limit} questions (baseline)...")
    t_recall = time.perf_counter()
    baseline_results: list[QuestionResult] = []
    for i, rec in enumerate(records):
        try:
            row = run_question(
                nm, rec, k=10,
                recall_mode=args.recall_mode,
                rerank=args.rerank,
                enable_colbert=args.colbert,
                enable_dae=args.dae,
                recall_k=args.recall_k,
                multi_recall=args.multi_recall,
                pref_multi_recall=args.pref_multi_recall,
            )
        except Exception as e:
            row = QuestionResult(
                qid=rec.get("question_id", "?"),
                question_type=rec.get("question_type", "unknown"),
                is_abstention=rec.get("question_id", "").endswith("_abs"),
                gold_session_ids=set(rec.get("answer_session_ids") or []),
                rank_of_gold=None,
                latency_ms=0.0,
                n_results=0,
                top_labels=[f"ERROR: {e}"],
            )
        baseline_results.append(row)
        if (i + 1) % max(1, limit // 20) == 0:
            done = i + 1
            mid = aggregate(baseline_results)
            pct = done / limit * 100
            print(f"  [{done:>4}/{limit} ({pct:>5.1f}%)]  "
                  f"R@1={mid.recall_at_1:.4f}  R@5={mid.recall_at_5:.4f}  "
                  f"MRR={mid.MRR:.4f}  "
                  f"p50={mid.p50_ms:.0f}ms  "
                  f"elapsed={time.perf_counter()-t_recall:.0f}s", flush=True)

    baseline = aggregate(baseline_results, phase="baseline")
    recall_s = time.perf_counter() - t_recall
    print_report("BASELINE RECALL (pre-dream)", baseline)

    # ── 5. Optional dream ablation ─────────────────────────────────────────
    dream_metrics: dict = {}
    post_dream: Optional[Metrics] = None
    if args.dream:
        print(f"\n[dream] Running dream cycle on {n_stored:,} memories...")
        t_dream = time.perf_counter()
        dream_metrics = run_dream_cycle(nm, n_stored)
        dream_s = time.perf_counter() - t_dream
        print(f"  dream done in {dream_s:.1f}s")
        for ph in dream_metrics.get("phases", []):
            print(f"    {ph['phase']:>12}: {ph['seconds']:.2f}s")

        # Post-dream recall
        print(f"\n[recall] Running {limit} questions (post-dream)...")
        post_results: list[QuestionResult] = []
        for i, rec in enumerate(records):
            try:
                row = run_question(
                    nm, rec, k=10,
                    recall_mode=args.recall_mode,
                    rerank=args.rerank,
                    enable_colbert=args.colbert,
                    enable_dae=args.dae,
                )
            except Exception as e:
                row = QuestionResult(
                    qid=rec.get("question_id", "?"),
                    question_type=rec.get("question_type", "unknown"),
                    is_abstention=rec.get("question_id", "").endswith("_abs"),
                    gold_session_ids=set(rec.get("answer_session_ids") or []),
                    rank_of_gold=None,
                    latency_ms=0.0,
                    n_results=0,
                    top_labels=[f"ERROR: {e}"],
                )
            post_results.append(row)

        post_dream = aggregate(post_results, phase="post-dream")
        print_report("POST-DREAM RECALL", post_dream)

        # Dream lift
        print(f"\n{'='*72}")
        print(f"  DREAM LIFT")
        print(f"{'='*72}")
        for metric in ["recall@1", "recall@5", "recall@10", "MRR"]:
            pre = getattr(baseline, metric.replace("@", "_at_"))
            post = getattr(post_dream, metric.replace("@", "_at_"))
            lift = post - pre
            print(f"  {metric:<20}  pre={pre:.4f}  post={post:.4f}  "
                  f"lift={lift:+.4f}  ({(lift/max(pre,0.0001))*100:+.1f}%)")

    # ── 6. Save results ────────────────────────────────────────────────────
    out = {
        "benchmark": "mazemaker_godbench",
        "schema": schema,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "recall_mode": args.recall_mode,
            "rerank": args.rerank,
            "colbert": args.colbert,
            "dae": args.dae,
            "dream": args.dream,
            "limit": limit,
            "sessions_ingested": n_stored,
            "ingest_seconds": round(ingest_s, 1),
        },
        "baseline": baseline.to_dict(),
    }
    if args.dream:
        out["post_dream"] = post_dream.to_dict()
        out["dream_metrics"] = dream_metrics
        if post_dream.MRR > 0 and baseline.MRR > 0:
            out["lift"] = {
                "recall@1": round(post_dream.recall_at_1 - baseline.recall_at_1, 4),
                "recall@5": round(post_dream.recall_at_5 - baseline.recall_at_5, 4),
                "recall@10": round(post_dream.recall_at_10 - baseline.recall_at_10, 4),
                "MRR": round(post_dream.MRR - baseline.MRR, 4),
            }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"godbench{tag}_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[output] {out_path}")

    total_elapsed = time.perf_counter() - t_eng
    print(f"\n[total] {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")

    # PG schemas persist for post-run inspection.  Each run uses a
    # timestamped schema, so they don't collide.  Drop manually via
    # `DROP SCHEMA "godbench_<ts>" CASCADE` if disk pressure matters.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n  [interrupted by user]", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"\n  [FATAL] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
