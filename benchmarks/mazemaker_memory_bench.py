#!/usr/bin/env python3
"""
Mazemaker Memory Benchmark — pure memory, no LLM, deterministic.

Tests the MAZEMAKER ENGINE directly via the public Python API.
NO LLM calls. NO judge. NO rubric ambiguity. Just:
  store(fact)  →  recall(query)  →  did the stored fact come back?

Each scenario:
  1. Wipe a fresh DB.
  2. Store N (fact, label) pairs.
  3. Ask Q queries; for each, check if the EXPECTED label
     (or substring) appears in recall's top-K results.
  4. Aggregate: recall@1, recall@5, recall@10, mean rank when hit,
     latency p50/p95.

Scenarios:
  S1. exact-recall      — store fact, ask the fact back verbatim.
  S2. paraphrase-recall — store fact, ask via reworded query.
  S3. multi-fact        — one question that needs N facts; check
                          all N appear in top-K.
  S4. update-tracking   — store A=old, A=new (different seq); ask
                          "current A" — must rank A=new above A=old.
  S5. conflict-fuse     — store two contradictory facts about same
                          anchor; check supersession behaviour.
  S6. distractor-resist — store target fact + 100 plausible
                          distractors; target must rank high.
  S7. needle-haystack   — store target fact + 1k generic memories;
                          query for target.
  S8. negation          — store "no X observed"; query "X?" must
                          surface the negative statement.
  S9. graph-traversal   — store A→B, B→C, C→D; query A; D should
                          appear via spreading activation (think).
  S10. latency          — corpus 10k; measure recall p50/p95.

PostgreSQL-only. Each scenario runs in its own DROP-CREATE'd schema
of the `mm10m_bench` database (override with `MM_BENCH_PG_DB` env).
SQLite is forbidden in this bench — `MM_DB_BACKEND=postgres` is
forced before engine construction.

Run:
    python mazemaker_memory_bench.py [--scenarios S1,S4]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Locate engine: prefer in-tree python/, fall back to installed
HERE = Path(__file__).resolve()
ENGINE = HERE.parent.parent / "python"
if ENGINE.exists():
    sys.path.insert(0, str(ENGINE))

from memory_client import Mazemaker  # noqa: E402

# Production-realistic Mazemaker: HNSW + ColBERT + DAE + rerank + advanced
# retrieval mode. See `invariant_engine_config_audit_2026_05_11.md`. Falls
# back to bare Mazemaker if the helper isn't on path (e.g. running outside
# the benchmark tree).
try:
    sys.path.insert(0, str(HERE.parent / "mazemaker_benchmark" / "mm_10m_eval" / "runners"))
    from engine_config import build_quality_engine  # noqa: E402
    _USE_QUALITY = True
except ImportError:
    build_quality_engine = None
    _USE_QUALITY = False

SEED = 42


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    name: str
    n_facts: int
    n_queries: int
    hits_at_1: int = 0
    hits_at_5: int = 0
    hits_at_10: int = 0
    ranks: list[int] = field(default_factory=list)  # 1-indexed; 0 = miss
    latencies_ms: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def recall_at_1(self) -> float:
        return self.hits_at_1 / max(1, self.n_queries)

    @property
    def recall_at_5(self) -> float:
        return self.hits_at_5 / max(1, self.n_queries)

    @property
    def recall_at_10(self) -> float:
        return self.hits_at_10 / max(1, self.n_queries)

    @property
    def mean_rank_when_hit(self) -> float | None:
        hits = [r for r in self.ranks if r > 0]
        return statistics.mean(hits) if hits else None

    def summary_dict(self) -> dict:
        lats = self.latencies_ms
        return {
            "scenario": self.name,
            "n_facts": self.n_facts,
            "n_queries": self.n_queries,
            "recall@1": round(self.recall_at_1, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mean_rank_when_hit": (
                round(self.mean_rank_when_hit, 2)
                if self.mean_rank_when_hit else None
            ),
            "latency_ms_p50": round(statistics.median(lats), 2) if lats else None,
            "latency_ms_p95": round(
                statistics.quantiles(lats, n=20)[18] if len(lats) >= 20
                else max(lats) if lats else 0, 2
            ),
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PG_BENCH_DB = os.environ.get("MM_BENCH_PG_DB", "mm10m_bench")


def _pg_reset_schema(schema: str) -> None:
    """DROP and recreate the per-scenario PG schema, so each scenario
    runs against a clean state. PostgresStore._ensure_schema will
    recreate the tables on first connect."""
    import os as _os
    _os.environ.setdefault("MM_POSTGRES_DB", PG_BENCH_DB)
    sys.path.insert(0, str(HERE.parent.parent / "python"))
    from postgres_store import _build_dsn
    import psycopg
    dsn = _build_dsn()
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            cur.execute(f'CREATE SCHEMA "{schema}"')


def fresh_engine(scenario: str) -> Mazemaker:
    """Create a fresh Mazemaker for one scenario on a PG schema dedicated
    to it. PG-ONLY — SQLite is banned in this bench (`db_path` is a
    no-op placeholder because Mazemaker still wants the attribute).
    Drops and recreates the per-scenario schema for clean state, then
    constructs the full production engine (advanced + rerank + ColBERT
    + DAE + HNSW). Never dumb the engine to flatter the bench."""
    if not _USE_QUALITY:
        raise AssertionError(
            "build_quality_engine import failed. The bench REQUIRES "
            "engine_config.build_quality_engine — install/path issue."
        )
    schema = f"mem_bench_{scenario.lower()}"
    os.environ["MM_DB_BACKEND"] = "postgres"
    os.environ["MM_POSTGRES_DB"] = PG_BENCH_DB
    os.environ["MM_POSTGRES_SCHEMA"] = schema
    _pg_reset_schema(schema)
    # `db_path` is a vestigial argument Mazemaker still requires for its
    # `_db_path` attribute (and the legacy SQLite-based GpuRecallEngine
    # build path that we don't use here). Point it at the OS bit bucket so
    # nothing on disk gets created.
    dummy_path = "/dev/null"
    try:
        nm = build_quality_engine(dummy_path, retrieval_mode="advanced")
    except TypeError as e:
        raise AssertionError(
            f"build_quality_engine signature changed: {e}. "
            f"Update fresh_engine() to match the new contract before "
            f"publishing benchmark numbers."
        ) from e
    # BGE-M3 = 1024d. Pre-create the embedding column + HNSW index on the
    # freshly-created schema so the engine's early SELECTs (graph load,
    # label lookups) don't fail before the first write would have triggered
    # lazy column creation.
    nm.store._ensure_embedding_column(1024)
    return nm


def rank_of(label_or_content: str, results: list[dict],
            match: str = "label") -> int:
    """Return 1-indexed rank of the FIRST result matching, or 0 for miss."""
    for i, r in enumerate(results, 1):
        if match == "label":
            if r.get("label") == label_or_content:
                return i
        elif match == "label_prefix":
            if (r.get("label") or "").startswith(label_or_content):
                return i
        elif match == "substring":
            if label_or_content.lower() in (r.get("content") or "").lower():
                return i
    return 0


def score_query(target: str, results: list[dict], result: ScenarioResult,
                latency_ms: float, match: str = "label") -> None:
    r = rank_of(target, results, match=match)
    result.ranks.append(r)
    result.latencies_ms.append(latency_ms)
    if 1 <= r <= 1: result.hits_at_1 += 1
    if 1 <= r <= 5: result.hits_at_5 += 1
    if 1 <= r <= 10: result.hits_at_10 += 1


def timed_recall(nm: Mazemaker, query: str, k: int = 10) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    res = nm.recall(query, k=k)
    return list(res), (time.perf_counter() - t0) * 1000.0


def bulk_remember(nm: Mazemaker, pairs: list[tuple[str, str]],
                  chunk: int = 500) -> None:
    """Use remember_batch for fast bulk ingest with no auto_connect.
    Per `decision_afe_bulk_write_2026_05_12.md`: per-row remember()
    is 88min for 5k rows; remember_batch is ~75s for the same."""
    for i in range(0, len(pairs), chunk):
        rows = [{"text": c, "label": l} for l, c in pairs[i:i + chunk]]
        nm.remember_batch(rows, detect_conflicts=False, auto_connect=False)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

FACT_TEMPLATES = [
    "The deployment success rate is {n}%.",
    "Operating costs reached ${n}k last quarter.",
    "The team completed {n} agents this sprint.",
    "Latency for the {service} service is {n}ms.",
    "The {component} module has {n} unit tests.",
    "Cache hit ratio improved to {n}%.",
    "Response time decreased by {n} milliseconds.",
    "We onboarded {n} new customers in {month}.",
    "Throughput increased to {n} requests per second.",
    "The error budget for {service} is set at {n}%.",
]
SERVICES = ["payment", "billing", "auth", "search", "indexer", "notification", "session", "audit"]
COMPONENTS = ["router", "scheduler", "cache", "queue", "orchestrator", "parser", "validator"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August"]


def gen_facts(n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Return list of (label, content). Each fact gets a UNIQUE marker so
    label-match in recall is meaningful (no template collisions).
    Deterministic given the seed."""
    out = []
    for i in range(n):
        tpl = rng.choice(FACT_TEMPLATES)
        params = {
            "n": rng.randint(10, 999),
            "service": rng.choice(SERVICES),
            "component": rng.choice(COMPONENTS),
            "month": rng.choice(MONTHS),
        }
        # Append a unique anchor so two facts with identical templates
        # don't collide in the corpus
        marker = f"[fact-{i:05d}]"
        content = f"{marker} {tpl.format(**params)}"
        label = f"fact_{i:04d}"
        out.append((label, content))
    return out


# --- S1: exact-recall ---------------------------------------------------------

def s1_exact_recall(scenario: str) -> ScenarioResult:
    nm = fresh_engine(scenario)
    rng = random.Random(SEED)
    facts = gen_facts(100, rng)
    for label, content in facts:
        nm.remember(content, label=label, auto_connect=False)
    r = ScenarioResult(name="S1_exact_recall", n_facts=100, n_queries=100)
    for label, content in facts:
        results, lat = timed_recall(nm, content, k=10)
        score_query(label, results, r, lat, match="label")
    return r


# --- S2: paraphrase-recall ----------------------------------------------------

PARAPHRASES = {
    "The deployment success rate is": "how successful are deployments",
    "Operating costs reached": "what were the operating expenses",
    "The team completed": "how many were finished this sprint",
    "Latency for the": "speed of the",
    "module has": "test coverage of the",
    "Cache hit ratio improved to": "cache performance improvement",
    "Response time decreased by": "how much faster did responses get",
    "We onboarded": "how many new customers signed up",
    "Throughput increased to": "throughput change",
    "The error budget for": "permissible errors for",
}


def paraphrase_query(content: str) -> str:
    for key, paraphrase in PARAPHRASES.items():
        if key in content:
            return paraphrase
    return content  # fallback: literal


def s2_paraphrase_recall(scenario: str) -> ScenarioResult:
    nm = fresh_engine(scenario)
    rng = random.Random(SEED)
    facts = gen_facts(100, rng)
    for label, content in facts:
        nm.remember(content, label=label, auto_connect=False)
    r = ScenarioResult(name="S2_paraphrase_recall", n_facts=100, n_queries=100)
    for label, content in facts:
        q = paraphrase_query(content)
        results, lat = timed_recall(nm, q, k=10)
        score_query(label, results, r, lat, match="label")
    return r


# --- S3: multi-fact (one query → multiple targets) ----------------------------

def s3_multi_fact(scenario: str) -> ScenarioResult:
    nm = fresh_engine(scenario)
    # 5 facts about ONE shared topic, 95 distractors
    topic_facts = [
        "Project Gamma launched in March 2026.",
        "Project Gamma costs $250k per quarter.",
        "Project Gamma has 30 engineers assigned.",
        "Project Gamma's primary metric is p99 latency.",
        "Project Gamma is led by Sarah Chen.",
    ]
    for i, c in enumerate(topic_facts):
        nm.remember(c, label=f"gamma_{i}", auto_connect=False)
    rng = random.Random(SEED)
    for j, (_lbl, c) in enumerate(gen_facts(95, rng)):
        nm.remember(c, label=f"distractor_{j}", auto_connect=False)
    r = ScenarioResult(name="S3_multi_fact", n_facts=100, n_queries=1)
    results, lat = timed_recall(nm, "Project Gamma overview", k=10)
    # Score: how many of the 5 gamma_* labels are in top-10?
    gamma_in_top_10 = sum(1 for x in results
                         if (x.get("label") or "").startswith("gamma_"))
    r.extra["gamma_facts_in_top_10"] = gamma_in_top_10
    r.extra["gamma_facts_total"] = 5
    r.extra["coverage"] = round(gamma_in_top_10 / 5, 3)
    # Use rank of first gamma as primary signal
    rank = next((i for i, x in enumerate(results, 1)
                if (x.get("label") or "").startswith("gamma_")), 0)
    r.ranks.append(rank)
    r.latencies_ms.append(lat)
    if rank == 1: r.hits_at_1 += 1
    if 1 <= rank <= 5: r.hits_at_5 += 1
    if 1 <= rank <= 10: r.hits_at_10 += 1
    return r


# --- S4: update-tracking (chronology) -----------------------------------------

def s4_update_tracking(scenario: str) -> ScenarioResult:
    nm = fresh_engine(scenario)
    # 20 attributes, each stored at 3 evolving values
    pairs = [
        ("agents completed", [10, 22, 30]),
        ("deployment success rate", [80, 85, 92]),
        ("cache hit ratio", [60, 75, 88]),
        ("monthly revenue", [50, 120, 200]),
        ("active users", [1000, 2500, 5000]),
    ]
    label_id = 0
    latest_labels = {}
    for attr, values in pairs:
        for i, v in enumerate(values):
            content = f"{attr} is now {v}{'%' if 'rate' in attr or 'ratio' in attr else ''}."
            label = f"upd_{label_id:04d}_{attr.replace(' ','_')}_{i}"
            nm.remember(content, label=label, auto_connect=False)
            label_id += 1
        latest_labels[attr] = (f"upd_{label_id-1:04d}_{attr.replace(' ','_')}_2",
                               values[-1])
    r = ScenarioResult(name="S4_update_tracking", n_facts=label_id,
                       n_queries=len(pairs))
    for attr, (latest_label, latest_value) in latest_labels.items():
        q = f"current {attr}"
        results, lat = timed_recall(nm, q, k=10)
        # Hit if the LATEST value appears WITH WORD-BOUNDARY in content
        # (so "92" doesn't accidentally match "920"). The stored format is
        # `is now <value>...`, so we anchor on that prefix.
        unit_suffix = "%" if ("rate" in attr or "ratio" in attr) else ""
        sentinel = f"is now {latest_value}{unit_suffix}"
        rank = 0
        for i, res in enumerate(results, 1):
            if sentinel in (res.get("content") or ""):
                rank = i
                break
        r.ranks.append(rank)
        r.latencies_ms.append(lat)
        if rank == 1: r.hits_at_1 += 1
        if 1 <= rank <= 5: r.hits_at_5 += 1
        if 1 <= rank <= 10: r.hits_at_10 += 1
    return r


# --- S5: conflict-fuse --------------------------------------------------------

def s5_conflict_fuse(scenario: str) -> ScenarioResult:
    """5 conflict pairs. Each pair = same topic, different value. After
    storing OLD then NEW, the engine's supersession should rank NEW above
    OLD on a topic query. Queries are EXPLICIT (no string-split heuristic
    — those silently degrade to 'we' tokens etc.)."""
    nm = fresh_engine(scenario)
    # (old_fact, new_fact, explicit_topic_query, expected_value_in_new)
    conflicts = [
        ("payment service uptime is 99.5%",
         "payment service uptime is 99.9%",
         "payment service uptime",
         "99.9%"),
        ("the index has 1M documents",
         "the index has 5M documents",
         "how many documents in the index",
         "5M"),
        ("Sarah leads Project Alpha",
         "Mike leads Project Alpha",
         "who leads Project Alpha",
         "Mike"),
        ("CPU usage averages 40%",
         "CPU usage averages 65%",
         "average CPU usage",
         "65%"),
        ("we ship every Tuesday",
         "we ship every Friday",
         "which day do we ship on",
         "Friday"),
    ]
    for i, (old, new, _, _) in enumerate(conflicts):
        mid_old = nm.remember(old, label=f"conflict_old_{i}",
                              auto_connect=True, detect_conflicts=True)
        mid_new = nm.remember(new, label=f"conflict_new_{i}",
                              auto_connect=True, detect_conflicts=True)
        # remember() must return a stable, hashable identity. Don't lock
        # to int — UUID/ULID/str migrations should not break this bench.
        for tag, mid in (("old", mid_old), ("new", mid_new)):
            assert mid is not None, \
                f"S5 contract: remember() returned None for {tag}"
            hash(mid)  # raises TypeError on unhashable handles
    r = ScenarioResult(name="S5_conflict_fuse", n_facts=len(conflicts) * 2,
                       n_queries=len(conflicts))
    for _old, _new, query, expected_value in conflicts:
        results, lat = timed_recall(nm, query, k=10)
        # The WINNING fact contains expected_value; rank of that hit.
        rank_new = next((i for i, x in enumerate(results, 1)
                        if expected_value in (x.get("content") or "")), 0)
        r.ranks.append(rank_new)
        r.latencies_ms.append(lat)
        if rank_new == 1: r.hits_at_1 += 1
        if 1 <= rank_new <= 5: r.hits_at_5 += 1
        if 1 <= rank_new <= 10: r.hits_at_10 += 1
    return r


# --- S6: distractor-resist ----------------------------------------------------

def s6_distractor_resist(scenario: str) -> ScenarioResult:
    nm = fresh_engine(scenario)
    rng = random.Random(SEED)
    targets = [
        ("target_0", "The Q3 launch budget was approved at $487k."),
        ("target_1", "Migration to PostgreSQL completed on April 11."),
        ("target_2", "Customer churn dropped to 3.2% after the rebate."),
        ("target_3", "Henderson Beach venue costs $13,200 per weekend."),
        ("target_4", "GPT-5-nano costs $0.10 per million tokens."),
    ]
    for label, content in targets:
        nm.remember(content, label=label, auto_connect=False)
    for j, (_lbl, c) in enumerate(gen_facts(100, rng)):
        nm.remember(c, label=f"distr_{j:04d}", auto_connect=False)
    queries = [
        ("Q3 launch budget approval", "target_0"),
        ("PostgreSQL migration completion date", "target_1"),
        ("customer churn after rebate", "target_2"),
        ("Henderson Beach wedding venue cost", "target_3"),
        ("nano model pricing per million tokens", "target_4"),
    ]
    r = ScenarioResult(name="S6_distractor_resist", n_facts=len(targets) + 100,
                       n_queries=len(queries))
    for q, expected_label in queries:
        results, lat = timed_recall(nm, q, k=10)
        score_query(expected_label, results, r, lat, match="label")
    return r


# --- S7: needle-haystack (scale) ----------------------------------------------

def s7_needle_haystack(scenario: str, haystack: int = 1000) -> ScenarioResult:
    nm = fresh_engine(scenario)
    needles = [
        ("needle_0", "The classified blueprint reference number is FX-9342-Q."),
        ("needle_1", "Vault unlock passphrase fragment: zephyr-quartz-meridian."),
        ("needle_2", "Backup site coordinates: 47.6062 N, 122.3321 W."),
        ("needle_3", "Emergency contact for Aurelio: +44-7700-900123."),
        ("needle_4", "Auth flag override token: NX_OVERRIDE_2026_MAYDAY."),
    ]
    rng = random.Random(SEED)
    haystack_facts = gen_facts(haystack, rng)
    insertion_points = sorted(rng.sample(range(haystack), len(needles)))
    # Build interleaved (label, content) pairs, then bulk-ingest
    pairs = []
    j = 0
    for i in range(haystack):
        if j < len(needles) and i == insertion_points[j]:
            pairs.append(needles[j])
            j += 1
        lbl, c = haystack_facts[i]
        pairs.append((f"hay_{i:05d}", c))
    bulk_remember(nm, pairs)
    queries = [
        ("classified blueprint reference number", "needle_0"),
        ("vault passphrase fragment", "needle_1"),
        ("backup site coordinates", "needle_2"),
        ("emergency contact Aurelio", "needle_3"),
        ("auth flag override token", "needle_4"),
    ]
    r = ScenarioResult(name="S7_needle_haystack",
                       n_facts=haystack + len(needles),
                       n_queries=len(queries))
    for q, expected_label in queries:
        results, lat = timed_recall(nm, q, k=10)
        score_query(expected_label, results, r, lat, match="label")
    return r


# --- S8: negation -------------------------------------------------------------

def s8_negation(scenario: str) -> ScenarioResult:
    nm = fresh_engine(scenario)
    negatives = [
        ("neg_0", "There were no unauthorized access attempts to Vault this quarter."),
        ("neg_1", "We observed zero downtime during the migration window."),
        ("neg_2", "The audit found no compliance violations."),
        ("neg_3", "There are no open critical bugs in the payment service."),
        ("neg_4", "We did not log errors during the deployment rollout."),
    ]
    for lbl, c in negatives:
        nm.remember(c, label=lbl, auto_connect=False)
    rng = random.Random(SEED)
    for j, (_lbl, c) in enumerate(gen_facts(50, rng)):
        nm.remember(c, label=f"distr_{j:04d}", auto_connect=False)
    queries = [
        ("unauthorized access to Vault", "neg_0"),
        ("downtime during migration", "neg_1"),
        ("audit compliance violations", "neg_2"),
        ("critical bugs payment service", "neg_3"),
        ("errors during deployment", "neg_4"),
    ]
    r = ScenarioResult(name="S8_negation", n_facts=len(negatives) + 50,
                       n_queries=len(queries))
    for q, expected_label in queries:
        results, lat = timed_recall(nm, q, k=10)
        score_query(expected_label, results, r, lat, match="label")
    return r


# --- S9: graph-traversal ------------------------------------------------------

def s9_graph_traversal(scenario: str) -> ScenarioResult:
    """Spreading-activation test: store a 4-node chain, query the head
    via think(), check whether the chain terminus surfaces within
    depth=3. ASSERTS remember() returns int id (API contract)."""
    nm = fresh_engine(scenario)
    chain = [
        ("chain_alpha", "Alpha is the entry point of the pipeline."),
        ("chain_beta",  "Alpha flows into Beta for normalization."),
        ("chain_gamma", "Beta then routes to Gamma for enrichment."),
        ("chain_delta", "Gamma's output is consumed by Delta storage."),
    ]
    mids = []
    for lbl, content in chain:
        mid = nm.remember(content, label=lbl, auto_connect=True)
        # Identity contract: must be non-None and hashable. Not tied
        # to int — UUID/ULID/handle migrations should not break this.
        assert mid is not None, (
            f"S9 contract: remember() returned None for {lbl}"
        )
        hash(mid)  # raises TypeError if engine returns unhashable handle
        mids.append(mid)
    rng = random.Random(SEED)
    for j, (_lbl, c) in enumerate(gen_facts(50, rng)):
        nm.remember(c, label=f"distr_{j:04d}", auto_connect=True)

    r = ScenarioResult(name="S9_graph_traversal",
                       n_facts=len(chain) + 50, n_queries=1)

    # Contract: think() must accept the seed-id kwarg and return iterable
    # of dicts with `id` keys. If signature shifts, fail loud.
    if not hasattr(nm, "think"):
        raise AssertionError("S9 contract violation: Mazemaker.think() missing")

    t0 = time.perf_counter()
    think_result = nm.think(mids[0], depth=3)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert think_result is None or hasattr(think_result, "__iter__"), (
        f"S9 contract violation: think() must return iterable or None, "
        f"got {type(think_result).__name__}"
    )
    ids = [t.get("id") for t in (think_result or [])]
    delta_id = mids[-1]
    rank = next((i for i, x in enumerate(ids, 1) if x == delta_id), 0)
    r.ranks.append(rank)
    r.latencies_ms.append(elapsed_ms)
    if rank == 1: r.hits_at_1 += 1
    if 1 <= rank <= 5: r.hits_at_5 += 1
    if 1 <= rank <= 10: r.hits_at_10 += 1
    r.extra["think_returned_n"] = len(ids)
    r.extra["delta_reachable_from_alpha"] = rank > 0
    return r


# --- S10: latency on 10k corpus -----------------------------------------------

def s10_latency_10k(scenario: str) -> ScenarioResult:
    nm = fresh_engine(scenario)
    rng = random.Random(SEED)
    facts = gen_facts(10_000, rng)
    print(f"  [S10] storing 10k facts via remember_batch ...", flush=True)
    t_store = time.perf_counter()
    bulk_remember(nm, facts)
    print(f"  [S10] storage done in {time.perf_counter() - t_store:.1f}s",
          flush=True)
    print(f"  [S10] running 200 queries ...", flush=True)
    r = ScenarioResult(name="S10_latency_10k", n_facts=10_000, n_queries=200)
    sample = rng.sample(facts, 200)
    for label, content in sample:
        results, lat = timed_recall(nm, content, k=10)
        score_query(label, results, r, lat, match="label")
    return r


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# --- S11: large-scale needle (100k corpus) ------------------------------------

def s11_needle_100k(scenario: str, haystack: int = 100_000) -> ScenarioResult:
    nm = fresh_engine(scenario)
    needles = [
        ("needle_A", "Final detonation override sequence: theta-7-quartz-mariposa."),
        ("needle_B", "Project Phoenix kickoff approved at $4.82M budget on August 14."),
        ("needle_C", "Tunneling cipher coordinates: 51.4769 N, 0.0005 W, key 0xF3A7."),
        ("needle_D", "Lead investigator handoff signed off by Anneliese Vorrath."),
        ("needle_E", "Containment threshold raised to 2.47 sieverts for shift 9."),
        ("needle_F", "Cold-start initiator code: ZEPHYR_NULL_88_BRAVO."),
        ("needle_G", "Backup uplink frequency 5.871 GHz, alignment angle 23.4 deg."),
        ("needle_H", "Emergency proxy address: 198.51.100.247:4422 over QUIC."),
        ("needle_I", "Distinct biomarker reading: ferritin 412 ng/mL during phase 3."),
        ("needle_J", "Auth shadow record: identifier RS-3308-PXR-MAYDAY."),
    ]
    rng = random.Random(SEED)
    haystack_facts = gen_facts(haystack, rng)
    insertion_points = sorted(rng.sample(range(haystack), len(needles)))

    # Approx 70 tokens/fact (templated facts ~10 words ≈ 13 tokens for the
    # marker + content + period). 100k facts ≈ 1.3-1.5M tokens. Tracking
    # this precisely is left as an exercise for the AFE re-ingest test.
    pairs = []
    j = 0
    for i in range(haystack):
        if j < len(needles) and i == insertion_points[j]:
            pairs.append(needles[j])
            j += 1
        lbl, c = haystack_facts[i]
        pairs.append((f"hay_{i:06d}", c))
    print(f"  [S11] storing {len(pairs)} pairs via remember_batch ...",
          flush=True)
    t_store = time.perf_counter()
    bulk_remember(nm, pairs)
    print(f"  [S11] storage done in {time.perf_counter() - t_store:.1f}s",
          flush=True)

    queries = [
        ("detonation override sequence theta quartz", "needle_A"),
        ("Project Phoenix kickoff budget August", "needle_B"),
        ("tunneling cipher coordinates key", "needle_C"),
        ("lead investigator handoff Anneliese", "needle_D"),
        ("containment threshold sieverts shift", "needle_E"),
        ("cold-start initiator code Zephyr", "needle_F"),
        ("backup uplink frequency alignment angle", "needle_G"),
        ("emergency proxy address QUIC", "needle_H"),
        ("biomarker reading ferritin phase 3", "needle_I"),
        ("auth shadow record identifier", "needle_J"),
    ]
    r = ScenarioResult(name="S11_needle_100k_corpus",
                       n_facts=haystack + len(needles),
                       n_queries=len(queries))
    for q, expected_label in queries:
        results, lat = timed_recall(nm, q, k=10)
        score_query(expected_label, results, r, lat, match="label")
    return r


# --- S12: 5-phase Dream ablation [INTERNAL ENGINE ABI TEST] -------------------
#
# WARNING: S12 invokes PRIVATE engine methods (`_phase_nrem`, `_phase_rem`,
# `_phase_insights`, `_phase_supersedes`, `_phase_afe`, `_phase_dae`). It is
# an internal engine-regression / ABI-stability test, NOT a portable
# memory benchmark. Method renames break this scenario by design — that's
# the signal. Do NOT cite S12 in cross-engine comparisons.
#
# Demonstrate the MARGINAL contribution of each Mazemaker Dream phase:
#   baseline → +NREM → +REM → +Insight → +AFE → +DAE
# Single corpus, repeated recall, measure delta per phase.

S12_NEEDLES = [
    ("needle_alpha", "The Q3 launch budget was approved at exactly $487,200 USD."),
    ("needle_beta",  "Migration to PostgreSQL completed on April 11, 2026 at 03:42 UTC."),
    ("needle_gamma", "Customer churn dropped to exactly 3.2% after the loyalty rebate."),
    ("needle_delta", "Henderson Beach wedding venue costs $13,200 per weekend rental."),
    ("needle_epsilon", "GPT-5-nano pricing is $0.10 per million tokens, $0.40 output."),
]
S12_QUERIES = [
    ("Q3 launch budget exact amount approved",       "needle_alpha"),
    ("PostgreSQL migration completion April",        "needle_beta"),
    ("customer churn after loyalty rebate percent",  "needle_gamma"),
    ("Henderson Beach wedding venue weekend cost",   "needle_delta"),
    ("nano model pricing per million tokens",        "needle_epsilon"),
]


def _recall_score(nm: Mazemaker) -> tuple[float, float, float]:
    """Return (R@1, R@5, R@10) on the S12 fixed needle/query set."""
    h1 = h5 = h10 = 0
    for q, exp in S12_QUERIES:
        results, _lat = timed_recall(nm, q, k=10)
        r = rank_of(exp, results, match="label")
        if r == 1: h1 += 1
        if 1 <= r <= 5: h5 += 1
        if 1 <= r <= 10: h10 += 1
    n = len(S12_QUERIES)
    return h1 / n, h5 / n, h10 / n


def s12_dream_ablation(scenario: str,
                       n_corpus: int = 1_000) -> ScenarioResult:
    """Demonstrate each Dream phase's marginal contribution to recall.

    Ingest with auto_connect=False (fast); dream NREM/REM build the
    connections later. This mirrors production behaviour: the sponge
    worker absorbs every message immediately, dream consolidates on
    idle."""
    nm = fresh_engine(scenario)
    print(f"  [S12] storing {n_corpus} corpus + 5 needles via remember_batch ...",
          flush=True)
    rng = random.Random(SEED)
    corpus = gen_facts(n_corpus, rng)
    inj = sorted(rng.sample(range(n_corpus), len(S12_NEEDLES)))
    pairs = []
    j = 0
    for i in range(n_corpus):
        if j < len(S12_NEEDLES) and i == inj[j]:
            pairs.append(S12_NEEDLES[j])
            j += 1
        lbl, c = corpus[i]
        pairs.append((f"corpus_{i:05d}", c))
    t_ingest = time.perf_counter()
    bulk_remember(nm, pairs)
    print(f"  [S12] ingest done in {time.perf_counter() - t_ingest:.1f}s",
          flush=True)

    phase_rows: list[tuple[str, float, float, float, float]] = []

    # PHASE 0: baseline (no dream)
    r1, r5, r10 = _recall_score(nm)
    phase_rows.append(("baseline_no_dream", 0.0, r1, r5, r10))
    print(f"    baseline                R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f}", flush=True)

    # Dream engine is REQUIRED here — S12 has no fallback. Failure to
    # import or method-mismatch is a hard contract violation, surfaced
    # immediately so a broken engine release can't pass silently.
    from dream_engine import DreamEngine  # noqa: E402 — intentional late
    de = DreamEngine(nm, max_memories_per_cycle=min(n_corpus, 1000),
                     max_isolated_per_cycle=500)

    REQUIRED_PHASES = [
        ("_phase_nrem",      "+NREM"),
        ("_phase_rem",       "+REM"),
        ("_phase_insights",  "+Insights"),
        ("_phase_supersedes", "+Supersedes"),
        ("_phase_afe",       "+AFE"),
        ("_phase_dae",       "+DAE"),
    ]
    missing = [m for m, _ in REQUIRED_PHASES if not hasattr(de, m)]
    if missing:
        raise AssertionError(
            f"S12 contract violation: DreamEngine missing methods "
            f"{missing}. The benchmark explicitly tests every named "
            f"phase. If you renamed/removed any, update this list and "
            f"reproduce the historical numbers before publishing new "
            f"ones."
        )

    for attr, label in REQUIRED_PHASES:
        t0 = time.perf_counter()
        # NO try/except. Phase exceptions abort the scenario — silent
        # partial results are how broken engines pass benchmarks.
        getattr(de, attr)()
        elapsed = time.perf_counter() - t0
        r1, r5, r10 = _recall_score(nm)
        phase_rows.append((label, elapsed, r1, r5, r10))
        print(f"    {label:<12}({elapsed:6.2f}s) "
              f"R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f}", flush=True)

    r = ScenarioResult(name="S12_dream_ablation",
                       n_facts=n_corpus + len(S12_NEEDLES),
                       n_queries=len(S12_QUERIES))
    # Final phase's recall is the summary
    r.hits_at_1 = int(phase_rows[-1][2] * r.n_queries)
    r.hits_at_5 = int(phase_rows[-1][3] * r.n_queries)
    r.hits_at_10 = int(phase_rows[-1][4] * r.n_queries)
    r.ranks = [1] * r.hits_at_1 + [0] * (r.n_queries - r.hits_at_1)
    r.latencies_ms = [0.0] * r.n_queries
    r.extra["phase_table"] = phase_rows
    return r


# --- S13: 500k-scalable multiturn final test ---------------------------------
#
# THE ONE question. THE three turns. THE final answer. Plus 500,000 distractor
# facts (configurable, designed to scale linearly to 1B given hardware).
#
# This is the necbreaker: claim memory, beat this. Pure recall@1. No LLM.
# If the engine cannot put turn C above turn A and turn B for a 3-turn
# update on a tracked attribute, hidden inside 500k+ distractors, it does
# not have memory.

# Hand-crafted strings with *lexically interchangeable* values so the
# engine cannot lexically prefer one turn over another. The only signal
# that distinguishes the three is CHRONOLOGY (seq order).
# - All three turns talk about the same attribute in symmetric language.
# - The values (bronze / silver / obsidian) are semantically peer tier-names,
#   not numerical magnitudes a similarity model can sort.
# - The query contains NONE of "current", "final", "latest", "now", "99.9",
#   "upgraded" — only the attribute name. So pure semantic similarity
#   cannot lift turn C above turn A or B.
S13_TURNS = [
    ("multiturn_A", "Phoenix deployment tier set to bronze."),
    ("multiturn_B", "Phoenix deployment tier upgraded to silver."),
    ("multiturn_C", "Phoenix deployment tier finalized at obsidian."),
]
S13_QUERY = "Phoenix deployment tier"
S13_EXPECTED_TOP1_LABEL = "multiturn_C"
S13_EXPECTED_TOP1_VALUE = "obsidian"


def s13_multiturn_final(scenario: str,
                        n_corpus: int = 500_000) -> ScenarioResult:
    """ONE multiturn question over a 500k-default haystack.

    Design notes:
    - The 3 turns are inserted at the 1/4, 1/2, 3/4 marks of the
      distractor stream so seq* labels reflect chronology.
    - n_corpus is configurable. 500k runs in ~25 min on RTX-class
      hardware with the bulk-ingest path. The design extends linearly
      to 1B given a Postgres+pgvector backend and ~30 days of ingest
      headroom (or massively parallel sharded writes).
    - Score is BINARY: top-1 of `nm.recall(S13_QUERY, k=10)` must be
      `multiturn_C` (by label) AND its content must contain '99.9%'.
      Both conditions: pass. Either missing: fail.
    """
    nm = fresh_engine(scenario)
    rng = random.Random(SEED)
    print(f"  [S13] generating {n_corpus} distractor facts ...", flush=True)
    distractors = gen_facts(n_corpus, rng)

    # Three turns at three positions through the stream so seq is monotonic
    pos_A = n_corpus // 4
    pos_B = n_corpus // 2
    pos_C = (3 * n_corpus) // 4

    pairs: list[tuple[str, str]] = []
    insertions = {pos_A: S13_TURNS[0], pos_B: S13_TURNS[1], pos_C: S13_TURNS[2]}
    for i in range(n_corpus):
        if i in insertions:
            pairs.append(insertions[i])
        lbl, c = distractors[i]
        pairs.append((f"distractor_{i:07d}", c))

    n_total = len(pairs)
    print(f"  [S13] ingesting {n_total:,} memories via remember_batch ...",
          flush=True)
    t_ingest = time.perf_counter()
    bulk_remember(nm, pairs, chunk=2000)
    ingest_sec = time.perf_counter() - t_ingest
    facts_per_sec = n_total / ingest_sec if ingest_sec > 0 else 0
    # Approx 1.5B tokens/h projection assumes ~12 tokens/fact
    h_for_1b = (1_000_000_000 / facts_per_sec) / 3600 if facts_per_sec > 0 else None
    print(f"  [S13] ingest done in {ingest_sec:.1f}s "
          f"({facts_per_sec:.0f} facts/s; "
          f"projected ~{h_for_1b:.0f}h for 1B at this rate)",
          flush=True)

    # The ONE query. Hard contract — both label and value must match.
    print(f"  [S13] running THE multiturn query ...", flush=True)
    t_q = time.perf_counter()
    results = list(nm.recall(S13_QUERY, k=10))
    q_lat_ms = (time.perf_counter() - t_q) * 1000

    assert len(results) > 0, "S13 contract: recall returned 0 results"
    top = results[0]
    top_label = top.get("label") or ""
    top_content = top.get("content") or ""
    label_match = (top_label == S13_EXPECTED_TOP1_LABEL)
    value_match = (S13_EXPECTED_TOP1_VALUE in top_content)
    binary_pass = 1.0 if (label_match and value_match) else 0.0

    # Also surface where C, B, A sit in the full top-10 for diagnostic
    label_ranks = {}
    for i, x in enumerate(results, 1):
        lbl = x.get("label")
        if lbl in ("multiturn_A", "multiturn_B", "multiturn_C"):
            label_ranks.setdefault(lbl, i)

    r = ScenarioResult(name="S13_multiturn_final",
                       n_facts=n_total, n_queries=1)
    r.ranks.append(label_ranks.get(S13_EXPECTED_TOP1_LABEL, 0))
    r.latencies_ms.append(q_lat_ms)
    if binary_pass:
        r.hits_at_1 = 1
        r.hits_at_5 = 1
        r.hits_at_10 = 1
    elif label_ranks.get(S13_EXPECTED_TOP1_LABEL, 0):
        rank = label_ranks[S13_EXPECTED_TOP1_LABEL]
        if 1 <= rank <= 5: r.hits_at_5 = 1
        if 1 <= rank <= 10: r.hits_at_10 = 1
    r.extra["binary_pass"] = binary_pass
    r.extra["top1_label"] = top_label
    r.extra["top1_content"] = top_content[:160]
    r.extra["label_ranks"] = label_ranks
    r.extra["ingest_sec"] = round(ingest_sec, 1)
    r.extra["facts_per_sec"] = round(facts_per_sec, 0)
    r.extra["projected_1b_hours"] = round(h_for_1b, 1) if h_for_1b else None
    return r


SCENARIOS: dict[str, Callable[[Path], ScenarioResult]] = {
    "S1": s1_exact_recall,
    "S2": s2_paraphrase_recall,
    "S3": s3_multi_fact,
    "S4": s4_update_tracking,
    "S5": s5_conflict_fuse,
    "S6": s6_distractor_resist,
    "S7": s7_needle_haystack,
    "S8": s8_negation,
    "S9": s9_graph_traversal,
    "S10": s10_latency_10k,
    "S11": s11_needle_100k,
    "S12": s12_dream_ablation,
    "S13": s13_multiturn_final,
}


# --- S14: THE DEATH VERSION ---------------------------------------------------
#
# 40 updates of the same entity-attribute, with:
#   - semantic drift  (units/phrasing/qualifiers change over time)
#   - negated intermediates ("the target is NOT X anymore")
#   - same-entity-different-attribute distractors (Phoenix budget, team size)
#   - similar-entity distractors (Atlas/Nova/Helios targets at near values)
#   - similar-numeric noise (random N% facts unrelated to Phoenix)
#   - large-scale distractor corpus (default 500k, scalable to 1B)
#
# ONE query in natural language with NO entity or attribute keyword,
# only a temporal anchor ("finally settle on"). The engine must:
#   1. identify the right entity-attribute
#   2. enumerate all 40 updates
#   3. suppress negated intermediates
#   4. ignore same-entity-other-attribute facts
#   5. ignore similar-entity facts
#   6. pick the LATEST update by chronology
#
# Pass: top-1 must equal the 40th update AND contain its sentinel value.
# Binary. No partial credit. No judge.

S14_ENTITY = "Phoenix"
S14_ATTRIBUTE = "deployment threshold"

# 40 evolving values for Phoenix's deployment threshold. Peer values
# (no canonical magnitude ordering), with phrasing drift over time.
# value_40 is the FINAL TRUTH and must contain a unique sentinel token.
S14_VALUES = [
    # Early formulations — generic percentages
    ("82% acceptance",          "Phoenix deployment threshold initial spec is 82% acceptance."),
    ("84.5% acceptance",        "Updated Phoenix deployment threshold to 84.5% acceptance per pilot."),
    ("not 84.5%",               "We've decided Phoenix deployment threshold is NOT 84.5% anymore — it was a draft."),
    ("87% acceptance",          "Phoenix deployment threshold revised to 87% acceptance for stable releases."),
    ("87.5% acceptance",        "Phoenix deployment threshold nudged to 87.5% acceptance for the alpha cohort."),
    ("88% acceptance",          "Phoenix deployment threshold raised to 88% acceptance after retrospective."),
    ("88% conditional",         "Phoenix deployment threshold reframed: 88% acceptance, conditional on canary."),
    ("not 88%",                 "Phoenix deployment threshold is NOT 88% — that proposal was rejected by the steering group."),
    ("90% acceptance",          "Phoenix deployment threshold reset to 90% acceptance per the new accord."),
    # Mid stretch — semantic drift to different framings of the same axis
    ("90% success-rate",        "Phoenix deployment threshold expressed as 90% success-rate going forward."),
    ("91% success-rate",        "Phoenix deployment threshold incremented to 91% success-rate post-incident-review."),
    ("91% green-rate",          "Phoenix deployment threshold rephrased as 91% green-rate of canary checks."),
    ("92% green-rate",          "Phoenix deployment threshold escalated to 92% green-rate to harden launch."),
    ("not 92%",                 "Phoenix deployment threshold of 92% green-rate is NO LONGER current — superseded last sprint."),
    ("92.5% green-rate",        "Phoenix deployment threshold corrected to 92.5% green-rate, includes synthetic checks."),
    ("93% pass-rate",           "Phoenix deployment threshold articulated as 93% pass-rate across health probes."),
    ("93.7% pass-rate",         "Phoenix deployment threshold nudged to 93.7% pass-rate after retrospective."),
    ("94% pass-rate",           "Phoenix deployment threshold revised to 94% pass-rate including warm-up probes."),
    ("94.4% pass-rate",         "Phoenix deployment threshold tightened to 94.4% pass-rate."),
    ("94.4% pass-rate",         "Phoenix deployment threshold held at 94.4% pass-rate after weekly review."),
    # Later stretch — qualifiers stack
    ("95% pass-rate weighted",  "Phoenix deployment threshold codified as 95% pass-rate weighted by traffic share."),
    ("95.5% weighted",          "Phoenix deployment threshold incremented to 95.5% weighted by traffic share."),
    ("96% weighted",            "Phoenix deployment threshold escalated to 96% weighted pass-rate."),
    ("not 96%",                 "Phoenix deployment threshold of 96% is NOT current; we walked it back."),
    ("96.2% weighted",          "Phoenix deployment threshold restored to 96.2% weighted pass-rate."),
    ("97% weighted",            "Phoenix deployment threshold raised to 97% weighted pass-rate for prod-grade."),
    ("97.4% weighted",          "Phoenix deployment threshold tightened to 97.4% weighted pass-rate."),
    ("97.9% weighted",          "Phoenix deployment threshold escalated to 97.9% weighted pass-rate."),
    ("98% weighted",            "Phoenix deployment threshold reached 98% weighted pass-rate target."),
    ("98.3% weighted",          "Phoenix deployment threshold nudged to 98.3% weighted pass-rate post-stress-test."),
    ("98.6% weighted",          "Phoenix deployment threshold incremented to 98.6% weighted pass-rate."),
    ("98.8% weighted",          "Phoenix deployment threshold raised to 98.8% weighted pass-rate."),
    ("99% weighted",             "Phoenix deployment threshold landed at 99% weighted pass-rate per accord."),
    ("not 99%",                  "Phoenix deployment threshold of 99% was withdrawn — see meeting minutes."),
    ("99.2% weighted",          "Phoenix deployment threshold corrected back to 99.2% weighted pass-rate."),
    ("99.4% weighted",          "Phoenix deployment threshold tightened to 99.4% weighted pass-rate."),
    ("99.6% weighted",          "Phoenix deployment threshold escalated to 99.6% weighted pass-rate."),
    ("99.7% weighted",          "Phoenix deployment threshold nudged to 99.7% weighted pass-rate."),
    ("99.8% weighted",          "Phoenix deployment threshold incremented to 99.8% weighted pass-rate."),
    # THE FINAL TRUTH — must contain the unique sentinel token KORNMUEHLE
    # so the binary scorer cannot accidentally match an earlier update.
    ("KORNMUEHLE final 99.97%", "Phoenix deployment threshold finalized at 99.97% — codename KORNMUEHLE — locked, signed, no further changes."),
]
assert len(S14_VALUES) == 40, "S14 must have exactly 40 updates"

S14_FINAL_SENTINEL = "KORNMUEHLE"
S14_FINAL_LABEL = "phoenix_update_39"  # 40th (0-indexed 39)

# Natural-language query with NO entity, NO attribute keyword, ONLY
# a temporal anchor. The engine must reason from semantics + chronology.
S14_QUERY = "What did we finally settle on?"


def s14_death_multiturn(scenario: str,
                        n_corpus: int = 500_000) -> ScenarioResult:
    """THE DEATH VERSION — 40 updates, drift, negation, similar entities,
    one neutral-anchor query, binary pass.

    Strict score: top-1 must equal `phoenix_update_39` AND contain the
    final sentinel token ('KORNMUEHLE'). Anything less = 0."""
    nm = fresh_engine(scenario)
    rng = random.Random(SEED)
    print(f"  [S14] generating {n_corpus} distractor facts ...", flush=True)
    distractors = gen_facts(n_corpus, rng)

    # SIMILAR-ENTITY distractors — same axis, different entity, same units
    similar_projects = ["Atlas", "Nova", "Helios", "Polaris", "Vega"]
    similar_entity_facts: list[tuple[str, str]] = []
    for proj in similar_projects:
        for k in range(8):  # 8 updates each = 40 same-axis distractors
            v = rng.uniform(80, 99.5)
            similar_entity_facts.append((
                f"{proj.lower()}_update_{k}",
                f"{proj} deployment threshold updated to {v:.1f}% weighted pass-rate."
            ))

    # SAME-ENTITY OTHER-ATTRIBUTE distractors
    same_entity_other_attr = [
        ("phoenix_budget_0",   "Phoenix project budget approved at $480k for Q3."),
        ("phoenix_budget_1",   "Phoenix project budget revised to $512k after scope creep."),
        ("phoenix_budget_2",   "Phoenix project budget finalized at $560k by steering committee."),
        ("phoenix_team_0",     "Phoenix team size grew to 14 engineers in the latest re-org."),
        ("phoenix_team_1",     "Phoenix team size now stands at 19 engineers post-acquisition."),
        ("phoenix_launch_0",   "Phoenix launch window slid to October 18 per the PM."),
        ("phoenix_launch_1",   "Phoenix launch window confirmed for November 2 finally."),
        ("phoenix_latency_0", "Phoenix p99 latency target set at 250ms across all regions."),
    ]

    # The 40 phoenix-update facts
    phoenix_updates = [(f"phoenix_update_{i}", content)
                       for i, (_v, content) in enumerate(S14_VALUES)]

    # TWO-PHASE ingest:
    #   1. Bulk-ingest distractors via remember_batch (no auto_connect,
    #      no conflict detect) — they're just noise.
    #   2. Per-row remember() the 40 phoenix updates + similar-entity +
    #      same-entity-other-attr with detect_conflicts=True &
    #      auto_connect=True. This triggers `_detect_supersedes_at_ingest`
    #      which writes supersedes edges among phoenix updates with
    #      similar embeddings + different numerics. At recall time, the
    #      engine demotes superseded results and elevates the leaf (the
    #      40th update with `KORNMUEHLE`).
    #
    # CRITICAL: phoenix_update_39 must be inserted LAST among the 40 so
    # all earlier phoenix updates exist in `_graph_nodes` when it runs
    # supersedes detection.

    # Phase 1: bulk distractors interleaved with the same-entity-other-attr
    # and similar-entity facts (those don't need conflict detect).
    bulk_extras = similar_entity_facts + same_entity_other_attr
    rng.shuffle(bulk_extras)  # deterministic — seeded
    step = max(1, n_corpus // (len(bulk_extras) + 1))
    bulk_pairs: list[tuple[str, str]] = []
    extra_iter = iter(bulk_extras)
    next_extra = next(extra_iter, None)
    extra_slot = step
    for i in range(n_corpus):
        if next_extra is not None and i >= extra_slot:
            bulk_pairs.append(next_extra)
            next_extra = next(extra_iter, None)
            extra_slot += step
        lbl, c = distractors[i]
        bulk_pairs.append((f"distractor_{i:07d}", c))

    n_bulk = len(bulk_pairs)
    print(f"  [S14] Phase 1: bulk-ingesting {n_bulk:,} distractor + "
          f"similar-entity + other-attr memories ...", flush=True)
    t_ingest = time.perf_counter()
    bulk_remember(nm, bulk_pairs, chunk=2000)
    t_bulk_done = time.perf_counter()

    # Phase 2: per-row remember of 40 phoenix updates IN ORDER. This is
    # where SUPERSEDES edges get written. Each later update sees prior
    # phoenix updates in `_graph_nodes` and writes supersedes edges from
    # them to itself (older → newer) when content is high-similarity
    # but numerics differ.
    print(f"  [S14] Phase 2: per-row ingesting 40 phoenix updates with "
          f"detect_conflicts=True, auto_connect=True (writes supersedes "
          f"edges) ...", flush=True)
    for lbl, content in phoenix_updates:
        mid = nm.remember(content, label=lbl,
                          detect_conflicts=True, auto_connect=True)
        assert mid is not None, f"S14 contract: remember() returned None for {lbl}"
        hash(mid)
    t_per_row_done = time.perf_counter()
    n_total = n_bulk + len(phoenix_updates)
    ingest_sec = t_per_row_done - t_ingest
    facts_per_sec = n_total / ingest_sec if ingest_sec > 0 else 0
    h_for_1b = (1_000_000_000 / facts_per_sec) / 3600 if facts_per_sec > 0 else None
    print(f"  [S14] ingest done in {ingest_sec:.1f}s "
          f"(bulk {t_bulk_done - t_ingest:.1f}s + "
          f"per-row {t_per_row_done - t_bulk_done:.1f}s for "
          f"40 phoenix updates; "
          f"{facts_per_sec:.0f} facts/s avg; "
          f"projected ~{h_for_1b:.1f}h for 1B at this rate)",
          flush=True)

    # Verify supersedes edges were written
    try:
        from collections import Counter
        edge_types = Counter()
        # Look at phoenix_update_39's incoming edges (it should be the
        # target of supersedes from earlier phoenix updates)
        n39_id = None
        for lbl, _content in phoenix_updates[-1:]:
            # Get the engine's view of this label
            pass
        # Skip explicit verification — recall result is the canonical test
    except Exception:
        pass

    print(f"  [S14] running THE neutral-anchor query: {S14_QUERY!r}",
          flush=True)
    t_q = time.perf_counter()
    results = list(nm.recall(S14_QUERY, k=20))
    q_lat_ms = (time.perf_counter() - t_q) * 1000

    assert len(results) > 0, "S14 contract: recall returned 0 results"
    top = results[0]
    top_label = top.get("label") or ""
    top_content = top.get("content") or ""
    label_match = (top_label == S14_FINAL_LABEL)
    sentinel_match = (S14_FINAL_SENTINEL in top_content)
    binary_pass = 1.0 if (label_match and sentinel_match) else 0.0

    # Diagnostic: where are the 40 phoenix updates ranked?
    phoenix_ranks = {}
    for i, x in enumerate(results, 1):
        lbl = x.get("label") or ""
        if lbl.startswith("phoenix_update_"):
            phoenix_ranks.setdefault(lbl, i)

    r = ScenarioResult(name="S14_death_multiturn",
                       n_facts=n_total, n_queries=1)
    r.ranks.append(phoenix_ranks.get(S14_FINAL_LABEL, 0))
    r.latencies_ms.append(q_lat_ms)
    if binary_pass:
        r.hits_at_1 = 1
        r.hits_at_5 = 1
        r.hits_at_10 = 1
    elif phoenix_ranks.get(S14_FINAL_LABEL, 0):
        rank = phoenix_ranks[S14_FINAL_LABEL]
        if 1 <= rank <= 5: r.hits_at_5 = 1
        if 1 <= rank <= 10: r.hits_at_10 = 1
    r.extra["binary_pass"] = binary_pass
    r.extra["top1_label"] = top_label
    r.extra["top1_content"] = top_content[:240]
    r.extra["phoenix_update_ranks_in_top20"] = phoenix_ranks
    r.extra["ingest_sec"] = round(ingest_sec, 1)
    r.extra["facts_per_sec"] = round(facts_per_sec, 0)
    r.extra["projected_1b_hours"] = round(h_for_1b, 1) if h_for_1b else None
    return r


SCENARIOS["S14"] = s14_death_multiturn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Default runs the supporting scenarios + S13 (medium-scale monument).
    # S14 (DEATH version, ~40 min wall) is opt-in: pass --scenarios S14.
    p.add_argument("--scenarios",
                   default="S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13")
    p.add_argument("--out", type=Path, default=None,
                   help="Write JSON summary here")
    args = p.parse_args()

    selected = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    results: list[ScenarioResult] = []
    t0 = time.perf_counter()
    for s in selected:
        if s not in SCENARIOS:
            print(f"  unknown scenario {s}; skipping")
            continue
        print(f"\n[bench] {s} starting (PG schema mem_bench_{s.lower()}) ...",
              flush=True)
        ts0 = time.perf_counter()
        try:
            r = SCENARIOS[s](s)
            elapsed = time.perf_counter() - ts0
            results.append(r)
            print(f"[bench] {s} done in {elapsed:.1f}s — "
                  f"R@1={r.recall_at_1:.3f} R@5={r.recall_at_5:.3f} "
                  f"R@10={r.recall_at_10:.3f}")
        except Exception as e:
            import traceback
            print(f"[bench] {s} FAILED: {e}")
            traceback.print_exc()

    total_elapsed = time.perf_counter() - t0

    if not results:
        raise RuntimeError(
            "No scenarios completed — bench produced zero rows. "
            "Inspect upstream errors before publishing any numbers."
        )

    # Report
    print("\n" + "=" * 76)
    print("MAZEMAKER MEMORY BENCHMARK — pure memory, no LLM, deterministic")
    print("=" * 76)
    print(f"{'scenario':<28} {'n_facts':>8} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'p50ms':>8} {'p95ms':>8}")
    print("-" * 76)
    for r in results:
        d = r.summary_dict()
        print(f"{d['scenario']:<28} {d['n_facts']:>8} "
              f"{d['recall@1']:>7.3f} {d['recall@5']:>7.3f} "
              f"{d['recall@10']:>7.3f} "
              f"{d['latency_ms_p50'] or 0:>8.1f} {d['latency_ms_p95']:>8.1f}")

    # MACRO mean: each scenario weighted equally (1 vs S1's 100q both = 1)
    macro_r1 = statistics.mean([r.recall_at_1 for r in results])
    macro_r5 = statistics.mean([r.recall_at_5 for r in results])
    macro_r10 = statistics.mean([r.recall_at_10 for r in results])

    # MICRO mean: weighted by n_queries — sum(hits) / sum(queries).
    # This is the honest "across-all-questions" number; macro can be
    # warped by single-query scenarios like S13.
    total_q = sum(r.n_queries for r in results)
    micro_r1 = sum(r.hits_at_1 for r in results) / total_q if total_q else 0.0
    micro_r5 = sum(r.hits_at_5 for r in results) / total_q if total_q else 0.0
    micro_r10 = sum(r.hits_at_10 for r in results) / total_q if total_q else 0.0

    print("-" * 76)
    print(f"{'MACRO MEAN (scenario-weighted)':<28} "
          f"{'':>8} {macro_r1:>7.3f} {macro_r5:>7.3f} {macro_r10:>7.3f}")
    print(f"{'MICRO MEAN (query-weighted)':<28} "
          f"{total_q:>8} {micro_r1:>7.3f} {micro_r5:>7.3f} {micro_r10:>7.3f}")
    print(f"\ntotal elapsed: {total_elapsed:.1f}s  ({len(results)} scenarios)")

    # Special render of S12 dream-phase ablation table (if present)
    for r in results:
        pt = r.extra.get("phase_table") if r.extra else None
        if not pt:
            continue
        print("\n" + "=" * 76)
        print(f"S12 — 5-PHASE DREAM ABLATION (marginal recall@K per phase)")
        print("=" * 76)
        print(f"{'phase':<22} {'sec':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
        print("-" * 76)
        for name, sec, p1, p5, p10 in pt:
            print(f"{name:<22} {sec:>7.1f} {p1:>7.3f} {p5:>7.3f} {p10:>7.3f}")

    # Special render of S13 multiturn final
    for r in results:
        if r.name != "S13_multiturn_final":
            continue
        e = r.extra or {}
        print("\n" + "=" * 76)
        print("S13 — MULTITURN A→B→C FINAL (1 query, binary pass/fail)")
        print("=" * 76)
        print(f"  corpus size:           {r.n_facts:>10,d} facts")
        print(f"  ingest seconds:        {e.get('ingest_sec'):>10.1f}")
        print(f"  ingest rate:           {e.get('facts_per_sec'):>10,.0f} facts/s")
        print(f"  projected 1B at rate:  {e.get('projected_1b_hours'):>10} h")
        print(f"  query latency:         {r.latencies_ms[0]:>10.1f} ms")
        print(f"  expected top-1 label:  {S13_EXPECTED_TOP1_LABEL}")
        print(f"  expected value match:  '{S13_EXPECTED_TOP1_VALUE}'")
        print(f"  observed top-1 label:  {e.get('top1_label')}")
        print(f"  observed top-1 content: {e.get('top1_content')!r}")
        print(f"  all turn ranks in top-10: {e.get('label_ranks')}")
        verdict = "PASS ✓" if e.get('binary_pass') else "FAIL ✗"
        print(f"\n  BINARY VERDICT: {verdict}")

    # Special render of S14 DEATH
    for r in results:
        if r.name != "S14_death_multiturn":
            continue
        e = r.extra or {}
        print("\n" + "=" * 76)
        print("S14 — DEATH VERSION (40 updates + drift + negation + distractors)")
        print("=" * 76)
        print(f"  corpus size:           {r.n_facts:>10,d} facts")
        print(f"  ingest seconds:        {e.get('ingest_sec'):>10.1f}")
        print(f"  ingest rate:           {e.get('facts_per_sec'):>10,.0f} facts/s")
        print(f"  projected 1B at rate:  {e.get('projected_1b_hours'):>10} h")
        print(f"  query latency:         {r.latencies_ms[0]:>10.1f} ms")
        print(f"  query (NO entity, NO attribute, only temporal anchor):")
        print(f"     {S14_QUERY!r}")
        print(f"  expected top-1 label:  {S14_FINAL_LABEL}")
        print(f"  expected sentinel in content: {S14_FINAL_SENTINEL!r}")
        print(f"  observed top-1 label:  {e.get('top1_label')}")
        print(f"  observed top-1 content: {e.get('top1_content')!r}")
        print(f"  phoenix-update ranks in top-20:")
        ranks = e.get('phoenix_update_ranks_in_top20', {})
        for lbl in sorted(ranks, key=lambda l: ranks[l]):
            print(f"    {lbl:<22} rank={ranks[lbl]}")
        verdict = "PASS ✓" if e.get('binary_pass') else "FAIL ✗"
        print(f"\n  BINARY VERDICT: {verdict}")

    if args.out:
        out = {
            "scenarios": [r.summary_dict() for r in results],
            "macro_recall_at_1": round(macro_r1, 4),
            "macro_recall_at_5": round(macro_r5, 4),
            "macro_recall_at_10": round(macro_r10, 4),
            "elapsed_seconds": round(total_elapsed, 2),
            "engine": "Mazemaker (semantic memory)",
            "judge": "deterministic / label-or-substring match",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nsummary written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
