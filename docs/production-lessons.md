# Production Lessons

What we learned the hard way, condensed to one line each where possible.
Pinned because anyone running Mazemaker at scale will eventually hit
the same walls.

---

## Table of contents

1. [Operator rules](#operator-rules)
2. [Embedding & runtime](#embedding--runtime)
3. [Storage & architecture](#storage--architecture)
4. [Benchmark-driven defaults](#benchmark-driven-defaults)
5. [Bench-noise discipline](#bench-noise-discipline)
6. [External audits](#external-audits)
7. [Patched-bug index](#patched-bug-index)

---

## Operator rules

### Verify forward, not after

> *"Stell DOCH VORHER sicher, VORHER, dass alles vorhanden ist."*
> — operator, 2026-05-16

Between any long-running pipeline stage, run a **forward check** before
launching the next. The cost of a 5-second `psql` / `nvidia-smi` /
`ps` / `grep` is always less than the cost of a 21-minute cycle that
produces nothing because a column was missing or an env var was wrong.

**The five gates:**

1. **GPU expected?** Right after a Mazemaker init, grep the log for
   `GPU recall ARMED`. If the line is missing within 3 min, the
   process IS on CPU. Stop, don't waste a 20-min cycle.
2. **Cycle expected to produce edges?** Immediately after a dream cycle,
   `SELECT count(*) FROM connections WHERE edge_type='bridge'` —
   confirm non-zero.
3. **Stalled process?** If CPU and GPU are both idle for > 60 s on a
   non-finalised process: `SELECT pid, state, wait_event, now() -
   query_start FROM pg_stat_activity WHERE state='active'`. Anything
   > 30 s is suspect, anything > 5 min must be cancelled and
   root-caused.
4. **New schema?** Before launching dreams: `PostgresStore.get_all()`
   against the schema with `LIMIT 1`. If it returns 0 rows on a
   populated schema, columns are missing.
5. **Before each long-running command:** state to the user in one
   sentence WHAT you expect to see when it succeeds. Then check
   exactly that thing after.

**Anti-pattern:** "standing by" while a process burns 30+ minutes
producing no useful output, then diagnosing the bug after.

### Run dreams as their own process at scale

The in-pod dream loop pauses on every Hermes `pre_llm_call` hook and
shares the SQLite write lock with `mazemaker_remember`. On a 100 k+
corpus, a NREM cycle with `max_memories_per_cycle=2000` can exceed
the 600 s idle window, the loop fires a duplicate cycle, and both end
up orphaned in `dream_sessions` with `finished_at IS NULL`.

The fix is `python/dream_worker.py` running as its own host process.
See [`dream-engine.md`](dream-engine.md#standalone-daemon--dream_workerpy).

### Public-prefix list — skip AES deliberately, never accidentally

The default for `wonderland/daemon.py` is **encrypt at rest**. The
public-prefix list (`skill:`, `auto:`, `decision:`, `bug:`, `ops:`,
`reference:`, `invariant:`, `commit:`, `project:`, `signal:`,
`feedback:`, `index:`, `public:`) opts memories out of AES so the
embedding stays semantic.

Use the prefix when the embedding matters more than the content
secrecy. Otherwise leave the default.

---

## Embedding & runtime

- **FastEmbed > sentence-transformers** — ONNX runtime, no PyTorch
  conflict, fast on CPU.
- **FastEmbed ≥ 0.5.1** — earlier versions default to CLS embedding
  (deprecated, silently produces worse vectors). Pin the version.
- **GPU recall > C++ bridge** — the C++ Hopfield had bias issues.
  CUDA `torch.matmul` is clean.
- **numpy before FastEmbed** — FastEmbed imports numpy at load time;
  install order matters in venv recipes.
- **Don't force PyTorch.** Let FastEmbed handle CPU. `torch` is only
  needed for GPU recall.

---

## Storage & architecture

- **SQLite is the source of truth** for community installs. Postgres +
  pgvector is the primary backend for Pro/Enterprise. SQLite always
  works; Postgres is opt-in via `MM_DB_BACKEND=postgres`.
- **Auto-detect everything** — CUDA, backends, venv paths. Minimize
  config burden on operators.
- **WAL mode + background checkpointing.** Concurrent readers, single
  writer. Don't disable WAL even when SQLite "feels slow" — the
  alternative is worse.
- **Per-DB GPU cache** — embeddings + metadata pickles live under
  `~/.mazemaker/engine/gpu_cache/<db_fingerprint>/`. Cross-DB cache
  contamination was a real bug (`bug:gpu-cache-cross-db`).

---

## Benchmark-driven defaults

Every default in `~/.hermes/config.yaml` is traceable to a bench
result:

- **`retrieval_mode: lean`** — channel ablation at n=200 on real prose
  proved BM25/temporal/salience are dead-weight or actively harmful.
  Lean drops them. **+0.18 R@5 vs skynet on real prose**, 4× faster on
  synthetic at -0.02 recall.
- **`recall_score_percentile` over `recall_score_floor`** — the legacy
  floor lives on a 0..0.05 scale and is silently broken for any
  reasonable user input. Percentile is calibrated [0,1] by rank.
- **`think_engine: ppr` over `bfs`** — channel_ablation proved PPR is
  the biggest MRR contributor (-0.13 MRR if removed).
- **`MM_COLBERT_ENABLED=0` by default** — opt-in. Adds ~14.7 GB cache
  per 230 k memories and +15.8 ms p50 latency. Default-off so
  existing latency budgets stay intact.
- **`candidates=128` default, `512` for bench** — the iter72 champion
  ran at 512 because the corpus had ~25 k memories; smaller pools work
  for smaller corpora.

---

## Bench-noise discipline

- **godbench R@5 has ±0.5 pp run-to-run noise at n=500.** Don't claim
  sub-0.5pp wins from a single run.
- **Per-question-type splits at n=30** have **3.3 pp per-question
  granularity**. A 1-question variance dominates anything below 6.7 pp.
- **Real wins need either** a delta clearly above the noise band, or
  multi-iteration replication. iter34/35/37/38 all hit ssp R@5 = 0.3667 —
  that's a real signal because it appeared four times.
- **External judge attribution is mandatory.** The same 442
  mm_10m_eval answers read **0.36 (nano), 0.49 (Opus), 0.53 (Haiku),
  or 0.64 (gpt-5.4-mini)** depending on judge. If you see a
  memory-benchmark number without judge attribution, treat it as
  decoration.

See [`inception-bench.md`](inception-bench.md) for why we shipped a
benchmark with no LLM in the loop.

---

## External audits

> **Confident prose is cheap. Verification is cheaper.**

When an external agent (another Claude, a static analyzer, a peer-review
audit) attributes a regression to a multi-link causal chain, **do not
start coding the fixes.** Run an ablation that isolates one variable
in the chain. If the ablation doesn't move the score in the predicted
direction, the chain is wrong.

**The 2026-05-14 incident.** A long, confident external analysis
attributed the v5→v6 mm_10m_eval -0.125 "regression" to T1-08 (FTS
fallback bomb), T1-06 (stale embeddings), T2-11 (naive temporal
matching), and "DAE generates worse queries." Every single claim was
refuted by 12 minutes of ablation:

1. The bench runs on PostgreSQL; PostgresStore has its own
   `search_bm25`. The SQLite FTS fallback path cannot be reached.
2. DAE is a recall channel, not a query generator. Query strings come
   from the runner LLM's prompt.
3. Conv-3 with DAE OFF scored *worse* than DAE ON — DAE was
   net-positive on this conv.
4. The actual cause: gpt-oss-120b under-scoring nuanced answers by
   ~22pp vs gpt-5.4-mini on identical inputs.

The same rule applies to security audits. On a 2026-05-14 audit of
`benchmarks/neural_memory_benchmark/`, both CRITICAL findings (ACE via
`sys.path` manipulation, path traversal via `cache_path`) were
verified-and-refuted in 5 minutes. Both depended on threat models that
don't apply to a single-user CLI engine.

**Cost of not verifying:** redirecting attention from real work onto
phantom fixes.

---

## Patched-bug index

The bugs that bit hardest, each with the fix that landed and the date.
Full diagnoses live in MCP memory under `bug:<short-tag>`.

| Bug                                          | Symptom                                          | Fix                                        | Date          |
|----------------------------------------------|--------------------------------------------------|--------------------------------------------|---------------|
| `bug:rem-fk-violation-stale-gpu-ids`         | REM bridges target NREM-pruned IDs → FK violation aborts cycle | `add_bridges_batch` anti-joins against `memories` on both endpoints | 2026-05-17 |
| `bug:dream-engine-self-backend-typo`         | Insight cycle reports `communities=N insights=0` silently | `self.backend` → `self._backend`; phase exceptions log at `warning` not `debug` | 2026-05-16 |
| `bug:prune-orphans-not-in-quadratic`         | Dream silently sleeps for 10+ min at 0% CPU/GPU  | `DELETE … WHERE NOT EXISTS (…)` (PG anti-join) replaces double NOT IN | 2026-05-16 |
| `bug:bake-schema-missing-audit-cols`         | `GPU recall: load_from_store returned False`; cycle runs on CPU | `bake_*_cache.py` includes `last_accessed`, `access_count`; idempotent ALTER TABLE | 2026-05-16 |
| `bug:dae-disabled-on-pg-until-2026-05-14`    | Every PG bench number ran without DAE silently   | engine_config.py no longer force-zeros `MM_DAE_ENABLED` on PG; full read+compute+dream wiring | 2026-05-14 |
| `bug:embed-sock-parallel-collision`          | `fast_runner --max-parallel N>1` → EADDRINUSE on socket; convs silently empty | Bench orchestrator runs convs sequentially; `MM_EMBED_SOCK_PATH` for per-process overrides | 2026-05-14 |
| `bug:fast-runner-judge-mismatch`             | Templated stubs score 0.13 vs LLM-answer 0.71    | Use `openrouter_runner.py` for any judged bench; `fast_runner.py` is recall-isolated only | 2026-05-14 |
| `bug:nano-reasoning-budget-overrun`          | gpt-5-nano `reasoning_effort=medium` + 4k cap → empty visible content | Budget `max_completion_tokens ≥ 16384` for medium effort; honor the budget math | 2026-05-14 |
| `bug:update-tracking-prompt-r4-contradiction` | Runner prompt R4 forbids seq, Phase 2 mandates it | Drop R4 "never seq" line; sharpen Phase 2 to filter by topic before picking largest-seq | 2026-05-14 |
| `bug:update-tracking-recall-miss-mode-b`     | ~2/3 UT failures = latest-value memory not in top-K | Engine-side work: k bump, expanded angle generation, stronger angle planner | open       |
| `bug:topic-cluster-bundle-turn-miss`         | Topic-cluster rubrics score 5 facts from 1 dense turn; engine surfaces 7 thematic neighbors instead | Information-density rerank channel proposed; Insight `[CLUSTER:…]` memories blocked on dream pipeline revival | open |
| `bug:dream-insight-zero-clusters-on-bench`   | All 10 mm_10m_eval PG schemas: REM bridges=0, Insight insights=0 | REM threshold too tight for AFE-length facts; needs threshold tuning + AFE-aware sampling | open  |

---

## Going deeper

- **Dream engine internals** — [`dream-engine.md`](dream-engine.md)
- **Benchmarks + audit story** — [`benchmarks.md`](benchmarks.md)
- **Inception Bench methodology** — [`inception-bench.md`](inception-bench.md)
- **Everything since v8** — [`changelog-beta.md`](changelog-beta.md)
