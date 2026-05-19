# Configuration

Every knob, with defaults, why each default was picked, and what happens if
you change it. All YAML lives in `~/.hermes/config.yaml`; all env vars are
plain environment variables you can `export`.

> **Quick orientation:** the defaults below are the recommended preset based
> on the v8 benchmark + the 100-iteration tuning loop. Most users should
> not change anything except `embedding_backend` (if CPU-only) and
> optionally `retrieval_mode`.

---

## Table of contents

1. [The minimal config](#the-minimal-config)
2. [The recommended config](#the-recommended-config)
3. [Retrieval mode cheat sheet](#retrieval-mode-cheat-sheet)
4. [Environment variables](#environment-variables)
5. [Tier-gated features](#tier-gated-features)
6. [Tuning recipes](#tuning-recipes)

---

## The minimal config

This is what `install.sh` writes if no `config.yaml` exists. It works on
every machine.

```yaml
memory:
  provider: neural
  neural:
    embedding_backend: auto
    retrieval_mode: lean
    db_path: ~/.mazemaker/engine/memory.db
```

That's it. Defaults take care of everything else.

---

## The recommended config

For operators who want every channel wired correctly out of the box.

```yaml
memory:
  provider: neural
  neural:
    # ── Storage ──────────────────────────────────────────────────────
    db_path: ~/.mazemaker/engine/memory.db

    # ── Embedding ────────────────────────────────────────────────────
    # auto = pick FastEmbed if installed, else sentence-transformers,
    #        else TF-IDF, else hash.
    embedding_backend: fastembed     # auto | fastembed | sentence-transformers | tfidf | hash

    # ── Retrieval ────────────────────────────────────────────────────
    # `lean` beat `skynet` by +0.18 R@5 / +0.16 MRR on real prose at
    # n=200 and is 4× faster on synthetic at -0.02 recall.
    retrieval_mode: lean             # semantic | hybrid | advanced | skynet | lean | trim
    retrieval_candidates: 128
    use_hnsw: auto                   # ANN index above ~1k memories
    think_engine: ppr                # bfs | ppr — PPR is load-bearing for ranking

    # Calibrated [0,1] noise floor by rank.
    recall_score_percentile: 0.3

    # Optional: MMR diversity in result set (0.0=pure relevance, 0.7=balanced).
    mmr_lambda: 0.0

    # ── Session ──────────────────────────────────────────────────────
    prefetch_limit: 10
    search_limit: 50
    consolidation_interval: 0
    session_extract_facts: true
    session_fact_limit: 5

    # ── Dream ────────────────────────────────────────────────────────
    dream:
      enabled: true
      idle_threshold: 600            # seconds before dream cycle
      memory_threshold: 50           # dream after N new memories
```

> **Postgres backend.** Set `MM_DB_BACKEND=postgres` in the environment and
> supply `MM_POSTGRES_DSN` (or discrete `MM_POSTGRES_*` vars). No YAML
> change needed — the engine picks up the env at boot.

---

## Retrieval mode cheat sheet

| Mode       | Channels active                            | Use when                                     |
|------------|--------------------------------------------|----------------------------------------------|
| `semantic` | semantic only                              | Lowest latency, no hybrid fusion needed      |
| `hybrid`   | semantic + BM25                            | Add lexical recall                           |
| `advanced` | semantic + BM25 + entity                   | + named-entity grounding                     |
| `skynet`   | all six channels (semantic + BM25 + entity + temporal + salience + PPR) | Default in older configs; over-channeled per the v8 audit |
| **`lean`** | semantic + entity + PPR                    | **Recommended.** Drops dead-weight channels  |
| `trim`     | semantic + BM25 + entity + temporal + PPR  | Conservative middle ground (drops only salience) |

**Why `lean` won.** Channel ablation at n=200 on real prose showed
BM25/temporal/salience are dead-weight (or actively harmful) on natural
language. `lean` drops them and gains 0.18 R@5 over `skynet`. On
synthetic / templated content, `skynet` still has the edge — pick by
corpus shape.

**PPR is the biggest MRR contributor.** Removing it costs −0.13 MRR.
Always leave `think_engine: ppr` unless you have a specific reason.

---

## Environment variables

The engine respects a long list of env vars. Most never need touching;
listed here so you know what's available.

### Storage / backend

| Variable                  | Default                              | What it does                                          |
|---------------------------|--------------------------------------|-------------------------------------------------------|
| `MM_DB_BACKEND`           | `sqlite`                             | Set to `postgres` to use pgvector primary             |
| `MM_POSTGRES_DSN`         | unset                                | Full Postgres DSN (overrides discrete vars)           |
| `MM_POSTGRES_HOST`        | `127.0.0.1`                          | PG host                                               |
| `MM_POSTGRES_PORT`        | `5432`                               | PG port                                               |
| `MM_POSTGRES_DB`          | `mazemaker`                          | PG database name                                      |
| `MM_POSTGRES_USER`        | `mazemaker`                          | PG user                                               |
| `MM_POSTGRES_PASSWORD`    | unset                                | PG password                                           |
| `MM_POSTGRES_SCHEMA`      | `public`                             | PG schema (set per-conv for bench)                    |

### Embedding

| Variable                  | Default                              | What it does                                          |
|---------------------------|--------------------------------------|-------------------------------------------------------|
| `EMBED_BACKEND`           | `auto`                               | Override embedding backend choice                     |
| `MM_EMBED_SOCK_PATH`      | `~/.mazemaker/engine/embed.sock`     | Per-process socket override (avoid bind collisions)   |
| `NO_COLOR` / `TERM=dumb`  | unset                                | Strips ANSI from ollama subprocess output             |

### Retrieval

| Variable                       | Default | What it does                                          |
|--------------------------------|---------|-------------------------------------------------------|
| `MM_COLBERT_ENABLED`           | `0`     | `1` enables ColBERT@1.5 late-interaction rerank       |
| `MAZEMAKER_INTENT_BOOST`       | `0.0`   | Per-intent boost weight (0.10 is iter72 champion)     |
| `MAZEMAKER_TEMPORAL_WEIGHT`    | `0.10`  | Temporal channel weight in relevance formula          |
| `MAZEMAKER_SALIENCE_WEIGHT`    | `0.0125`| Salience channel weight                               |
| `MAZEMAKER_PPR_WEIGHT`         | `0.055` | PPR channel weight (0.55 in champion is bench-scale)  |
| `MAZEMAKER_CANONICAL_PRIOR`    | `0`     | Canonical-prior boost (net-negative; leave off)       |

### Dream

| Variable                  | Default | What it does                                          |
|---------------------------|---------|-------------------------------------------------------|
| `MM_DREAM_DISABLED`       | `0`     | `1` disables the in-pod dream loop (use with `dream_worker.py`) |
| `MM_DAE_ENABLED`          | `1`     | DAE channel on/off (was force-`0` on PG before 2026-05-14) |
| `MM_DAE_RECOMPUTE_EVERY`  | `5`     | Cadence: rebuild DAE every N NREM cycles. `0` disables |
| `MAZEMAKER_AFE_STAGE_A`   | `1`     | Markdown extraction stage (set `0` to force Stage C)  |
| `MAZEMAKER_AFE_SKIP_CHUNKS` | `0`   | `1` skips chunked sources (formation tier focus)      |
| `MAZEMAKER_AFE_N_WORKERS` | `1`     | Parallel AFE workers (shards via `id % n_workers`)    |
| `WORKER_ID`               | `0`     | Worker shard index when `N_WORKERS > 1`               |

### Bench / runner

| Variable           | Default          | What it does                                  |
|--------------------|------------------|-----------------------------------------------|
| `MM10M_CONV`       | unset            | Set integer N to route bench to PG schema `conv_N` |
| `MM_RUNNER_MODEL`  | `gpt-5-nano-2025-08-07` | Runner LLM model ID                  |
| `MM_JUDGE_MODEL`   | `gpt-5-nano-2025-08-07` | Judge LLM model ID                   |
| `MM_JUDGE_MAX_TOKENS` | `4096`        | Judge `max_completion_tokens` (16k+ for reasoning mid-effort) |

---

## Tier-gated features

Every gate is a plain `if has_feature()` call in
[`python/license.py`](../python/license.py) — grep, audit, fork.

| Feature                          | Community | Pro / Enterprise |
|----------------------------------|:---------:|:----------------:|
| Hybrid recall (BM25 + dense)     | ✅        | ✅               |
| LongMemEval-S R@5 = 0.96         | ✅        | ✅               |
| NREM dream consolidation         | ✅        | ✅               |
| SQLite WAL backend               | ✅        | ✅               |
| MCP server + CLI                 | ✅        | ✅               |
| Hop-2 graph reasoning (R@10 = 1.0)| ✅       | ✅               |
| Conflict supersession            | ✅        | ✅               |
| **ColBERT@1.5 late-interaction** | ❌        | ✅ → R@5 = 0.98  |
| **REM dream phase** (bridges)    | ❌        | ✅               |
| **Insight dream phase** (clusters)| ❌       | ✅               |
| **DAE (Dream-Augmented Embeddings)** | ❌    | ✅               |
| **AFE Stage C + Stage S synthesis** | ❌     | ✅               |
| **Autonomous dream-worker**      | ❌        | ✅               |
| **Architect UI** (visual cockpit)| ❌        | ✅               |
| **Postgres + pgvector backend**  | ❌        | ✅               |
| One-line installer + auto-update | ❌        | ✅               |
| Operator-direct email support    | ❌        | ✅               |

The community engine is **real and useful** — it produced the audit-grade
R@5 = 0.9787 LongMemEval-S hybrid number. Pro adds the cognition layers
(formation + consolidation + synthesis) that produced the iter97 R@5 = 0.8340
on the harder full-corpus oracle harness.

---

## Tuning recipes

### "I want lowest latency"

```yaml
retrieval_mode: semantic
recall_score_percentile: 0.5
use_hnsw: true
```

`MM_COLBERT_ENABLED=0`. Expect p50 ~10 ms for a 1 k corpus.

### "I want best quality, latency be damned"

```yaml
retrieval_mode: skynet
retrieval_candidates: 512
think_engine: ppr
recall_score_percentile: 0.3
mmr_lambda: 0.3
```

```bash
export MM_COLBERT_ENABLED=1
export MAZEMAKER_INTENT_BOOST=0.10
export MAZEMAKER_TEMPORAL_WEIGHT=0.7
export MAZEMAKER_PPR_WEIGHT=0.55
```

Expect p95 ~3 s on a 25 k corpus with the full Pro stack. This is the
iter72 champion config from the 100-iteration loop.

### "I'm running benchmarks"

```bash
export MM_DB_BACKEND=postgres
export MM10M_CONV=3                      # bench routes to schema conv_3
export MM_COLBERT_ENABLED=1
export MAZEMAKER_AFE_SKIP_CHUNKS=1
export MAZEMAKER_AFE_N_WORKERS=4
```

See [`benchmarks.md`](benchmarks.md) for the full runner setup.

### "I want autonomous dreams on a big corpus"

Move the dream loop out of the pod into a standalone daemon — it stops
fighting the writer lock.

```bash
# Disable in-pod dream (ephemeral; clears on reboot)
systemctl --user edit --runtime mazemaker-mcp.service
# add:  [Service]
#       Environment="MM_DREAM_DISABLED=1"
systemctl --user daemon-reload && systemctl --user restart mazemaker-mcp.service

# Run the standalone daemon
cd ~/projects/mazemaker/python
python dream_worker.py --max-memories 2000 --max-isolated 800
```

See [`dream-engine.md`](dream-engine.md) for the daemon flags.
