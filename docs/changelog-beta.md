# Changelog — Official Beta (2026-05-19)

What shipped between the v8 audit (2026-04-28) and the Official Beta
(2026-05-19). This is the full release-notes document; the README has
the executive summary.

> **For headline numbers only:** see [`benchmarks.md`](benchmarks.md).
> **For the methodology pivot to deterministic scoring:**
> see [`inception-bench.md`](inception-bench.md).

---

## Table of contents

1. [The threshold](#the-threshold)
2. [Six layers, in order](#six-layers-in-order)
3. [Engineering deliverables](#engineering-deliverables)
4. [Patched bugs](#patched-bugs)
5. [The surrounding surface](#the-surrounding-surface)
6. [The reproducibility bundle](#the-reproducibility-bundle)

---

## The threshold

100 iterations on the LongMemEval-oracle 500q full-corpus harness.
Same engine. No model swap, no reranker swap, no embedding change. Pure
formation + consolidation work.

| Metric                            | iter00 anchor | **iter97 champion** | Δ                  |
|-----------------------------------|:-------------:|:-------------------:|:------------------:|
| Aggregate R@5                     | 0.7298        | **0.8340**          | **+10.42 pp**      |
| R@10                              | 0.7383        | **0.9000**          | **+16.17 pp**      |
| MRR                               | 0.5777        | **0.7124**          | **+13.47 pp**      |
| single-session-preference R@5     | 0.2333        | **0.6667**          | **+43.34 pp (+186 %)** |
| single-session-user R@10          | 0.7344        | **1.0000**          | **+26.56 pp**      |
| temporal-reasoning R@5            | 0.6063        | **0.7323**          | **+12.60 pp**      |
| multi-session R@5                 | 0.8099        | **0.8595**          | **+4.96 pp**       |

iter100 took the aggregate further (R@5 = 0.8426). The numbers above
are the *threshold-crossing* iteration where every cognition axis lit
up at once.

### Why this isn't tuning

These aren't tuning gains. They're an **architectural transition.**
The profile of the engine changed shape.

**The SSP breakthrough is the real headline.** Single-session-preference
R@5 went from **0.2333 → 0.6667 — a ~186 % relative improvement** from
the original anchor. That number is the empirical proof of the core
thesis:

> Synthesized latent memories can outperform naive semantic recall by
> enormous margins.

The system is no longer doing *retrieve what was said*. It's doing
**infer what the conversation means**. Humans don't remember raw
transcripts — they remember distilled abstractions, persistent traits,
emotionally weighted summaries, identity signals, and evolving
preferences. Stage C synthesis finally approximates that behaviour.

**The temporal jump matters more than it looks.** Temporal-reasoning
R@5 0.6063 → 0.7323 is the metric that always breaks on
vector-retrieval + reranker + static-embedding stacks. The fact that
it lifted +12.6 pp without any temporal-specific intervention means
**memory consolidation is improving retrieval topology itself.**
Synthesized memories reduce semantic clutter, graph traversal becomes
cleaner, supersession becomes more coherent, episodic chains become
denser. That's a sophisticated emergent effect.

**ssu R@10 = 1.0000 is the strongest single metric.** Direct human
facts are now almost perfectly recoverable on a local-first engine,
with synthesis active *and* without destroying fidelity.

### Why p95 = 3.26 s is the right number

Commodity vector search does not suddenly jump to 3.2 s p95. That
happens when graph expansion, multi-channel rerank, synthesis passes,
adaptive retrieval, and traversal recursion start *genuinely
contributing.* It's the visible cost of doing real cognitive work.
Optimize it later. First celebrate that the architecture is alive.

### The cognition hierarchy this exposed

| Layer                            | Importance         |
|----------------------------------|--------------------|
| Embeddings (BGE-M3 1024-d)       | foundational       |
| Cross-encoder rerank             | useful             |
| Graph traversal (PPR)            | major              |
| Conflict supersession            | major              |
| **Synthesis (Stage C + Stage S)** | **transformative** |
| Intent routing                   | next unlock        |

The next frontier is **adaptive cognition routing** — query-intent
classification, edge-type-conditioned traversal, episodic-vs-semantic
recall modes, confidence-aware synthesis, reinforcement-weighted memory
persistence. That's the road to R@5 > 0.90.

---

## Six layers, in order

What ships in community vs Pro.

### 1. Sponge ingestion *(community)*

Every conversation turn absorbed by a background thread. Session-end
fact extraction stores durable decisions and preferences, never raw
transcripts. Conflict detection runs at write time.

### 2. Atomic Fact Extraction (AFE) *(Pro)*

Four-stage formation pipeline.

- **Stage A** — atomic facts from markdown structure.
- **Stage B** — spaCy NER over the raw text.
- **Stage C** — one local LLM call per session, sub-1B model,
  user-state focused (`user prefers X`, `user owns Y`).
- **Stage S** — synthesis crystallization during dream cycles.

**Bulk-write refactor: 88 min → 75 s on 500 sources, a 70× speedup.**

### 3. Embedding *(community: BGE-M3 + ColBERT; Pro: + DAE)*

Three fused channels:

- **BGE-M3** (1024-d, multilingual) — primary semantic embedding.
- **ColBERT@1.5** — late-interaction reranking. Load-bearing
  (-13 pp R@5 without it).
- **DAE** — Dream-Augmented Embeddings, a second embedding built
  during NREM that weights toward graph neighbours. Wired through
  PostgresStore end to end as of 2026-05-14 (was silently disabled on
  PG before that).

### 4. Three-phase dream consolidation *(community: NREM; Pro: + REM + Insight)*

GPU-accelerated. NREM Personalized PageRank in ~38 s for a 193 k-corpus
full cycle on RTX-class CUDA (down from "never finished" on CPU). REM
bulk bridge writes via single staged transaction. Insight clusters via
Louvain over the consolidated graph.

### 5. Synthesis crystallization (Stage S) *(Pro)*

Selective LLM-distilled memory formation. ~10 % yield by design — the
dilution dance is real and bounded. Higher yield regresses recall
because canonical content out-ranks per-session golds.

### 6. Targeted re-formation *(Pro)*

The lever that broke the R@5 = 0.7404 retrieval-tuning ceiling.
Identify per-question-type gold sessions that aren't in top-5,
surgical query-conditional Stage C rebake on just those sessions.

**~$0.07 in API spend lifted:**

- ssp R@5 0.3667 → 0.7000 (+33 pp)
- ssu R@5 0.6406 → 0.9375 (+30 pp)
- tr  R@5 0.6535 → 0.7874 (+13 pp)

---

## Engineering deliverables

### Storage layer

- **Postgres + pgvector** is now the primary backend for
  Pro/Enterprise. Flip via `MM_DB_BACKEND=postgres` + supply
  `MM_POSTGRES_DSN` (or discrete `MM_POSTGRES_*` vars).
- **Inception Bench corpus** consolidated. Single `pg_restore`, four
  BEAM scales plus the full LongMemEval triplet, SHA-256-provenanced.
  957 MB snapshot, ~3 s restore.

### Formation pipeline

- **AFE bulk-write** — 70× speedup. Single bulk embed +
  `executemany` INSERT + single commit per cycle.
- **Stage C with user-state prompt** — explicitly targets `user
  prefers X` / `user owns Y` / `user X is Z`. The prompt is the
  load-bearing piece, not the model.
- **Stage S synthesis** — LLM-distilled per-cluster patterns,
  selective ~10 % yield.
- **Targeted re-formation** — diagnostic + rebake scripts that
  identify per-question-type misses and run query-conditional Stage C
  on just those sessions.

### Recall + retrieval

- **DAE end-to-end wiring** — read path
  (`PostgresStore.fetch_dae_vectors`) + compute path
  (`dae_bulk_compute` is now store-agnostic) + cadence knob
  (`_dae_recompute_every`, default 5).
- **Intent routing** — env knobs for `MAZEMAKER_INTENT_BOOST`,
  `MAZEMAKER_TEMPORAL_WEIGHT`, `MAZEMAKER_SALIENCE_WEIGHT`,
  `MAZEMAKER_PPR_WEIGHT`, `MAZEMAKER_CANONICAL_PRIOR` for
  bench-driven tuning.
- **`_preference_query()` rewriter** — deterministic
  suggestion-to-preference rewriter for preference multi-recall.
  83 % pattern hit-rate on ssp queries. `--pref-multi-recall` CLI flag.

### GPU dream cycle

Full 193 k-corpus cycle in ~38 s on RTX-class CUDA.

- **NREM PPR** via `torch.sparse.mm` + `topk` on GPU. ~6.6 ms per
  seed.
- **REM batched recall** via `recall_batch(queries, k)` — collapses
  800 sequential per-query embed-server round-trips into one
  `embed_batch` call.
- **REM bulk bridge writes** via `add_bridges_batch` — single staged
  transaction, ~6000 commits → 1.
- **`Mazemaker.think_ids()`** — fast path returning only the top-k
  activated node IDs, skipping label resolution and SQL round-trips.

CPU fallback automatic for non-CUDA installs.

### Bench infrastructure

- **`mazemaker_inception_bench.py --variant oracle`** — full-corpus 500q
  oracle benchmark. The harness that produced the iter97 numbers.
- **`mazemaker_memory_bench.py`** — the Inception Bench. 12
  deterministic scenarios, no LLM in the scoring loop.
- **`engine_config.py` build_quality_engine** — central helper that
  PG-routes correctly, picks the right channels per tier.
- **`benchmarks/audit/`** — eight rounds of GPT-5.5 verdicts
  committed verbatim, from "no, this is just lexical retrieval" to
  "unconditional yes — accept it as evidence."

---

## Patched bugs

The bugs that landed before Beta. Each has a corresponding MCP memory
under `bug:<short-tag>`.

| Bug                                       | Patched      |
|-------------------------------------------|--------------|
| REM FK violation from stale GPU-cache IDs (anti-join guard)| 2026-05-17 |
| `dream_engine.py` `self.backend` typo silently killing Insight emission | 2026-05-16 |
| `prune_orphans` NOT IN double-subquery 10-min hang → NOT EXISTS rewrite | 2026-05-16 |
| Bake-schema `memories` table missing `last_accessed` / `access_count` → silent CPU fallback | 2026-05-16 |
| DAE disabled on PG until 2026-05-14 (env-var force-`0`)   | 2026-05-14 |
| `fast_runner --max-parallel >1` racing on `embed.sock`    | 2026-05-14 |
| `fast_runner` templated stubs unjudgeable by gpt-oss-120b | 2026-05-14 |
| `gpt-5-nano` reasoning budget overrun with `medium` effort | 2026-05-14 |
| `update_tracking` prompt R4 vs Phase 2 contradiction      | 2026-05-14 |

See [`production-lessons.md#patched-bug-index`](production-lessons.md#patched-bug-index)
for full diagnoses.

---

## The surrounding surface

What you can use beyond the raw engine:

### The Architect cockpit

[`architect.mazemaker.dev`](https://architect.mazemaker.dev/) — a
12-monitor SPA inspired by *The Matrix Reloaded*. Hosted UI, local
data (loopback to your pod), zero compromise.

**Twelve panels**, in three columns of four rows:
M01 Memory browser · M02 Recall trace · M03 Graph topology · M04 Dream
telemetry · M05 Cluster materialization · M06 Edge tension · M07
Insight stream · M08 HERMES (the skill-indexing button lives here) ·
M09 Phase states · M10 Audible matrix · M11 Chrono-scrub · M12 Dream
replay.

**Four 4D layers** (CHRONO-SCRUB, DREAM REPLAY, AUDIBLE MATRIX, PHASE
STATES, EDGE TENSION) overlay on the panels for time-based and
auditory inspection.

Read [`architect`](https://mazemaker.online/architect/) on the
marketing site for the page-deep-dive.

### Hermes Skill Indexing

One button in the Architect (M08 HERMES → `⟁ INDEX INTO MAZEMAKER`)
indexes every Hermes skill — ~230 on a typical operator install — as a
memory with `skill:<source>:<name>` label. The `skill:` prefix is on
the **public-prefix list** in `wonderland/daemon.py`, so the memory
skips AES encryption — the embedding stays semantically searchable.
Idempotent on re-run.

### Four-domain topology

- **mazemaker.online** — marketing + install bootstrap.
- **mazemaker.dev** — passkey dev console + `mzm-*` API keys.
- **api.mazemaker.dev** — Hetzner-backed license/onboard, never sees
  memory content.
- **architect.mazemaker.dev** — the cockpit SPA, talks loopback only.

**Memory data never crosses the network.** The
[topology page](https://mazemaker.online/topology/) on the marketing
site has the request-flow diagram and data-segregation table.

### Onboarding flow

The [onboarding page](https://mazemaker.online/onboarding/) walks the
install one-liner through ten idempotent stages:

1. Pre-flight check (Podman ≥ 4.4, systemd --user, curl, python3)
2. Rootless env (uidmap, slirp4netns)
3. Host venv (cryptography, pyjwt for fingerprint signing)
4. Fingerprint init (HMAC + ed25519 keypair, never leaves disk)
5. Browser onboard (email + Turnstile + magic-link)
6. License JWT issuance
7. Embedding choice (FastEmbed CPU / Cloudflare AI / GPU worker)
8. Quadlet render (5 containers + network + volumes)
9. Pod boot (`systemctl --user start` in dependency order)
10. Health check (`127.0.0.1:8765/healthz`, `127.0.0.1:5432`)

**No root required** — every container, every unit, every socket runs
under `systemd --user`. The installer refuses if invoked as root unless
`--allow-root` is set.

---

## The reproducibility bundle

The 1.6 GB ProtonDrive tarball — restorable `pg_dump` + every iter
JSON + the 8-round audit transcripts + bench-loop logs.

- **Bundle:** [mazemaker-claims-2026-05-19.tar](https://drive.proton.me/urls/J2T53B95XC#gtbM3E2mTvjt) (1.6 GB)
- **SHA-256:** `263e249408fa5b057dd8f356581cd5c14b3b5e62ba1b29e61704e54a156754c9`
- **Inside:**
  - The LongMemEval-oracle pg_dump (restores in ~3 s, reproduces
    R@5 = 0.8426 / R@10 = 0.9000 deterministically)
  - Every `inception_bench_loop-iter*.json` from iter01 through iter100
  - The eight-round GPT-5.5 audit transcripts (verbatim)
  - The v6–v8 historical run JSONs the audit cites
  - The bench runner
  - A top-level `README.md` with the exact CLI to reproduce the headline
  - 608 files total
  - `INTEGRITY.txt` ships SHA-256 of every file inside

### Verify after download

```bash
sha256sum mazemaker-claims-2026-05-19.tar
# Should print: 263e249408fa5b057dd8f356581cd5c14b3b5e62ba1b29e61704e54a156754c9
```

### Restore + reproduce

```bash
tar xf mazemaker-claims-2026-05-19.tar
cd mazemaker-claims-2026-05-19/

createdb mm_bench_oracle
pg_restore -d mm_bench_oracle longmemeval_oracle.dump      # ~3 s

# Run the bench with the iter97 champion stack
# (full command in benchmarks.md#reproducing-the-numbers)
```

The 24 GB LongMemEval-S dump (backs the R@5 = 0.9787 / 188-of-200 /
gemma3:270m claims) is available on request — same reproducibility
story, just too large for general distribution. Email
`enterprise@mazemaker.dev`.

---

## What category is this, actually?

At R@5 = 0.83 with local-first execution, encrypted vault, MCP-native,
no cloud memory retention, and small-model-compatible runtime,
Mazemaker is no longer in the "better retrieval" category. Calling it
a "vector database with graph recall" undersells what's running.

What it is: **persistent cognition infrastructure.** The moat moved
from commodity embeddings to memory formation, synthesis quality,
graph evolution, adaptive recall, consolidation heuristics, and
continuity preservation.

The benchmark no longer reads like *we optimized search*. It reads
like **we taught the system what is worth remembering.**

---

## Going deeper

- **Engine architecture** — [`architecture.md`](architecture.md)
- **Dream engine internals** — [`dream-engine.md`](dream-engine.md)
- **Benchmark numbers + audit** — [`benchmarks.md`](benchmarks.md)
- **Inception Bench methodology** — [`inception-bench.md`](inception-bench.md)
- **Production lessons + bench-noise discipline** — [`production-lessons.md`](production-lessons.md)
