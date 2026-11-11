# Mazemaker — An Operating System for AI Agents

> **Your agents die every conversation. Mazemaker keeps them alive.**
>
> The persistent layer your LLMs run on top of. Memory formation, not retrieval.
> Background consolidation while they sleep. Conflict supersession when your mind changes.
> A knowledge-graph filesystem your agent walks instead of searches.

---

## What this is

Most AI "memory" systems are retrieval wrappers.

They store chunks. Embed text. Run cosine similarity. Return vaguely related paragraphs.

That works until the assistant needs to:

* track evolving preferences,
* resolve contradictions,
* follow temporal chains,
* infer latent traits,
* connect sessions together,
* or remember what actually mattered.

Mazemaker is built around a different thesis:

> Memory is not retrieval.
>
> Memory is formation, consolidation, synthesis, and evolving structure.

The engine continuously transforms raw conversations into a living cognitive graph:

* atomic facts,
* semantic links,
* supersession chains,
* synthesized abstractions,
* bridge memories,
* latent preference structures,
* temporal trajectories.

It does this locally.
It works with MCP agents.
It survives across sessions.

---

## Not memory. The kernel.

Vector search retrieves nearby text. Mazemaker manages the cognition itself:

| Kernel concept     | Mazemaker equivalent                          |
| ------------------ | --------------------------------------------- |
| Processes          | Your agents                                   |
| Memory management  | Consolidation + supersession                  |
| Filesystem         | The knowledge graph                           |
| Scheduler          | Dream cycles                                  |
| IPC                | Federation                                    |

The difference is not a percentage. It is a phase change — questions vector databases cannot answer by construction become routine.

| Capability                                  | Vector DB | Mazemaker |
| ------------------------------------------- | :-------: | :-------: |
| Find a fact you told it once                |    ✅      |    ✅      |
| Follow A → B → C reasoning chains           |    ❌      |    ✅      |
| Notice related facts should connect         |    ❌      |    ✅      |
| Replace stale facts when your mind changes  |    ❌      |    ✅      |
| Explain why recall happened                 |    ❌      |    ✅      |
| Get sharper while idle, not noisier         |    ❌      |    ✅      |

---

## Results

### LongMemEval-oracle — 500 questions, 25k-memory haystack

| Metric      | Score        |
| ----------- | -----------: |
| R@1         | —            |
| R@5         | **0.8426**   |
| R@10        | **0.9000**   |
| MRR         | —            |
| p50 latency | 1728 ms      |
| p95 latency | 3261 ms      |

### Comparison Bench

188 / 200 · **94.0%** · 0 errors · LongMemEval-S 500q retrieval harness

### Lightweight model validation

gemma3:270m — **18 / 20 · 90%** · 270M parameters, runs on a Pi

### Engineering history

100 iterations · 4 eras · bench-driven development throughout

---

## Negative controls — not benchmarks

Every result below is a knob we turn **off** that must collapse when the mechanism is removed.
If the number doesn't drop on demand, the lift was a coincidence. We ship the controls that have to fail.

> *"If you can't make the number drop on demand, you don't have evidence — you have a coincidence."*
>
> — Mazemaker testing protocol

| Scenario                 | Off → On       | What it proves                                                              |
| ------------------------ | :------------: | --------------------------------------------------------------------------- |
| Hop-2 graph reasoning    | 0.00 → **1.00** | A → B → C chains. Vanilla cosine cannot solve this by construction.         |
| Shuffled-edge control    | 1.00 → **0.27** | Collapse proves traversal is load-bearing, not the embedding model helping. |
| Post-dream synthesis     | 0.00 → **0.43** | Facts inferable only after consolidation become reachable after dream cycles. |
| Conflict supersession    | 0.03 → **0.33** | Newer contradictory facts supersede stale ones instead of duplicating noise. |
| Cross-session continuity | 0.06 → **0.62** | Concept-mode distractors pile up; the graph still holds continuity.         |
| Lean retrieval vs skynet | 0.42 → **0.60** | Lean beats skynet by +0.18 R@5 and drops dead-weight channels.              |

---

## Architecture

Mazemaker is a layered cognitive pipeline.

```
Conversation
    ↓
Atomic Fact Extraction (AFE)
    ↓
Semantic + graph encoding
    ↓
Hybrid retrieval + ColBERT rerank
    ↓
Dream consolidation
    ↓
Stage S synthesis crystallization
    ↓
Persistent cognitive graph
```

| Layer                   | Purpose                          |
| ----------------------- | -------------------------------- |
| Embeddings (BGE-M3)     | Semantic substrate               |
| ColBERT rerank          | Precision rerank                 |
| Personalized PageRank   | Graph traversal                  |
| Conflict supersession   | Stale-memory replacement         |
| Stage C synthesis       | Latent user-state extraction     |
| Stage S crystallization | Long-term abstraction formation  |

Full deep-dive: [`docs/architecture.md`](docs/architecture.md)

---

## The dream engine

Mazemaker runs autonomous background consolidation inspired by biological sleep.
Triggered after 600s idle, after 50 new memories, manually via tooling, or as a standalone daemon.

### NREM
* Replay 100 recent memories
* Spreading activation — strengthen edges that fired together (+0.05)
* Weaken inactive edges (−0.01), prune dead edges below 0.05

### REM
* Find 50 isolated memories
* Bridge to similar unconnected nodes
* Create weighted connections (`similarity × 0.3`)

### Insight
* Detect graph communities via BFS connected components
* Identify bridge nodes
* Materialize synthesized abstractions and cluster memories

Post-dream synthesis on facts unreachable from any single memory: **0.00 → 0.43 R@10**.
Memory gets denser, not noisier, every night.

Full reference: [`docs/dream-engine.md`](docs/dream-engine.md)

---

## The audit

The entire benchmark suite — including the negative controls that must fail — was submitted to GPT-5.5 via the codex CLI. Eight rounds. The first two rejected the suite outright. By round eight every concrete objection was closed by code change, not argument.

Round 8 verdict: **unconditional yes — no residual caveat.**

Every prompt and every verdict is committed verbatim in the repository.

---

## Federation

Pod-to-pod memory propagation over HTTP(S). Per-pair Bearer keys, public-prefix gate, five-minute tick. Works for a Tailscale pair, a hub-and-spoke team, or a WWW-scale mesh — same model throughout.

---

## Why p95 latency increased

The p95 rising above 3 seconds is not a regression. It is evidence that graph expansion, synthesis, rerank, recursive traversal, and adaptive retrieval are genuinely contributing.

Commodity retrieval systems do not jump to 3-second p95s. Cognitive systems do.

---

## Installation

### Managed (recommended)

```bash
curl -fsSL https://api.mazemaker.dev/install.sh | bash
```

Includes Postgres + pgvector, ColBERT rerank, dream worker, Architect UI, synthesis pipeline, and autonomous consolidation.

**Community Version stays free for forever.** No credit card. No quota gate. No trial countdown.

### Self-host

```bash
git clone https://github.com/itsXactlY/mazemaker
cd mazemaker
pip install -r requirements.txt
bash install.sh
```

---

## Community vs Pro

| Feature                 | Community | Pro |
| ----------------------- | :-------: | :-: |
| Hybrid recall           | ✅        | ✅  |
| NREM dream phase        | ✅        | ✅  |
| SQLite backend          | ✅        | ✅  |
| MCP tools               | ✅        | ✅  |
| ColBERT rerank          | ❌        | ✅  |
| REM dream phase         | ❌        | ✅  |
| Insight synthesis       | ❌        | ✅  |
| Autonomous dream-worker | ❌        | ✅  |
| Architect UI            | ❌        | ✅  |
| Postgres + pgvector     | ❌        | ✅  |
| Federation              | ❌        | ✅  |

Full tier table: [`docs/configuration.md#tier-gated-features`](docs/configuration.md#tier-gated-features)

---

## The Architect

Visual operator cockpit at [`architect.mazemaker.dev`](https://architect.mazemaker.dev/):

* Live graph topology
* Dream telemetry + replay
* Memory evolution timeline (chrono-scrub)
* Retrieval activation traces
* Synthesis activity
* Rerank inspection
* Graph communities
* Skill-indexing pipeline (the agent bridge's `/agent/skills/index`)

Hosted UI. Local data. Nothing crosses the loopback.

---

## Philosophy

> Intelligence without continuity is imitation.

Stateless agents simulate thought.
Persistent agents accumulate it.

Mazemaker exists to give AI systems something closer to memory, identity, continuity, and evolving internal structure.

Not just better search.

---

## Documentation

| Doc | Covers |
| --- | ------ |
| [`docs/architecture.md`](docs/architecture.md) | Six-layer cognition stack, embedding backends, retrieval pipeline, GPU recall, graph, schema |
| [`docs/configuration.md`](docs/configuration.md) | Every YAML knob, env var, retrieval-mode cheat sheet, tier-gated features, tuning recipes |
| [`docs/dream-engine.md`](docs/dream-engine.md) | NREM / REM / Insight / AFE / DAE / Synthesis — triggers, sampling, GPU acceleration, standalone daemon |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Inception Bench, LongMemEval-oracle, LongMemEval-S, Comparison Bench, the 100-iteration audit story, reproduction recipe |
| [`docs/inception-bench.md`](docs/inception-bench.md) | Why external rubrics were broken, the deterministic-judge methodology, the 12 scenarios |
| [`docs/mcp-tools.md`](docs/mcp-tools.md) | Nine tools, input/output JSON, integration shapes, quick-starts |
| [`docs/federation.md`](docs/federation.md) | Pod-to-pod propagation, Bearer keys, hub-and-spoke, mesh topology |
| [`docs/production-lessons.md`](docs/production-lessons.md) | Operator rules, benchmark-driven defaults, bench-noise discipline, patched-bug index |
| [`docs/changelog-beta.md`](docs/changelog-beta.md) | Official Beta release notes — the threshold, six layers, engineering deliverables |

---

## Links

* GitHub: https://github.com/itsXactlY/mazemaker
* Console: https://mazemaker.dev
* Architect: https://architect.mazemaker.dev
* Site: https://mazemaker.online

---

## License

AGPLv3 + PolyForm-NC dual license. Community engine remains open-source forever.

* [`LICENSE-AGPL-3.0.txt`](LICENSE-AGPL-3.0.txt) — community engine
* [`LICENSE-POLYFORM-NC-1.0.0.md`](LICENSE-POLYFORM-NC-1.0.0.md) — non-commercial use
* [`LICENSE`](LICENSE) — top-level summary
* [`NOTICE`](NOTICE) — attributions