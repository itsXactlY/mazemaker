# Mazemaker — Persistent Cognition Infrastructure for AI Agents

> **Give your AI a memory that actually sticks.**
>
> Mazemaker turns stateless assistants into persistent cognitive systems:
> memory formation, graph reasoning, synthesis, dream consolidation, and cross-session continuity — locally, privately, and MCP-native.

![Neural Brain Hero](assets/neural_brain_hero.png)

---

# What this is

Most AI "memory" systems are retrieval wrappers.

They store chunks.
Embed text.
Run cosine similarity.
Return vaguely related paragraphs.

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
And it survives across sessions.

---

# Why people care

With one install, your assistant can:

* 🧠 Remember user preferences across weeks or months
* 🔗 Connect related ideas automatically
* ⏳ Track evolving facts over time
* 😴 Run autonomous "dream cycles" that strengthen useful memories
* ✏️ Replace stale information instead of duplicating it
* 🧩 Infer higher-level patterns from fragmented conversations
* 🔍 Explain *why* a memory surfaced
* 🔒 Keep memory local-first and encrypted

This is not "chat history."

It is persistent cognition infrastructure.

---

# The result

## LongMemEval Oracle Harness — iter97

500-question full-corpus oracle benchmark.

| Metric      |     iter97 |
| ----------- | ---------: |
| Recall@1    | **0.6255** |
| Recall@5    | **0.8340** |
| Recall@10   | **0.9000** |
| MRR         | **0.7124** |
| p50 latency |    1728 ms |
| p95 latency |    3261 ms |

### Per-question-type breakdown

| Question Type             |        R@5 |
| ------------------------- | ---------: |
| knowledge-update          | **0.8750** |
| multi-session             | **0.8595** |
| single-session-assistant  | **0.9286** |
| single-session-preference | **0.6667** |
| single-session-user       | **0.9375** |
| temporal-reasoning        | **0.7323** |

---

# Why these numbers matter

The important result is not the aggregate score.

The important result is *what improved.*

The system crossed a threshold where:

* preference memory started behaving coherently,
* temporal reasoning stabilized,
* synthesized memories became useful,
* graph traversal stopped acting like retrieval glue and started acting like cognitive structure.

The biggest breakthrough was:

| Metric                        | Before |      After |
| ----------------------------- | -----: | ---------: |
| single-session-preference R@5 | 0.2333 | **0.6667** |

That is a ~186% relative improvement.

Why?

Because the engine stopped trying to retrieve exact wording and started synthesizing latent user-state abstractions.

Not:

> "retrieve what was said"

But:

> "infer what the conversation means"

That distinction changes everything.

Humans do not remember transcripts.
They remember distilled abstractions.

Mazemaker now approximates that behaviour.

---

# The architecture

Mazemaker is a layered cognitive pipeline.

```text
Conversation
    ↓
Atomic Fact Extraction (AFE)
    ↓
Semantic + graph encoding
    ↓
Hybrid retrieval + rerank
    ↓
Dream consolidation
    ↓
Synthesis crystallization
    ↓
Persistent cognitive graph
```

The current stack:

| Layer                   | Purpose                         |
| ----------------------- | ------------------------------- |
| Embeddings (BGE-M3)     | semantic substrate              |
| ColBERT rerank          | precision rerank                |
| Personalized PageRank   | graph traversal                 |
| Conflict supersession   | stale-memory replacement        |
| Stage C synthesis       | latent user-state extraction    |
| Stage S crystallization | long-term abstraction formation |

The benchmark loop showed something unexpected:

> Synthesis mattered more than retrieval tuning.

That is the architectural transition.

Full engine deep-dive: [`docs/architecture.md`](docs/architecture.md).

---

# The dream system

Mazemaker runs autonomous background consolidation inspired by biological sleep.

## NREM

* replay memories
* strengthen useful paths
* weaken weak edges
* prune dead structure

## REM

* discover bridge memories
* connect isolated clusters
* generate associative structure

## Insight

* detect graph communities
* materialize synthesized abstractions
* form higher-level memory clusters

This is where the system becomes more than a vector database.

The topology itself evolves.

Full dream-cycle reference: [`docs/dream-engine.md`](docs/dream-engine.md).

---

# Why p95 latency increased

The p95 latency rising above 3 seconds was not a regression.

It was evidence that:

* graph expansion,
* synthesis,
* rerank,
* recursive traversal,
* and adaptive retrieval

were genuinely contributing.

Commodity retrieval systems do not suddenly jump to 3-second p95s.

Cognitive systems do.

Optimization comes later.

First the architecture has to become alive.

---

# The benchmark story

One of the biggest discoveries during development was that many published memory benchmarks are deeply unstable.

We measured:

* rubric defects,
* judge drift,
* inconsistent grading,
* and benchmark leakage.

So Mazemaker ships its own deterministic benchmark methodology:

## Inception Benchmarking

Properties:

* deterministic scoring
* no LLM judge
* no JSON schema dependence
* canonical gold labels
* reproducible end-to-end

Scenarios include:

* needle-in-haystack recall
* temporal chains
* distractor resistance
* graph traversal
* conflict supersession
* dream-derived inference
* latency scaling

The benchmark harness lives in:

```bash
benchmarks/mazemaker_memory_bench.py
```

Methodology, the 12 scenarios, and full numbers: [`docs/inception-bench.md`](docs/inception-bench.md).

---

# Installation

## Managed install (recommended)

```bash
curl -fsSL https://api.mazemaker.dev/install.sh | bash
```

Includes:

* Postgres + pgvector
* ColBERT rerank
* dream worker
* Architect UI
* synthesis pipeline
* autonomous consolidation

**Free for the entire beta.** No credit card, no quota gate, no "trial" countdown.

---

## Community self-host

```bash
git clone https://github.com/itsXactlY/mazemaker
cd mazemaker
pip install -r requirements.txt
bash install.sh
```

---

# Community vs Pro

| Feature                 | Community | Pro |
| ----------------------- | --------- | --- |
| Hybrid recall           | ✅         | ✅   |
| NREM dream phase        | ✅         | ✅   |
| SQLite backend          | ✅         | ✅   |
| MCP tools               | ✅         | ✅   |
| ColBERT rerank          | ❌         | ✅   |
| REM dream phase         | ❌         | ✅   |
| Insight synthesis       | ❌         | ✅   |
| Autonomous dream-worker | ❌         | ✅   |
| Architect UI            | ❌         | ✅   |
| Postgres + pgvector     | ❌         | ✅   |

Full tier table + per-feature notes: [`docs/configuration.md#tier-gated-features`](docs/configuration.md#tier-gated-features).

---

# The Architect

Mazemaker ships with a visual operator cockpit:

* live graph topology,
* dream telemetry,
* memory evolution,
* retrieval traces,
* synthesis activity,
* rerank inspection,
* graph communities.

Inspired by *The Matrix Reloaded* operator aesthetic.

Walkthrough + 12-monitor map: [architect.mazemaker.dev](https://architect.mazemaker.dev/) · [page deep-dive](https://mazemaker.online/architect/).

---

# What category is this?

Not:

* vector database
* retrieval wrapper
* semantic search plugin

Mazemaker crossed into a different category:

> Persistent cognition infrastructure.

The moat is no longer embeddings.

The moat is:

* memory formation,
* synthesis quality,
* graph evolution,
* consolidation,
* temporal continuity,
* adaptive recall,
* and cognitive persistence.

---

# Current frontier

The benchmark loop exposed the next unlock:

| Current           | Next                       |
| ----------------- | -------------------------- |
| static retrieval  | adaptive routing           |
| generic traversal | edge-type-aware traversal  |
| fixed recall      | intent-conditioned recall  |
| static synthesis  | confidence-aware synthesis |

The path toward R@5 > 0.90 is now visible:

* query-intent classification,
* episodic vs semantic retrieval modes,
* edge-conditioned graph traversal,
* adaptive consolidation,
* reinforcement-weighted persistence.

---

# Philosophy

The core realization behind Mazemaker is simple:

> Intelligence without continuity is imitation.

Stateless agents simulate thought.
Persistent agents accumulate it.

Mazemaker exists to give AI systems something closer to:

* memory,
* identity,
* continuity,
* and evolving internal structure.

Not just better search.

---

# Documentation

The full documentation lives in [`docs/`](docs/). Pick the path that
matches what you're trying to accomplish:

| Doc                                                       | What it covers                                                          |
| --------------------------------------------------------- | ----------------------------------------------------------------------- |
| [`docs/architecture.md`](docs/architecture.md)            | Six-layer cognition stack, embedding backends, retrieval pipeline, GPU recall, graph, schema |
| [`docs/configuration.md`](docs/configuration.md)          | Every YAML knob, every env var, retrieval-mode cheat sheet, tier-gated features, tuning recipes |
| [`docs/dream-engine.md`](docs/dream-engine.md)            | NREM / REM / Insight / AFE / DAE / Synthesis — triggers, sampling, GPU acceleration, standalone daemon |
| [`docs/benchmarks.md`](docs/benchmarks.md)                | Inception Bench, LongMemEval-oracle, LongMemEval-S, Comparison Bench, the eight-round audit story, reproduction recipe |
| [`docs/inception-bench.md`](docs/inception-bench.md)      | Why external rubrics were broken, the deterministic-judge methodology, the 12 scenarios |
| [`docs/mcp-tools.md`](docs/mcp-tools.md)                  | Nine tools, input/output JSON, integration shapes, quick-starts        |
| [`docs/testing.md`](docs/testing.md)                      | Smoke test, full suite, clean-VM verification, file structure          |
| [`docs/production-lessons.md`](docs/production-lessons.md)| Operator rules, benchmark-driven defaults, bench-noise discipline, external-audit handling, patched-bug index |
| [`docs/changelog-beta.md`](docs/changelog-beta.md)        | Official Beta release notes — the threshold, six layers, engineering deliverables, the 1.6 GB claim-evidence bundle |

Start at [`docs/README.md`](docs/README.md) for the suggested reading order.

---

# Repository

* GitHub: https://github.com/itsXactlY/mazemaker
* Managed endpoint: https://api.mazemaker.dev
* Console: https://mazemaker.dev
* Marketing surface: https://mazemaker.online
* Contact: `info@mazemaker.dev` · `enterprise@mazemaker.dev` · `privacy@mazemaker.dev`

---

# License

AGPLv3 + PolyForm-NC dual license. Community engine remains
open-source forever.

* [`LICENSE-AGPL-3.0.txt`](LICENSE-AGPL-3.0.txt) — community engine
* [`LICENSE-POLYFORM-NC-1.0.0.md`](LICENSE-POLYFORM-NC-1.0.0.md) — non-commercial commercial use
* [`LICENSE`](LICENSE) — top-level summary
* [`NOTICE`](NOTICE) — attributions
