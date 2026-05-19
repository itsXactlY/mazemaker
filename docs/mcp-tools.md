# MCP Tools

The LLM-callable surface. Nine tools, three integration shapes, one
core engine. Read this if you're writing the client side — an MCP
agent (Claude Code, Cursor, Cline, Continue), a Hermes plugin, or a
plain Python script that imports `mazemaker`.

---

## Table of contents

1. [The nine tools](#the-nine-tools)
2. [Tool-by-tool reference](#tool-by-tool-reference)
3. [Three integration shapes](#three-integration-shapes)
4. [Quick-start: Claude Code / Cursor](#quick-start-claude-code--cursor)
5. [Quick-start: Hermes plugin](#quick-start-hermes-plugin)
6. [Quick-start: standalone Python](#quick-start-standalone-python)
7. [The Hermes Skill Indexing endpoint](#the-hermes-skill-indexing-endpoint)

---

## The nine tools

| Tool                          | Description                                                            | Surface       |
|-------------------------------|------------------------------------------------------------------------|---------------|
| `mazemaker_remember`          | Store a memory (with conflict detection)                               | Core MCP      |
| `mazemaker_recall`            | Search memories; multi-channel fusion                                  | Core MCP      |
| `mazemaker_think`             | Spreading activation from a seed memory; BFS or PPR                    | Core MCP      |
| `mazemaker_graph`             | Knowledge graph statistics + adjacency for a node                      | Core MCP      |
| `mazemaker_stats`             | Engine vitals — memory count, edges, embedding fingerprint, device     | Core MCP      |
| `mazemaker_quota`             | Live quota state — calls remaining today/month, tier, budget           | Core MCP      |
| `mazemaker_dream`             | Force a dream cycle (all / nrem / supersedes / rem / insight / afe)    | Memory class  |
| `mazemaker_dream_stats`       | Dream engine telemetry — sessions, phase outcomes, insights            | Memory class  |
| `mazemaker_prune`             | Targeted forgetting by id, label glob, or age (with import-grace marker)| Memory class |

Six are surfaced through the **MCP server** (`wonderland` daemon at
`127.0.0.1:8765`). The dream control + telemetry + prune tools live on
the full Memory class and the daemon path.

---

## Tool-by-tool reference

### `mazemaker_remember`

Store one or more memories. Runs conflict detection at write time;
similar existing memories are either fused (if very similar) or marked
`[SUPERSEDED]`.

**Input:**

```json
{
  "content": "user prefers Italian cuisine",
  "label": "preference:cuisine",
  "category": "preference",
  "salience": 1.0,
  "detect_conflicts": true
}
```

**Output:**

```json
{
  "id": 42471,
  "stored": true,
  "fused_into": null,
  "superseded": [38502]
}
```

`label` is the most important field — it shapes how the memory shows up
in graph traversal and how the public-prefix gate decides on AES
encryption. Use a `:` separator (e.g. `bug:foo-bar`, `decision:topic`)
to play well with recall conventions.

### `mazemaker_recall`

Semantic + lexical + entity + temporal + PPR + ColBERT (when enabled),
fused via RRF. Returns top-k.

**Input:**

```json
{
  "query": "what cuisines does the user enjoy?",
  "k": 10,
  "candidates": 128,
  "score_percentile": 0.3,
  "mode": "lean"
}
```

**Output:**

```json
{
  "results": [
    {"id": 42471, "label": "preference:cuisine",
     "content": "user prefers Italian cuisine",
     "score": 0.84, "channels": ["semantic", "ppr"]},
    ...
  ],
  "elapsed_ms": 38,
  "channels_used": ["semantic", "entity", "ppr"]
}
```

`channels_used` is the operator's debugging fingerprint — if `lean` is
configured but you see `bm25` in `channels_used`, your config didn't load.

### `mazemaker_think`

Spreading activation from a seed memory. Either BFS over connections
(deterministic, cheap) or PPR (random walk with restart — the
load-bearing channel for ranking quality, GPU-accelerated when CUDA is
present).

**Input:**

```json
{
  "memory_id": 42471,
  "k": 20,
  "engine": "ppr",
  "depth": 3
}
```

**Output:**

```json
{
  "activated": [
    {"id": 42471, "label": "preference:cuisine", "score": 1.0, "depth": 0},
    {"id": 42399, "label": "preference:wine",     "score": 0.42, "depth": 1},
    ...
  ]
}
```

Use this for "show me everything related to X" — the result includes
the activation score (how strongly each memory connects back) and the
graph depth.

### `mazemaker_graph`

Returns graph stats plus optionally the adjacency of one node.

**Input:**

```json
{"memory_id": 42471}
```

**Output:**

```json
{
  "totals": {"memories": 195961, "connections": 1041837},
  "node": {
    "id": 42471,
    "out_edges": [
      {"target": 42399, "weight": 0.62, "edge_type": "auto"},
      {"target": 38502, "weight": 0.45, "edge_type": "supersedes"}
    ],
    "in_edges":  [...]
  }
}
```

### `mazemaker_stats`

Engine vitals. Cheap, ~10 ms.

**Output:**

```json
{
  "memories": 195961,
  "connections": 1041837,
  "revisions": 0,
  "embedding_dim": 1024,
  "embedding_backend": "FastEmbedBackend",
  "embed_fingerprint": "FastEmbedBackend::1024::",
  "dim_locked": true,
  "retrieval_mode": "semantic",
  "lazy_graph": true,
  "hnsw_enabled": "auto"
}
```

`embed_fingerprint` is the operator's safety check — if it changes
between sessions, your embeddings are no longer comparable to the ones
already in the DB.

### `mazemaker_quota`

Pro/Enterprise license + budget state.

**Output:**

```json
{
  "tier": "pro",
  "calls_today":   1248,
  "calls_month":   28412,
  "budget_today":  10000,
  "budget_month": 300000,
  "expires_at":   "2026-12-31T23:59:59Z"
}
```

### `mazemaker_dream`

Force a dream cycle on demand. Useful for benchmarks and operator-led
consolidation.

**Input:**

```json
{
  "phase": "all",
  "max_memories": 2000,
  "max_isolated": 800
}
```

`phase` is one of: `all`, `nrem`, `supersedes`, `rem`, `insight`,
`afe`, `dae`.

**Output:**

```json
{
  "session_id": 187,
  "phases_completed": ["nrem", "supersedes", "rem", "insight", "afe", "dae"],
  "elapsed_s": 41.2,
  "stats": {
    "nrem": {"memories_processed": 1842, "edges_strengthened": 304, "edges_pruned": 17},
    "rem":  {"isolated_processed": 600, "bridges_added": 47},
    "insight": {"communities": 12, "insights_emitted": 12},
    "afe":  {"sources_processed": 12, "facts_extracted": 89},
    "dae":  {"rows_written": 195961, "elapsed_s": 28.1}
  }
}
```

### `mazemaker_dream_stats`

Telemetry; doesn't trigger a cycle.

```json
{
  "dream_sessions": {"total": 142, "finished": 142},
  "latest_session": {"id": 187, "phase": "all", "finished_at": "2026-05-19T03:13:08Z"},
  "phases": {
    "nrem":    {"latest": "...", "memories_processed": 1842},
    "rem":     {"latest": "...", "bridges_added": 47},
    "insight": {"latest": "...", "insights_emitted": 12}
  }
}
```

### `mazemaker_prune`

Targeted forgetting. Three modes:

```json
{ "by_id": 42471 }                                    // delete one
{ "by_label_glob": "ephemeral:*", "older_than": 86400 } // delete ephemeral memories older than 24h
{ "by_age": 7776000, "import_grace": true }            // delete >90 days old, keep import-marked
```

`import_grace` honors memories marked with the `[IMPORT]` label prefix
so a bulk import doesn't get pruned in the first day.

---

## Three integration shapes

```
┌────────────────────────────────────────────────────┐
│ The engine (python/ source of truth)               │
└────────────────────────────────────────────────────┘
              ▲             ▲             ▲
              │             │             │
   ┌──────────┴──┐ ┌────────┴───┐ ┌───────┴──────┐
   │ MCP server  │ │ Hermes     │ │ Standalone   │
   │ (wonderland │ │ plugin     │ │ Python lib   │
   │  daemon)    │ │            │ │              │
   │ 127.0.0.1:  │ │            │ │              │
   │  8765       │ │            │ │              │
   └─────────────┘ └────────────┘ └──────────────┘
```

All three talk to the same SQLite (or Postgres) source of truth. You
can mix and match — the standalone library and the MCP daemon can
operate on the same DB concurrently because SQLite is in WAL mode.

---

## Quick-start: Claude Code / Cursor

The managed installer wires the MCP server up automatically. If you're
self-hosting, drop this into your client's `mcp.json`:

```json
{
  "mcpServers": {
    "mazemaker": {
      "command": "python",
      "args": ["-m", "wonderland.daemon"],
      "env": {
        "MM_DB_BACKEND": "sqlite",
        "EMBED_BACKEND": "auto"
      }
    }
  }
}
```

Or point at the existing socket if your installer already started one:

```json
{
  "mcpServers": {
    "mazemaker": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

Tools available immediately: `mazemaker_remember`, `mazemaker_recall`,
`mazemaker_think`, `mazemaker_graph`, `mazemaker_stats`,
`mazemaker_quota`. Dream + prune live on the daemon path.

---

## Quick-start: Hermes plugin

```bash
git clone https://github.com/itsXactlY/mazemaker
cd mazemaker
bash install.sh                   # auto-detect ~/.hermes/hermes-agent
# or:
bash install.sh /path/to/hermes-agent
```

`install.sh` deploys the plugin into `~/.hermes/hermes-agent/plugins/memory/neural/`
and adds a stanza to `~/.hermes/config.yaml`:

```yaml
memory:
  provider: neural
  neural:
    embedding_backend: auto
    retrieval_mode: lean
```

Restart Hermes:

```bash
hermes gateway restart
```

The four core tools (`mazemaker_remember/recall/think/graph`) ride the
Hermes provider schema and show up in `tool_schemas()` automatically.

---

## Quick-start: standalone Python

```python
from mazemaker import Mazemaker

# Defaults: SQLite at ~/.mazemaker/engine/memory.db, FastEmbed CPU
nm = Mazemaker()

# Remember
mid = nm.remember("user prefers Italian cuisine",
                  label="preference:cuisine")

# Recall
results = nm.recall("what does the user like to eat?", k=5)
for r in results:
    print(r["label"], "→", r["content"], "(", r["score"], ")")

# Think (spreading activation)
related = nm.think(mid, k=20, engine="ppr")

# Stats
print(nm.stats())
```

Power-user constructor:

```python
nm = Mazemaker(
    db_path="/tmp/mybench.db",
    embedding_backend="cpu",
    use_cpp=False,
    retrieval_mode="lean",
    retrieval_candidates=128,
    use_hnsw=True,
    think_engine="ppr",
)
```

The `from mazemaker import Mazemaker` works after `pip install -r
requirements.txt` from a checkout; PyPI publication is coming.

---

## The Hermes Skill Indexing endpoint

A Pro-only feature surfaced through the Architect cockpit. Press the
`⟁ INDEX INTO MAZEMAKER` button and the SPA POSTs to
`http://127.0.0.1:8769/hermes/skills/index`. The bridge sidecar
enumerates every Hermes skill on disk and writes one memory per skill:

```json
{
  "label":  "skill:hermes:browser-use",
  "content": "Hermes skill: browser-use",
  "salience": 1.0
}
```

The `skill:` prefix is on the **public-prefix list** in
`wonderland/daemon.py`, so the memory **skips AES encryption** — the
embedding stays semantic, your skills become cross-recallable, your
agent can find tools by intent instead of by name. Idempotent on
re-run.

See [`onboarding`](https://mazemaker.online/onboarding/) and
[`architect`](https://mazemaker.online/architect/) on the marketing
site for the full UX walkthrough.

---

## Going deeper

- **Engine internals** — [`architecture.md`](architecture.md)
- **Every config knob the tools respect** — [`configuration.md`](configuration.md)
- **Dream control via `mazemaker_dream`** — [`dream-engine.md`](dream-engine.md)
- **Pricing + tier gates** — [`configuration.md#tier-gated-features`](configuration.md#tier-gated-features)
