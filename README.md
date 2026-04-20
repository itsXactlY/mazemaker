# Neural Memory Adapter for Hermes Agent

Semantic memory system with knowledge graph, spreading activation, embedding-based recall, autonomous dream consolidation, and GPU-accelerated recall for the Hermes Agent.

<video src="https://github.com/itsXactlY/neural-memory/raw/refs/heads/master/assets/cover_video.mp4" controls width="100%"></video>

[![Dashboard](assets/neural_memory.png)](https://raw.githubusercontent.com/itsXactlY/neural-memory/refs/heads/master/assets/neural_memory.png)
## Features

- **Semantic Memory Storage**: Store memories with auto embedding (FastEmbed ONNX, 1024d)
- **Knowledge Graph**: Auto-connect related memories via cosine similarity
- **Spreading Activation**: Explore connected ideas (BFS graph traversal with decay)
- **Conflict Detection**: Detect and supersede conflicting memories
- **Dream Engine**: Autonomous consolidation (NREM/REM/Insight phases)
- **GPU Recall**: CUDA-accelerated cosine similarity (~100ms, torch.matmul)
- **SQLite-First**: Always works, no external DB needed
- **MSSQL Optional**: Shared DB for multi-agent setups

## Quick Start

```bash
cd ~/projects/neural-memory-adapter
bash install.sh          # auto-detect hermes-agent
bash install.sh /path    # explicit path
```

The installer handles everything:
1. Python deps (FastEmbed, torch, numpy)
2. CUDA detection for GPU recall
3. Plugin deployment to hermes-agent
4. Database init (SQLite at ~/.neural_memory/memory.db)
5. config.yaml setup

Restart hermes after install: `hermes gateway restart`

## Architecture

### Embedding Backends (auto-priority)

| Priority | Backend | Model | Speed | Requirements |
|----------|---------|-------|-------|--------------|
| 1st | FastEmbed | intfloat/multilingual-e5-large | ~50ms | `pip install fastembed` |
| 2nd | sentence-transformers | BAAI/bge-m3 1024d | ~200ms | GPU recommended |
| 3rd | tfidf | — | varies | numpy only |
| 4th | hash | — | instant | nothing |

FastEmbed uses ONNX runtime — no PyTorch conflict, works on CPU. Falls back automatically.

### GPU Recall Engine

```python
# gpu_recall.py — CUDA cosine similarity
# Loads all embeddings into GPU, does torch.matmul for batch similarity
# ~100ms for 10K memories vs ~500ms CPU

from gpu_recall import GPURecall
engine = GPURecall()
results = engine.recall(query_embedding, all_embeddings, top_k=10)
```

Auto-detects CUDA. Falls back to Python/numpy if no GPU.

### Data Flow

```mermaid
flowchart TD
    subgraph Store["neural_remember"]
        A[User content] --> B[FastEmbed encode]
        B --> C[1024d vector]
        C --> D[SQLite INSERT]
        C --> E[Cosine similarity search]
        E --> F[Create connections]
    end

    subgraph Recall["neural_recall"]
        G[User query] --> H[FastEmbed encode]
        H --> I{CUDA available?}
        I -->|Yes| J[GPU torch.matmul]
        I -->|No| K[CPU numpy dot]
        J --> L[Top-k results]
        K --> L
    end

    subgraph Think["neural_think"]
        M[Source memory] --> N[BFS on connections]
        N --> O[Apply decay factor]
        O --> P[Ranked activation]
    end

    subgraph Dream["neural_dream"]
        Q[Idle trigger] --> R[NREM replay]
        R --> S[REM bridge discovery]
        S --> T[Insight communities]
        T --> U[Consolidated graph]
    end
```

### Storage

- **SQLite (always)**: `~/.neural_memory/memory.db` — source of truth
- **Embeddings cache**: `~/.neural_memory/models/` (auto-downloaded, ~2.2 GB)
- **GPU cache**: `~/.neural_memory/gpu_cache/` (embeddings.npy + metadata.pkl)
- **Access logs**: `~/.neural_memory/access_logs/` (JSON Lines)
- **MSSQL (optional)**: 127.0.0.1/NeuralMemory — multi-agent mirror

### SQLite Schema

```sql
-- Core tables
memories (id, content, embedding, category, salience, ...)
connections (source_id, target_id, weight, edge_type)
connection_history (source_id, target_id, last_weight, last_updated)

-- Dream engine
dream_sessions (id, phase, started_at, completed_at, stats)
dream_insights (id, session_id, type, data)

-- Indexes
idx_memories_category ON memories(category)
idx_connections_source ON connections(source_id)
idx_connections_target ON connections(target_id)
```

## Configuration

All settings in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: neural
  neural:
    db_path: ~/.neural_memory/memory.db
    embedding_backend: fastembed       # auto | fastembed | sentence-transformers | tfidf | hash
    prefetch_limit: 10
    search_limit: 10
    dream:
      enabled: true
      idle_threshold: 600              # seconds before dream cycle
      memory_threshold: 50             # dream after N new memories
      mssql:                           # optional — only if using MSSQL
        server: 127.0.0.1
        database: NeuralMemory
        username: SA
        password: 'your_password'
        driver: '{ODBC Driver 18 for SQL Server}'
```

## Tools

When active, these tools are available in Hermes:

| Tool | Description |
|------|-------------|
| `neural_remember` | Store a memory (with conflict detection) |
| `neural_recall` | Search memories by semantic similarity |
| `neural_think` | Spreading activation from a memory |
| `neural_graph` | View knowledge graph statistics |
| `neural_dream` | Force a dream cycle (all/nrem/rem/insight) |
| `neural_dream_stats` | Dream engine statistics |

## Dream Engine

Autonomous background memory consolidation (biological sleep inspired):

```mermaid
flowchart LR
    subgraph Trigger
        T1[Idle 600s] --> D
        T2[50 new memories] --> D
        T3[Manual / Cron] --> D
    end

    D[Dream Cycle] --> NREM
    D --> REM
    D --> INSIGHT

    subgraph NREM["Phase 1 — NREM"]
        direction TB
        N1[Replay 100 recent memories] --> N2[Spreading activation]
        N2 --> N3{Connection active?}
        N3 -->|Yes| N4[Strengthen +0.05]
        N3 -->|No| N5[Weaken -0.01]
        N3 -->|Dead <0.05| N6[Prune]
    end

    subgraph REM["Phase 2 — REM"]
        direction TB
        R1[Find 50 isolated memories] --> R2[Search similar unconnected]
        R2 --> R3[Create bridge connections]
        R3 --> R4[weight = similarity × 0.3]
    end

    subgraph INSIGHT["Phase 3 — Insight"]
        direction TB
        I1[BFS connected components] --> I2[Identify communities]
        I2 --> I3[Find bridge nodes]
        I3 --> I4[Store dream_insights]
    end
```

### Triggers

- Automatic: after 600s idle (configurable)
- Automatic: every 50 new memories (configurable)
- Manual: `neural_dream` tool
- Standalone: `python python/dream_worker.py --daemon`

## Clean VM Test (2026-04-21)

Tested on fresh Debian 12 QEMU/KVM VM — hermes-agent + neural memory only, no jack-in-a-box.

```
VM:           Debian 12, 4GB RAM, KVM enabled
hermes-agent: git clone (itsXactlY fork)
neural-mem:   git clone + FastEmbed ONNX
Embedding:    intfloat/multilingual-e5-large (1024d)
C++ bridge:   Not built (Python fallback)
```

**All 12 tests passed:**

| # | Test | Result |
|---|------|--------|
| 1 | NeuralMemory standalone (remember/recall/graph) | PASS |
| 2 | Memory Provider (FastEmbed 1024d) | PASS |
| 3 | NeuralMemoryProvider.__init__ | PASS |
| 4 | is_available() | PASS |
| 5 | initialize(session_id) | PASS |
| 6 | get_tool_schemas() → 4 tools | PASS |
| 7 | system_prompt_block() (250 chars) | PASS |
| 8 | handle_tool_call — neural_remember | PASS |
| 9 | handle_tool_call — neural_recall | PASS |
| 10 | handle_tool_call — neural_graph | PASS |
| 11 | prefetch() | PASS |
| 12 | shutdown() | PASS |

**Findings:**
- **4GB RAM minimum** for FastEmbed model download (~500MB). 2GB = OOM killed.
- **HashBackend** works as fallback on low-RAM systems (1024d, instant, no deps).
- **C++ bridge optional** — Python fallback is fully functional.
- **FastEmbed >= 0.5.1** recommended — earlier versions use CLS embedding by default.
- **`python3-venv` required** on Debian — `apt install python3.11-venv` if missing.
- **`pip install --break-system-packages`** needed if no venv (PEP 668).
- Tool discovery shows 58 tools (missing `firecrawl`/`fal_client` optional deps).
- prefetch() returns empty on fresh DB (expected — no prior memories to pre-load).

## Testing

```bash
# Quick smoke test
cd ~/projects/neural-memory-adapter/python
python3 demo.py

# From plugin dir
cd ~/.hermes/hermes-agent/plugins/memory/neural
python3 test_suite.py

# Clean VM test (hermes-agent + neural memory only)
cd ~/projects/neural-memory-adapter
python3 -c "
import sys; sys.path.insert(0, 'python')
from neural_memory import NeuralMemory
nm = NeuralMemory(db_path='/tmp/test.db', embedding_backend='cpu', use_cpp=False)
mid = nm.remember('test memory', label='smoke')
results = nm.recall('test')
assert len(results) > 0, 'recall failed'
print(f'SMOKE TEST PASS: {len(results)} results')
"
```

## File Structure

```
neural-memory-adapter/
├── install.sh                    # Installer
├── hermes-plugin/                # Plugin (deployed to hermes-agent)
│   ├── __init__.py               # MemoryProvider + tools
│   ├── config.py                 # Config loader
│   ├── plugin.yaml               # Plugin metadata
│   ├── neural_memory.py          # Unified Memory class
│   ├── memory_client.py          # Main client (NeuralMemory, SQLiteStore)
│   ├── embed_provider.py         # Embedding backends (FastEmbed, st, tfidf, hash)
│   ├── gpu_recall.py             # CUDA cosine similarity engine
│   ├── dream_engine.py           # Dream engine (NREM/REM/Insight)
│   ├── dream_worker.py           # Standalone daemon
│   ├── access_logger.py          # Recall event logger
│   └── ...
├── python/                       # Python source (mirrors hermes-plugin)
│   └── ...
├── src/                          # C++ source (optional, legacy)
│   ├── memory/lstm.cpp           # LSTM predictor
│   ├── memory/knn.cpp            # kNN engine
│   └── memory/hopfield.cpp       # Hopfield network
└── README.md
```

## Lessons Learned (Production)

### Embedding & Runtime
- **FastEmbed > sentence-transformers** — ONNX, no PyTorch conflict, fast on CPU
- **FastEmbed >= 0.5.1** — earlier versions default to CLS embedding (deprecated). Set `add_custom_model` or pin version.
- **GPU recall > C++ Bridge** — C++ Hopfield was biased, GPU matmul is clean
- **Raw embeddings + GPU matmul** — best recall performance
- **Don't force PyTorch** — let FastEmbed handle CPU, torch only for GPU recall
- **numpy must be installed BEFORE FastEmbed** — FastEmbed depends on numpy at import time

### Storage & Architecture
- **SQLite = Source of Truth** — MSSQL optional, SQLite always works
- **Auto-detect everything** — CUDA, backends, venv paths

### Clean VM (2026-04-21)
- **4GB RAM minimum** for FastEmbed model download (~500MB). 2GB = OOM killed during model fetch.
- **HashBackend** is a valid fallback for constrained environments — 1024d, instant, zero deps.
- **C++ bridge is optional** — Python fallback covers all functionality. Only needed for perf.
- **python3-venv required** on Debian 12 — `apt install python3.11-venv` if cloud-init missed it.
- **Debian PEP 668** — `pip install` fails without venv or `--break-system-packages`.
- **Cloud-init takes 60-90s** — packages install on first boot. Don't assume SSH is ready.
- **`neural_remember` / `neural_recall` / `neural_think` / `neural_graph`** are the 4 tool schemas exposed by NeuralMemoryProvider. `neural_dream` and `neural_dream_stats` are in the standalone Memory class only.
- **prefetch() returns empty** on fresh DB — expected behavior, no prior memories to pre-load.
- **MemoryProvider interface** requires `initialize(session_id)` — not just `initialize()`.
- **hermes-agent tools** show 58 registered (missing `firecrawl`/`fal_client` optional deps).
