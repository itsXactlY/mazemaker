---
name: neural-dream-engine
category: devops
description: Autonomous background memory consolidation for Mazemaker — NREM/REM/Insight phases
---

# Neural Dream Engine

Autonomous background memory consolidation for the Mazemaker plugin.
Inspired by biological sleep: NREM → REM → Insight phases.

## Architecture

Three phases running as a background daemon during idle periods:

**Phase 1 — NREM (Replay & Consolidation)**
- Replays recent memories via spreading activation (`neural_think`)
- Activated connections: weight +0.05
- Non-activated: weight -0.01
- Connections below 0.05: pruned

**Phase 2 — REM (Exploration & Bridge Discovery)**
- Finds isolated memories (few connections)
- Searches via embedding similarity for unconnected but semantically similar memories
- Creates tentative bridge connections (weight 0.1-0.3)

**Phase 3 — Insight (Abstraction & Community Detection)**
- Connected component analysis (BFS) to find communities
- Identifies bridge nodes connecting communities
- Creates insight entries for clusters >= 3 memories

## Files

```
plugins/memory/neural/
├── dream_engine.py        # Core engine + SQLite backend
├── dream_mssql_store.py   # MSSQL backend (pyodbc) — CASE not MIN/MAX for MSSQL
├── dream_worker.py        # Standalone full-stack worker (MSSQL + sentence-transformers)
├── __init__.py            # Plugin integration (tools, hooks)
tests/plugins/memory/
└── test_dream_engine.py   # 23 tests
```

## Standalone Worker (Full Stack)

```bash
# One-shot dream cycle
python plugins/memory/neural/dream_worker.py

# Single phase
python plugins/memory/neural/dream_worker.py --phase nrem

# Background daemon (dreams after 5min idle)
python plugins/memory/neural/dream_worker.py --daemon --idle 300
```

Results (tested on 918 memories, 40K connections):
- NREM: 100 memories, 400 connections strengthened (~11s)
- REM: 50 isolated explored, 134 bridges created (~4s)
- Insight: 15 communities, 3 insights (~0.1s)
- Total: ~15s for full cycle

## Tools

- `neural_dream [phase]` — Force a dream cycle (all/nrem/rem/insight)
- `neural_dream_stats` — Get engine statistics

## Triggers

- After 5 minutes idle (configurable via `dream.idle_threshold`)
- Every 50 new memories (configurable via `dream.memory_threshold`)
- Manually via `neural_dream` tool

## Config (config.yaml)

```yaml
memory:
  neural:
    dream:
      enabled: true
      idle_threshold: 300
      memory_threshold: 50
      mssql:
        server: localhost      # Optional: override MSSQL_SERVER env
        database: Mazemaker # Optional: override MSSQL_DATABASE env
```

Credentials go in `~/.hermes/.env` — NEVER hardcode passwords:
```
MSSQL_SERVER=localhost
MSSQL_DATABASE=Mazemaker
MSSQL_USERNAME=SA
MSSQL_PASSWORD=your_password
MSSQL_DRIVER={ODBC Driver 18 for SQL Server}
```

Resolution order: OS env vars > .env file > config.yaml > defaults.

## MSSQL Pitfalls

### MIN/MAX are Aggregate Functions in MSSQL
MSSQL `MIN()`/`MAX()` are aggregate functions, NOT scalar like SQLite/Python.
Use `CASE WHEN` for clamping weights:

```sql
-- WRONG (MSSQL error: "The MAX function requires 1 argument")
UPDATE connections SET weight = MAX(weight - 0.01, 0.0) WHERE ...

-- CORRECT
UPDATE connections SET weight = CASE
    WHEN weight - 0.01 < 0.0 THEN 0.0
    ELSE weight - 0.01
END WHERE ...
```

### NREM Performance on Large Datasets (40K+ connections)
Don't load ALL connections into memory. Instead:
1. Batch-embed the recent memories
2. For each memory, query only its connections from DB
3. Compare embedding similarity against the connection endpoints
4. Strengthen only activated edges

This keeps memory usage O(batch_size) instead of O(total_connections).

### Model Caching — Always Use `cache_folder` + `local_files_only`

The `embed_provider.py` and `dream_worker.py` must use the SAME cache path.
Without `cache_folder`, sentence-transformers uses the default HF cache
(`~/.cache/huggingface/`) and may re-download.

```python
MODEL_DIR = Path.home() / ".neural_memory" / "models"

# Check if cached
cached = MODEL_DIR / f"models--sentence-transformers--{model_name}"
is_cached = cached.exists()

model = SentenceTransformer(
    model_name,
    cache_folder=str(MODEL_DIR),      # shared cache
    local_files_only=is_cached,        # no re-download
)
```

Also use a class-level `_shared_model` singleton so multiple `EmbedProvider`
instances reuse the same loaded model (avoids loading ~400MB twice).

## MSSQL Backend

When `dream.mssql` is configured, the dream engine uses a shared MSSQL
database instead of SQLite. This enables multi-agent consolidation where
multiple agents can dream independently on the same knowledge graph.

Falls back to SQLite if MSSQL is unavailable.

## Installer Pattern (Lite vs Full Stack)

Two-mode installer for different environments:

- **Lite** (`install.sh --lite`): SQLite + hash/tfidf, ~50MB RAM, no GPU/Docker
- **Full Stack** (`install.sh --full`): SQLite + MSSQL + sentence-transformers, ~500MB RAM

Interactive menu if no argument given. `install_database.sh` handles DB setup
(separate from plugin install).

## .env Credentials Pattern

Both `mssql_store.py` and `dream_mssql_store.py` use a lightweight `.env`
loader (no python-dotenv dependency). The `_load_dotenv()` function reads
from multiple paths in order:

```python
_dotenv = _load_dotenv([
    ".env",                                # CWD
    str(Path.home() / ".hermes" / ".env"), # ~/.hermes/.env
    str(Path(__file__).parent / ".env"),   # plugin dir
])

def _env(key: str, fallback: str = "") -> str:
    return os.environ.get(key) or _dotenv.get(key, fallback)
```

Passwords are resolved as: `explicit arg > OS env > .env > config > empty`.

## Cron Integration

```bash
# Every 6 hours
cronjob create --name neural-dream-mssql --schedule "0 */6 * * *" \
  --prompt "Run: python ~/plugins/memory/neural/dream_worker.py"
```
