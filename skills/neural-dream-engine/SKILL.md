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

### MSSQL Pitfall: MIN()/MAX() Are Aggregates

MSSQL rejects `MIN(weight + 0.05, 1.0)` — MIN/MAX are aggregate functions.
Use CASE WHEN instead:
```sql
UPDATE connections SET weight = CASE
  WHEN weight + 0.05 > 1.0 THEN 1.0 ELSE weight + 0.05 END
```

### Credentials

Never hardcode. Resolution: env vars → ~/.hermes/.env → config → defaults.
```bash
# ~/.hermes/.env
MSSQL_SERVER=localhost
MSSQL_DATABASE=Mazemaker
MSSQL_USERNAME=SA
MSSQL_PASSWORD=your_password
```

### Pitfalls

- **MSSQL `MIN()/MAX()` are aggregate, not scalar** — Use `CASE WHEN weight + 0.05 > 1.0 THEN 1.0 ELSE weight + 0.05 END` instead of `MIN(weight + 0.05, 1.0)`. SQLite has scalar MIN/MAX, MSSQL does not.
- **Thread safety in SQLiteStore.get_all()** — Must use `self._lock` in `get_all()` not just `store()`. Concurrent reads during writes return None blobs causing `TypeError: object of type 'NoneType' has no len()`.
- **`_initial_context` consumed by prefetch()** — Don't clear `_initial_context` in `prefetch()`. Keep it available for both `system_prompt_block()` and `prefetch()` user message injection. Double-injection (system prompt + user message) is intentional.
- **C++ build fails without ODBC** — Add `#ifdef USE_MSSQL` guards in headers/source. CMakeLists.txt needs `option(USE_MSSQL ...)` with conditional source list and link flags.
- **Credentials: .env not hardcoded** — `_env('MSSQL_PASSWORD')` with fallback chain: OS env > .env > config.yaml > defaults. (learned the hard way)

**MSSQL has no scalar MIN/MAX** — `MIN(col + 0.05, 1.0)` is an AGGREGATE in MSSQL,
not scalar like SQLite. Use CASE WHEN instead:
```sql
-- WRONG (MSSQL): UPDATE t SET weight = MIN(weight + 0.05, 1.0)
-- RIGHT (MSSQL): UPDATE t SET weight = CASE WHEN weight + 0.05 > 1.0 THEN 1.0 ELSE weight + 0.05 END
-- RIGHT (SQLite): UPDATE t SET weight = MIN(weight + 0.05, 1.0)
```

**Model caching** — sentence-transformers defaults to HF hub cache (~/.cache).
Must explicitly set `cache_folder=str(MODEL_DIR)` and `local_files_only=is_cached`
to use the project cache at `~/.neural_memory/models/`. Otherwise re-downloads every time.
```python
cached = MODEL_DIR / f"models--sentence-transformers--{self.MODEL_NAME}"
model = SentenceTransformer(
    self.MODEL_NAME,
    cache_folder=str(MODEL_DIR),
    local_files_only=cached.exists(),
)
```

**Credentials** — Never hardcode passwords. Use .env loader (no python-dotenv dependency):
```python
_dotenv = _load_dotenv([".env", str(Path.home() / ".hermes" / ".env")])
def _env(key, fallback=""): return os.environ.get(key) or _dotenv.get(key, fallback)
password = _env('MSSQL_PASSWORD', '')
```

**Batch updates** — Don't do individual SQL UPDATE for 40K connections.
Batch them in a loop with cursor.execute per row + one commit at the end.

**Standalone worker** — dream_worker.py needs its own EmbedProvider with the same
cache path as embed_provider.py. Shared class variable `_shared_model` prevents
loading the model twice in the same process.

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

## SQLite Thread Safety

When using `check_same_thread=False` for concurrent access, ALL methods that
touch the connection need the lock — not just writes. The `get_all()` method
was missing the lock, causing `NoneType has no len()` errors during concurrent
read/write.

```python
def get_all(self):
    with self._lock:  # MUST have lock even for reads
        rows = self.conn.execute(...).fetchall()
    results = []
    for row in rows:
        if blob is None:  # null guard for race conditions
            continue
```

## Conflict Detection + Hash Backend Pitfall

The `detect_conflicts` feature (merging similar memories) interacts badly with
hash embeddings — hash generates similar vectors for structurally similar text,
causing unintended merges in batch operations.

Fix for stress tests: `m.remember(text, label, detect_conflicts=False)`
In production with sentence-transformers, conflict detection works correctly.

## Cron Integration

```bash
# Every 6 hours
cronjob create --name neural-dream-mssql --schedule "0 */6 * * *" \
  --prompt "Run: python ~/plugins/memory/neural/dream_worker.py"
```

## C++ SIMD Bridge (recall <1ms)

The C++ bridge (`cpp_bridge.py` → `libmazemaker.so`) provides SIMD-accelerated
retrieval. Wire it into Mazemaker with `use_cpp=True`:

```python
m = Mazemaker(db_path=DB, use_cpp=True)
# C++ index loaded at init from all stored memories
# recall() uses C++ retrieve(k*3) then applies temporal scoring on small set
```

### CSearchResult Struct MUST Match C Header

The Python `ctypes.Structure` MUST exactly match `MazemakerResult` in `c_api.h`:

```c
// c_api.h
typedef struct {
    uint64_t id;
    float*   embedding;      // 8 bytes on 64-bit
    int      embedding_dim;  // 4 bytes
    char     label[256];
    char     content[4096];  // NOT 1024!
    float    similarity;
    float    salience;
} MazemakerResult;
```

```python
# cpp_bridge.py — MUST match exactly or segfault
class CSearchResult(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("embedding", ctypes.POINTER(ctypes.c_float)),
        ("embedding_dim", ctypes.c_int),
        ("label", ctypes.c_char * 256),
        ("content", ctypes.c_char * 4096),   # was 1024 — WRONG, caused buffer overflow
        ("similarity", ctypes.c_float),
        ("salience", ctypes.c_float),
    ]
```

Field name change: C struct uses `similarity` not `score`. Update Python access accordingly.

## Cython fast_ops (66x cosine_similarity)

```python
# fast_ops.pyx — compiled with: python setup_fast.py build_ext --inplace
from fast_ops import cosine_similarity, batch_cosine_similarity

# 0.8us/call vs 53us Python = 66x faster
# batch_cosine_similarity(query, matrix): 0.4ms for 918x384 vectors
```

Drop-in integration in `memory_client.py`:
```python
class Mazemaker:
    try:
        from fast_ops import cosine_similarity as _cosine_sim_fast
    except ImportError:
        _cosine_sim_fast = None

    @staticmethod
    def _cosine_similarity(a, b):
        if Mazemaker._cosine_sim_fast is not None:
            import numpy as np
            if not isinstance(a, np.ndarray):
                a = np.asarray(a, dtype=np.float64)
            if not isinstance(b, np.ndarray):
                b = np.asarray(b, dtype=np.float64)
            return float(Mazemaker._cosine_sim_fast(a, b))
        # Python fallback
        dot = sum(x*y for x, y in zip(a, b))
        ...
```

Avoid repeated `np.asarray()` calls by storing embeddings as numpy arrays in the graph.

## MSSQL IPv4 vs IPv6 Pitfall

On Arch Linux (and some Ubuntu configs), `localhost` resolves to `::1` (IPv6) first.
MSSQL listens on `127.0.0.1` (IPv4) only → login timeout.

**Fix:** Use `127.0.0.1` instead of `localhost` in all defaults:
```python
server = server or _env('MSSQL_SERVER', '127.0.0.1')  # NOT 'localhost'
```

And in `.env`:
```
MSSQL_SERVER=127.0.0.1
```

This cost ~30min of debugging. The error message "Login timeout expired" is misleading —
it's a connection issue, not an auth issue.

## Gemma-4 Reasoning Model (llama-server)

Gemma-4 via llama-server uses "thinking" mode by default. All output goes to
`reasoning_content`, `content` stays empty. Benchmark gets 0% accuracy.

**Fix:** Set `reasoning_budget: -1` in API call to force content output:
```python
resp = client.chat.completions.create(
    model="GPT",
    messages=[...],
    extra_body={"reasoning_budget": -1}  # forces content alongside reasoning
)
```

Without this: 55% → 100% tokens spent on CoT, 0% on content.
With this: reasoning capped, content produced → proper benchmark results.

Server-side: `--reasoning-budget 512` caps at 512 tokens CoT.
Client-side: `reasoning_budget: -1` further forces content output.

## Full Stack Benchmark Harness

`benchmarks/bench_neural.py` uses the complete Mazemaker stack:

```bash
python bench_neural.py --dataset mmlu_pro --mssql    # MSSQL + C++ + Cython
python bench_neural.py --dataset mmlu_pro             # SQLite + C++ + Cython
python bench_neural.py --dataset mmlu_pro --no-memory  # Baseline
python bench_neural.py --all --mssql                   # All datasets
```

Stack: Mazemaker(use_mssql=True, use_cpp=True, embedding_backend="sentence-transformers")
- sentence-transformers CUDA for embeddings
- C++ SIMD bridge for recall (<1ms)
- Cython fast_ops for cosine_similarity (66x)
- llama-server (Gemma-4) as LLM judge (temp=0.0, seed=42)

## Initial Context Double-Injection

`_initial_context` must NOT be consumed by `prefetch()`. Keep it available for both:
1. `system_prompt_block()` — system prompt injection
2. `prefetch()` — user message injection (build_memory_context_block)

```python
def prefetch(self, query, **kwargs):
    if not result and self._initial_context:
        # DON'T consume: self._initial_context = ""
        return f"## Mazemaker Context\n{self._initial_context}"
```

Double-injection is intentional — model sees context in both places.

## SQLite Never Bench, MSSQL as Primary

SQLiteStore stays as fallback but benchmarks and production should prefer MSSQL:
- MSSQL is shared with Dream Engine (consolidation happens on same DB)
- C++ index loads from whatever store is configured
- Dream cron consolidates MSSQL overnight, C++ index rebuilds on next start

Architecture:
```
MSSQL (persist) ←→ Dream Engine (overnight consolidation)
       ↓
C++ SIMD Index (RAM) ←→ recall() <1ms
       ↓
SQLite (fallback, never bench)
```
