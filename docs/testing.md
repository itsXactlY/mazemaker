# Testing

Three test surfaces, in increasing depth: a 10-second smoke, the full
Python suite, and the clean-VM verification matrix. Plus the
file-structure map for when you need to know where a thing lives.

> **If a single test should pass on every machine,** it's the
> [Clean Smoke Test](#clean-smoke-test). Run that first.

---

## Table of contents

1. [The 10-second smoke](#the-10-second-smoke)
2. [Full Python test suite](#full-python-test-suite)
3. [Clean Smoke Test](#clean-smoke-test)
4. [Verified clean VM — Debian 12](#verified-clean-vm--debian-12)
5. [VM / constrained-environment notes](#vm--constrained-environment-notes)
6. [File structure](#file-structure)

---

## The 10-second smoke

```bash
cd ~/projects/mazemaker/python
python3 demo.py
```

If this prints results without error, the engine is working.

---

## Full Python test suite

The plugin test suite (after `install.sh` deploys to Hermes):

```bash
cd ~/.hermes/hermes-agent/plugins/memory/neural
python3 test_suite.py
```

Run a subset by tag:

```bash
python3 test_suite.py --tags embed,memory
python3 test_suite.py --tags graph,dream
```

Available tags: `embed`, `memory`, `graph`, `dream`, `recall`,
`conflict`, `persistence`.

### Upside-Down Test Suite

Edge cases, corruption recovery, concurrency, SQL injection, malformed
input. Run when you're about to change something important.

```bash
cd ~/projects/mazemaker
python3 tests/test_upside_down.py
```

### C++ unit tests (if you built the optional `.so`)

```bash
cd build && ctest
```

The `.so` is optional — Python falls back gracefully when it's absent.

---

## Clean Smoke Test

Works on any machine, no Hermes dependency, no plugin install. Useful
for new VMs and CI.

```bash
cd ~/projects/mazemaker
python3 -c "
import sys; sys.path.insert(0, 'python')
from mazemaker import Mazemaker
nm = Mazemaker(db_path='/tmp/test.db', embedding_backend='cpu', use_cpp=False)
mid = nm.remember('test memory', label='smoke')
results = nm.recall('test')
assert len(results) > 0, 'recall failed'
print(f'SMOKE TEST PASS: {len(results)} results')
"
```

If this prints `SMOKE TEST PASS: 1 results` (or more), the engine
boots, the embedding backend loaded, SQLite opened, and recall found
the memory. Three problems would surface here:

- **Import error** → check `pip install -r requirements.txt`
- **Embedding backend fail** → FastEmbed missing or model download
  blocked; falls through to TF-IDF/hash and still works
- **recall returns 0** → `[SUPERSEDED]` marker / conflict-detector
  threshold misfire; rare on smoke content

---

## Verified clean VM — Debian 12

Tested on a fresh Debian 12 QEMU/KVM VM — hermes-agent + mazemaker only.

| Property        | Value                                 |
|-----------------|---------------------------------------|
| VM              | Debian 12, 4 GB RAM, KVM enabled      |
| hermes-agent    | git clone (itsXactlY fork)            |
| mazemaker       | git clone + FastEmbed ONNX            |
| Embedding       | intfloat/multilingual-e5-large (1024d)|
| C++ bridge      | Not built (Python fallback)           |

**All 12 integration tests passed:**

| # | Test                                            | Result |
|---|-------------------------------------------------|--------|
| 1 | Mazemaker standalone (remember/recall/graph)    | PASS   |
| 2 | Memory Provider (FastEmbed 1024d)               | PASS   |
| 3 | MemoryProvider.__init__                         | PASS   |
| 4 | is_available()                                  | PASS   |
| 5 | initialize(session_id)                          | PASS   |
| 6 | get_tool_schemas() → 4 tools                   | PASS   |
| 7 | system_prompt_block() (250 chars)               | PASS   |
| 8 | handle_tool_call — mazemaker_remember           | PASS   |
| 9 | handle_tool_call — mazemaker_recall             | PASS   |
| 10| handle_tool_call — mazemaker_graph              | PASS   |
| 11| prefetch()                                      | PASS   |
| 12| shutdown()                                      | PASS   |

---

## VM / constrained-environment notes

The gotchas from real installs on constrained hosts:

- **4 GB RAM minimum.** FastEmbed model download is ~500 MB; 2 GB
  systems get OOM-killed during the first init.
- **HashBackend** works as fallback on low-RAM systems. 1024-d output,
  instant, no deps. You lose semantic quality but recall still
  functions.
- **C++ bridge optional.** The Python fallback covers every code path
  the bridge ever did.
- **FastEmbed ≥ 0.5.1.** Earlier versions default to CLS embedding,
  which is deprecated by the model authors and silently produces
  worse vectors. Pin it.
- **`python3-venv` required on Debian.** `apt install python3.11-venv`
  if missing.
- **PEP 668 (Debian).** `pip install` needs a venv or
  `--break-system-packages`. The installer creates the venv for you;
  if you skip the installer, create one manually.
- **Cloud-init delay.** 60–90 s on first boot of many cloud images.
  Don't assume SSH is ready immediately.
- **`prefetch()` returns empty** on fresh DB — expected, no prior
  memories to pre-load.

---

## File structure

```
mazemaker/
├── install.sh                    Installer
├── hermes-plugin/                Plugin (deployed to hermes-agent)
│   ├── __init__.py               MemoryProvider + 4 tools
│   ├── config.py                 Config loader
│   ├── plugin.yaml               Plugin metadata
│   ├── memory_client.py          Main client (Mazemaker, SQLiteStore)
│   ├── embed_provider.py         Embedding backends
│   ├── gpu_recall.py             CUDA cosine similarity engine
│   ├── dream_engine.py           NREM / REM / Insight
│   ├── dream_worker.py           Standalone daemon
│   ├── access_logger.py          Recall event logger
│   └── ...
│
├── python/                       Source of truth (mirrors hermes-plugin)
│   ├── mazemaker.py              The Mazemaker class
│   ├── memory_client.py          SQLiteStore + NeuralMemory hot path
│   ├── postgres_store.py         Postgres + pgvector primary backend
│   ├── dream_postgres_store.py   PG dream backend
│   ├── colbert_helper.py         ColBERT late-interaction token extractor
│   ├── migrate_colbert_tokens.py One-shot backfill for existing memories
│   ├── afe.py                    Atomic Fact Extraction (Stage A/B/C)
│   ├── synthesis.py              Stage S synthesis
│   ├── dae.py                    Dream-Augmented Embeddings
│   ├── license.py                Ed25519 license verification + feature gates
│   └── ...
│
├── src/                          Optional C++ extras (legacy)
│   ├── memory/lstm.cpp           LSTM predictor
│   ├── memory/knn.cpp            kNN engine
│   └── memory/hopfield.cpp       Hopfield network
│
├── benchmarks/                   Inception Bench + audit
│   ├── README.md                 Headline numbers + reproduction recipe
│   ├── mazemaker_memory_bench.py     Pure-memory bench (12 deterministic scenarios)
│   ├── mazemaker_inception_bench.py  Inception Bench — full-corpus LongMemEval harness
│   ├── audit/                    Eight rounds of GPT-5.5 audit (verbatim)
│   ├── mazemaker_benchmark/      Internal capability suites
│   └── external/
│       ├── longmemeval_s.py      500q public retrieval benchmark
│       ├── comparison_bench.py   10 small LLMs, plain-text scoring
│       └── results/              Canonical reference JSONs
│
├── docs/                         You are here
│   ├── README.md                 Doc index + reading order
│   ├── architecture.md           Six-layer cognition stack
│   ├── configuration.md          Every knob, every env var
│   ├── dream-engine.md           NREM/REM/Insight/AFE/DAE/Synthesis
│   ├── benchmarks.md             Numbers + audit story
│   ├── inception-bench.md        Methodology + the 12 scenarios
│   ├── mcp-tools.md              Tool reference + quick-start
│   ├── testing.md                This file
│   ├── production-lessons.md     What we learned the hard way
│   └── changelog-beta.md         Official Beta changelog
│
├── README.md                     Project README (lean)
├── LICENSE                       AGPLv3 + PolyForm-NC dual
└── CLAUDE.md                     Agent-side guidance
```

---

## Going deeper

- **Architecture map** — [`architecture.md`](architecture.md)
- **Failure-mode + recovery recipes** — [`production-lessons.md`](production-lessons.md)
- **Bench reproduction** — [`benchmarks.md#reproducing-the-numbers`](benchmarks.md#reproducing-the-numbers)
