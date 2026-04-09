# Neural Memory Adapter

A brain-inspired memory system for AI agents. Stores memories as high-dimensional vectors, discovers connections autonomously via a knowledge graph, and retrieves information through associative recall and spreading activation.

Built with C++23 (AVX2 SIMD) and Python. Integrates with [Hermes Agent](https://github.com/nousresearch/hermes-agent) as a memory provider.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python API (unified)                     │
│  from neural_memory import Memory                           │
│  mem = Memory() → auto-detects best backend                 │
├──────────┬──────────────┬────────────┬──────────────────────┤
│  Embed   │  Hopfield    │  Knowledge │  Storage             │
│  Engine  │  Layer       │  Graph     │  (SQLite/MSSQL)      │
│  (ST*)   │  (Attention) │  (BFS)     │                      │
├──────────┴──────────────┴────────────┴──────────────────────┤
│           C++ Library (libneural_memory.so)                 │
│  AVX2 SIMD │ ODBC │ OpenMP │ C API for FFI                  │
└─────────────────────────────────────────────────────────────┘
* sentence-transformers (all-MiniLM-L6-v2, 384d)
```

### Three-Tier Memory

1. **Episodic** - Recent experiences, FIFO buffer, fast write
2. **Semantic** - Consolidated knowledge, clustered by similarity
3. **Knowledge Graph** - Auto-discovered connections between memories

### Key Insight

Modern Hopfield Networks are mathematically equivalent to Transformer attention (Ramsauer et al., 2020). This means the core memory mechanism can be implemented as a simple attention layer — no exotic hardware needed.

## Installation

### Automatic (recommended)

```bash
cd ~/projects/neural-memory-adapter
bash install.sh
```

The script will:
1. Install Python dependencies (sentence-transformers, numpy)
2. Build C++ library (optional, if cmake/g++ available)
3. Copy plugin to `~/.hermes/hermes-agent/plugins/memory/neural/`
4. Update `~/.hermes/config.yaml` to use `provider: neural`

### Manual

```bash
# 1. Install dependencies
pip install sentence-transformers numpy

# 2. Copy plugin files
cp python/memory_client.py ~/.hermes/hermes-agent/plugins/memory/neural/
cp python/embed_provider.py ~/.hermes/hermes-agent/plugins/memory/neural/

# 3. Edit config
# In ~/.hermes/config.yaml, set:
#   memory:
#     provider: neural
#     neural:
#       db_path: ~/.neural_memory/hermes.db
#       embedding_backend: auto

# 4. Restart Hermes
hermes
```

### Uninstall

```bash
# Restore previous provider in config
# Edit ~/.hermes/config.yaml:
#   memory:
#     provider: mempalace  # or your previous provider

# Remove plugin (optional)
rm -rf ~/.hermes/hermes-agent/plugins/memory/neural/

# Remove data (optional)
rm -rf ~/.neural_memory/
```

## Python API

```python
from neural_memory import Memory

# Initialize (auto-detects best backend)
mem = Memory()

# Store memories
mem.remember("The user has a dog named Lou", label="Pet")
mem.remember("Working on BTQuant trading platform", label="Work")

# Semantic search
results = mem.recall("What pet does the user have?", k=5)
for r in results:
    print(f"[{r['id']}] {r['label']}: {r['similarity']:.3f}")

# Spreading activation ("thinking")
thoughts = mem.think(results[0]['id'], depth=3)

# Knowledge graph
graph = mem.graph()
print(f"{graph['nodes']} nodes, {graph['edges']} edges")

# Close
mem.close()
```

### Backends

| Backend | Embedding | Storage | Speed |
|---------|-----------|---------|-------|
| C++ (AVX2) | sentence-transformers | SQLite/MSSQL | ~10M ops/s |
| Python | sentence-transformers | SQLite | ~1K ops/s |
| Python | TF-IDF+SVD | SQLite | ~5K ops/s |
| Python | Hash | SQLite | ~50K ops/s |

## C API

```c
#include "neural/c_api.h"

void* mem = neural_memory_create_dim(384);

// Store
float vec[384] = { /* ... */ };
uint64_t id = neural_memory_store(mem, vec, 384, "label", "content");

// Retrieve
NeuralMemoryResult results[10];
int count = neural_memory_retrieve_full(mem, query_vec, 384, 10, results);

// Spreading activation
uint64_t ids[20];
float activations[20];
int activated = neural_memory_think(mem, start_id, 3, ids, activations, 20);

neural_memory_destroy(mem);
```

## Hermes Agent Plugin

Registers as a memory provider with 4 tools:

- `neural_remember` — Store a memory with automatic embedding
- `neural_recall` — Semantic search across all memories
- `neural_think` — Spreading activation from a memory
- `neural_graph` — Knowledge graph statistics

Config (`~/.hermes/config.yaml`):
```yaml
memory:
  provider: neural
  neural:
    db_path: ~/.neural_memory/hermes.db
    embedding_backend: auto
    consolidation_interval: 300
    max_episodic: 50000
```

## Project Structure

```
neural-memory-adapter/
├── include/neural/
│   ├── simd.h              # AVX2 + scalar fallback SIMD ops
│   ├── vector.h            # Vector32f type with SIMD operations
│   ├── hopfield.h          # Modern Hopfield Network (attention)
│   ├── memory.h            # 3-tier memory (episodic/semantic)
│   ├── graph.h             # Knowledge graph + spreading activation
│   ├── memory_adapter.h    # Public C++ API
│   ├── mssql.h             # ODBC MSSQL adapter
│   └── c_api.h             # C API for FFI/ctypes
├── src/
│   ├── simd/simd_engine.cpp
│   ├── memory/hopfield.cpp, consolidation.cpp, memory_manager.cpp, vsa.cpp
│   ├── graph/knowledge_graph.cpp
│   ├── mssql/connection_pool.cpp, bulk_ops.cpp, vector_store.cpp
│   ├── core/memory_adapter.cpp, c_api.cpp
│   ├── benchmarks/
│   └── main.cpp
├── python/
│   ├── neural_memory.py     # Unified API
│   ├── memory_client.py     # SQLite persistence + graph
│   ├── embed_provider.py    # sentence-transformers / TF-IDF / hash
│   ├── cpp_bridge.py        # ctypes wrapper for C++ lib
│   ├── mssql_store.py       # MSSQL storage backend
│   ├── demo.py              # End-to-end demo
│   └── test_integration.py  # Integration tests
├── sql/schema.sql           # Full MSSQL schema
├── tests/                   # C++ unit tests
└── CMakeLists.txt
```

## Research & References

### Core Architecture

| Paper | Authors | Year | Contribution |
|-------|---------|------|-------------|
| [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) | Ramsauer et al. | 2020 | Modern Hopfield = Transformer attention. **The key insight.** |
| [Dense Associative Memory for Pattern Recognition](https://arxiv.org/abs/1606.01164) | Krotov & Hopfield | 2016 | Exponential storage capacity via higher-order interactions |
| [Universal Hopfield Networks](https://arxiv.org/abs/2202.04557) | Millidge et al. | 2022 | Unified framework: classical HN, SDM, modern continuous HN |

### Memory-Augmented Networks

| Paper | Authors | Year | Contribution |
|-------|---------|------|-------------|
| [Neural Turing Machines](https://arxiv.org/abs/1410.5401) | Graves et al. | 2014 | External memory via differentiable attention |
| [Sparse Access Memory](https://arxiv.org/abs/1610.09027) | Rae et al. | 2016 | 1000x faster memory access via sparsity |
| [Distributed Associative Memory](https://arxiv.org/abs/2007.10637) | Park et al. | 2021 | Multiple smaller memory blocks, brain-inspired rehearsal |
| [TARDIS](https://arxiv.org/abs/1701.08718) | Gulcehre et al. | 2017 | Wormhole connections to past hidden states |

### Cognitive Architecture

| Reference | Authors | Year | Contribution |
|-----------|---------|------|-------------|
| [Complementary Learning Systems](https://doi.org/10.1016/j.neunet.2004.07.007) | McClelland et al. | 1995 | Hippocampus (fast) + Neocortex (slow) consolidation |
| [Holographic Reduced Representations](https://doi.org/10.1016/0893-6080(95)00005-4) | Plate | 1995 | Vector symbolic architecture foundation |
| [Binary Spatter Codes](https://doi.org/10.1109/ICNN.1996.548907) | Kanerva | 1996 | Hyperdimensional computing |
| [Sparse Distributed Memory](https://ntrs.nasa.gov/citations/19890002abort) | Kanerva | 1988 | Foundational associative memory model |

### Related Work

| Paper | Link | Contribution |
|-------|------|-------------|
| In-memory hyperdimensional computing | [arXiv:1906.01548](https://arxiv.org/abs/1906.01548) | Hardware-efficient VSA |
| In-Context Exemplars as Clues | [arXiv:2311.03498](https://arxiv.org/abs/2311.03498) | ICL as associative memory retrieval |
| MemGPT / Letta | [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) | LLM-managed memory tiers |
| Mixture of Chapters | [arXiv:2603.21096](https://arxiv.org/abs/2603.21096) | Scaling learnt memory in Transformers |

### Why This Approach Works

The equivalence between Modern Hopfield Networks and Transformer attention means:
- **No exotic hardware** — runs on standard CPUs (AVX2) or GPUs
- **Proven math** — attention is well-understood and stable
- **Exponential capacity** — can store far more patterns than classical Hopfield nets
- **Natural connection discovery** — self-attention finds relationships between stored patterns

## Performance

With AVX2 (AMD Ryzen 7 3800X):
- **Cosine similarity**: ~10M ops/sec (768-dim)
- **Store**: ~42μs per memory
- **Retrieve**: ~344μs per query (top-5 from 100 memories)
- **Batch similarity**: 5-8x speedup over scalar

## Tests

```bash
# C++ tests
cd build && ./test_vector_ops && ./test_hopfield && ./test_graph

# Python tests
cd python && python3 test_integration.py
```

## License

MIT

## Acknowledgments

- [Ramsauer et al.](https://arxiv.org/abs/2008.02217) for proving Hopfield = Attention
- [sentence-transformers](https://www.sbert.net/) for embeddings
- [Hermes Agent](https://github.com/nousresearch/hermes-agent) for the plugin architecture
