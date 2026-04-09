// neural/core/c_api.cpp - C-compatible API implementation
// Wraps NeuralMemoryAdapter for use via ctypes / FFI.
#include "neural/c_api.h"
#include "neural/memory_adapter.h"
#include <cstring>
#include <new>

using namespace neural;

// Helper: convert handle to typed pointer
static inline NeuralMemoryAdapter* to_adapter(NeuralMemoryHandle h) {
    return static_cast<NeuralMemoryAdapter*>(h);
}

// ============================================================================
// Lifecycle
// ============================================================================

NEURAL_API NeuralMemoryHandle neural_memory_create(void) {
    return neural_memory_create_dim(384);
}

NEURAL_API NeuralMemoryHandle neural_memory_create_dim(int vector_dim) {
    if (vector_dim <= 0) vector_dim = 384;

    auto* adapter = new (std::nothrow) NeuralMemoryAdapter();
    if (!adapter) return nullptr;

    AdapterConfig config;
    config.vector_dim = static_cast<size_t>(vector_dim);
    // Disable background threads for Python use (Python manages its own lifecycle)
    config.enable_consolidation_thread = false;
    config.enable_decay_thread = false;
    config.enable_link_prediction = false;
    // Disable MSSQL (Python client uses SQLite)
    config.db_config.server = "";  // Will cause mssql init to be a no-op

    if (!adapter->initialize(config)) {
        delete adapter;
        return nullptr;
    }

    return static_cast<NeuralMemoryHandle>(adapter);
}

NEURAL_API void neural_memory_destroy(NeuralMemoryHandle handle) {
    if (!handle) return;
    auto* adapter = to_adapter(handle);
    adapter->shutdown();
    delete adapter;
}

// ============================================================================
// Core operations
// ============================================================================

NEURAL_API uint64_t neural_memory_store(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    const char* label,
    const char* content
) {
    if (!handle || !vec || dim <= 0) return 0;
    auto* adapter = to_adapter(handle);

    std::vector<float> embedding(vec, vec + dim);
    std::string lbl = label ? label : "";
    std::string cnt = content ? content : "";

    return adapter->store(embedding, lbl, cnt, "api");
}

NEURAL_API uint64_t neural_memory_store_text(
    NeuralMemoryHandle handle,
    const char* text,
    const char* label
) {
    if (!handle || !text) return 0;
    auto* adapter = to_adapter(handle);

    std::string txt = text;
    std::string lbl = label ? label : "";

    return adapter->store_text(txt, lbl);
}

NEURAL_API int neural_memory_retrieve(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    int k,
    uint64_t* ids,
    float* scores
) {
    if (!handle || !vec || dim <= 0 || k <= 0 || !ids || !scores) return 0;
    auto* adapter = to_adapter(handle);

    std::vector<float> cue(vec, vec + dim);
    auto results = adapter->retrieve(cue, static_cast<size_t>(k));

    int count = 0;
    for (const auto& r : results) {
        if (count >= k) break;
        ids[count] = r.id;
        scores[count] = r.similarity;
        count++;
    }
    return count;
}

NEURAL_API int neural_memory_retrieve_full(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    int k,
    NeuralMemoryResult* results
) {
    if (!handle || !vec || dim <= 0 || k <= 0 || !results) return 0;
    auto* adapter = to_adapter(handle);

    std::vector<float> cue(vec, vec + dim);
    auto mem_results = adapter->retrieve(cue, static_cast<size_t>(k));

    int count = 0;
    for (const auto& r : mem_results) {
        if (count >= k) break;
        auto& out = results[count];
        out.id = r.id;
        out.embedding = const_cast<float*>(r.embedding.data());
        out.embedding_dim = static_cast<int>(r.embedding.size());

        // Safe string copy
        std::strncpy(out.label, r.label.c_str(), sizeof(out.label) - 1);
        out.label[sizeof(out.label) - 1] = '\0';
        std::strncpy(out.content, r.content.c_str(), sizeof(out.content) - 1);
        out.content[sizeof(out.content) - 1] = '\0';

        out.similarity = r.similarity;
        out.salience = r.salience;
        count++;
    }
    return count;
}

NEURAL_API int neural_memory_search(
    NeuralMemoryHandle handle,
    const char* query,
    int k,
    uint64_t* ids,
    float* scores
) {
    if (!handle || !query || k <= 0 || !ids || !scores) return 0;
    auto* adapter = to_adapter(handle);

    std::string q = query;
    auto results = adapter->search(q, static_cast<size_t>(k));

    int count = 0;
    for (const auto& r : results) {
        if (count >= k) break;
        ids[count] = r.id;
        scores[count] = r.similarity;
        count++;
    }
    return count;
}

NEURAL_API int neural_memory_read(
    NeuralMemoryHandle handle,
    uint64_t id,
    NeuralMemoryResult* result
) {
    if (!handle || !result) return 0;
    auto* adapter = to_adapter(handle);

    auto opt = adapter->read(id);
    if (!opt) return 0;

    const auto& r = *opt;
    result->id = r.id;
    result->embedding = const_cast<float*>(r.embedding.data());
    result->embedding_dim = static_cast<int>(r.embedding.size());

    std::strncpy(result->label, r.label.c_str(), sizeof(result->label) - 1);
    result->label[sizeof(result->label) - 1] = '\0';
    std::strncpy(result->content, r.content.c_str(), sizeof(result->content) - 1);
    result->content[sizeof(result->content) - 1] = '\0';

    result->similarity = r.similarity;
    result->salience = r.salience;
    return 1;
}

// ============================================================================
// Graph / Spreading Activation
// ============================================================================

NEURAL_API int neural_memory_think(
    NeuralMemoryHandle handle,
    uint64_t start_id,
    int depth,
    uint64_t* node_ids,
    float* activations,
    int max_results
) {
    if (!handle || !node_ids || !activations || max_results <= 0) return 0;
    auto* adapter = to_adapter(handle);

    auto traversal = adapter->think(start_id, depth);

    int count = 0;
    for (const auto& tr : traversal) {
        if (count >= max_results) break;
        // Convert node_id back to memory_id (subtract 1 offset)
        node_ids[count] = tr.node_id > 0 ? tr.node_id - 1 : tr.node_id;
        activations[count] = tr.activation;
        count++;
    }
    return count;
}

// ============================================================================
// Consolidation & Decay
// ============================================================================

NEURAL_API size_t neural_memory_consolidate(NeuralMemoryHandle handle) {
    if (!handle) return 0;
    return to_adapter(handle)->consolidate();
}

NEURAL_API void neural_memory_decay(NeuralMemoryHandle handle) {
    if (!handle) return;
    to_adapter(handle)->decay();
}

NEURAL_API size_t neural_memory_predict_links(NeuralMemoryHandle handle) {
    if (!handle) return 0;
    return to_adapter(handle)->predict_links();
}

// ============================================================================
// Configuration
// ============================================================================

NEURAL_API void neural_memory_set_beta(NeuralMemoryHandle handle, float beta) {
    if (!handle) return;
    to_adapter(handle)->set_beta(beta);
}

NEURAL_API float neural_memory_get_beta(NeuralMemoryHandle handle) {
    if (!handle) return 0.0f;
    return to_adapter(handle)->get_beta();
}

NEURAL_API void neural_memory_set_consolidation_threshold(NeuralMemoryHandle handle, float threshold) {
    if (!handle) return;
    to_adapter(handle)->set_consolidation_threshold(threshold);
}

// ============================================================================
// Statistics
// ============================================================================

NEURAL_API void neural_memory_stats(NeuralMemoryHandle handle, NeuralMemoryStats* stats) {
    if (!handle || !stats) return;
    auto* adapter = to_adapter(handle);

    auto s = adapter->get_stats();

    stats->episodic_count = s.episodic_count;
    stats->semantic_count = s.semantic_count;
    stats->episodic_occupancy = s.episodic_occupancy;
    stats->semantic_occupancy = s.semantic_occupancy;
    stats->hopfield_patterns = s.hopfield_patterns;
    stats->hopfield_occupancy = s.hopfield_occupancy;
    stats->graph_nodes = s.graph_nodes;
    stats->graph_edges = s.graph_edges;
    stats->graph_density = s.graph_density;
    stats->avg_store_us = s.avg_store_us;
    stats->avg_retrieve_us = s.avg_retrieve_us;
    stats->avg_search_us = s.avg_search_us;
    stats->total_stores = s.total_stores;
    stats->total_retrieves = s.total_retrieves;
    stats->total_searches = s.total_searches;
    stats->total_consolidations = s.total_consolidations;
}
