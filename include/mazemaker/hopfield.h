// include/neural/hopfield.h - Hopfield / Modern Associative Memory
#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <mutex>
#include "simd.h"

namespace neural {

// Forward declaration
namespace memory { class MemoryManager; }

struct HopfieldConfig {
    size_t dimensions = 512;        // Pattern vector dimension
    size_t capacity = 1024;         // Max stored patterns
    float beta = 20.0f;             // Temperature (inverse) - sharpens attention
    float learning_rate = 0.01f;    // For online updates
    float decay_rate = 0.999f;      // Salience decay per access cycle
    size_t cleanup_threshold = 64;  // Evict when patterns > capacity - threshold
};

struct Pattern {
    std::vector<float> data;        // The pattern vector (dimensions long)
    uint64_t id = 0;                // Unique identifier
    uint64_t timestamp = 0;         // Creation time (us since epoch)
    uint64_t access_count = 0;      // Number of retrievals
    float salience = 1.0f;          // Importance score (recency + frequency)
    std::string label;              // Optional semantic label
    std::string source;             // Origin (episodic, semantic, etc.)
};

struct RetrievalResult {
    std::vector<float> pattern;     // The completed/retrieved pattern
    float confidence = 0.0f;        // Cosine similarity to query
    size_t pattern_id = 0;          // ID of closest stored pattern
    float entropy = 0.0f;           // Attention distribution entropy
    bool converged = false;         // Whether iterative retrieval converged
    size_t iterations = 0;          // Iterations to convergence
};

struct BatchRetrievalResult {
    std::vector<RetrievalResult> results;
    float mean_confidence = 0.0f;
};

class HopfieldLayer {
public:
    explicit HopfieldLayer(const HopfieldConfig& config = {});
    ~HopfieldLayer();

    // --- Core operations ---

    // Store a pattern in memory. Returns assigned pattern ID.
    // Evicts least-salient pattern if at capacity.
    uint64_t store(const std::vector<float>& pattern, const std::string& label = "",
                   const std::string& source = "episodic");

    // Retrieve pattern completion from partial/noisy cue.
    // Runs iterative attention: xi_new = sum_j softmax(beta * sim(xi,xj)) * xj
    RetrievalResult retrieve(const std::vector<float>& cue, size_t max_iterations = 10,
                             float convergence_eps = 1e-4f) const;

    // Batch retrieval: multiple queries in parallel
    BatchRetrievalResult retrieve_batch(const std::vector<std::vector<float>>& cues,
                                        size_t max_iterations = 10) const;

    // Direct similarity lookup (no iteration)
    RetrievalResult lookup(const std::vector<float>& query) const;

    // --- Capacity management ---

    // Evict lowest-salience pattern
    void evict();

    // Update salience of all patterns (decay + boost accessed)
    void update_salience();

    // Get pattern by ID
    const Pattern* get_pattern(uint64_t id) const;

    // Remove pattern by ID
    bool remove(uint64_t id);

    // --- Queries ---

    size_t pattern_count() const { return patterns_.size(); }
    size_t dimension() const { return config_.dimensions; }
    bool is_full() const { return patterns_.size() >= config_.capacity; }
    float occupancy() const { return static_cast<float>(patterns_.size()) / config_.capacity; }

    // Get all pattern IDs
    std::vector<uint64_t> pattern_ids() const;

    // Similarity matrix between all stored patterns (attention matrix)
    // Returns NxN matrix (flat, row-major)
    std::vector<float> attention_matrix() const;

    // Find K most similar patterns to a query
    std::vector<std::pair<uint64_t, float>> top_k(const std::vector<float>& query, size_t k) const;

    // --- Configuration ---

    void set_beta(float beta) { config_.beta = beta; }
    float get_beta() const { return config_.beta; }
    const HopfieldConfig& config() const { return config_; }

private:
    // Internal: compute softmax attention weights for a query against all patterns
    // Returns vector of weights (size = pattern_count)
    std::vector<float> attention_weights(const float* query) const;

    // Internal: compute attention-weighted sum of stored patterns
    void attention_sum(const float* query, float* output) const;

    // Internal: softmax over similarity scores
    std::vector<float> softmax(const std::vector<float>& scores) const;

    void evict_internal();  // Internal eviction (called with mutex held)

    HopfieldConfig config_;
    std::vector<Pattern> patterns_;
    uint64_t next_id_ = 1;
    mutable std::mutex mutex_;

    // Scratch buffers (pre-allocated)
    mutable std::vector<float> scratch_sims_;
    mutable std::vector<float> scratch_weights_;
};

} // namespace neural
