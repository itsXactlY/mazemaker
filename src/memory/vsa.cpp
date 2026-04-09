// src/memory/vsa.cpp - Vector Symbolic Architecture (VSA)
// Enables compositional representations through binding, bundling, permutation
#include "neural/simd.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <mutex>
#include <stdexcept>

namespace neural::vsa {

// ============================================================
// Core VSA Operations
// ============================================================

// Binding: element-wise multiply (Hadamard product)
// Encodes associations between concepts: A * B -> AB
// Properties: A * B = B * A (commutative), A * A^-1 = 1 (inverse)
// For bipolar vectors (+1/-1), A * A = 1 (self-inverse)
void bind(const float* a, const float* b, float* result, size_t n) {
    simd::hadamard(a, b, result, n);
}

// Binding with output as new vector
std::vector<float> bind(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("VSA bind: dimension mismatch");
    }
    std::vector<float> result(a.size());
    simd::hadamard(a.data(), b.data(), result.data(), a.size());
    return result;
}

// Bundling: element-wise add (creates superposition)
// Encodes sets / collections: A + B -> {A, B}
// Result is a noisy version of constituent vectors
void bundle(const float* a, const float* b, float* result, size_t n) {
    simd::add(a, b, result, n);
}

// Bundle multiple vectors
std::vector<float> bundle(const std::vector<std::vector<float>>& vectors) {
    if (vectors.empty()) return {};
    size_t n = vectors[0].size();
    std::vector<float> result(n, 0.0f);

    for (const auto& v : vectors) {
        if (v.size() != n) {
            throw std::invalid_argument("VSA bundle: dimension mismatch");
        }
        simd::add(result.data(), v.data(), result.data(), n);
    }

    // Normalize to maintain magnitude
    float norm = simd::l2_norm(result.data(), n);
    if (norm > 1e-10f) {
        simd::scale(result.data(), 1.0f / norm, result.data(), n);
    }

    return result;
}

// Permutation: circular shift (rotation)
// Encodes sequences / order: rho(A, B) != rho(B, A)
// rho^n(A) rotates A by n positions
void permute(const float* input, float* result, size_t n, int shift) {
    if (shift == 0) {
        simd::copy(input, result, n);
        return;
    }

    // Handle negative shifts
    int s = shift % static_cast<int>(n);
    if (s < 0) s += static_cast<int>(n);

    for (size_t i = 0; i < n; ++i) {
        result[(i + s) % n] = input[i];
    }
}

// Inverse permutation (shift in opposite direction)
void unpermute(const float* input, float* result, size_t n, int shift) {
    permute(input, result, n, -shift);
}

// Sequence encoding: encode ordered list [A, B, C]
// Uses positional encoding: rho^0(A) + rho^1(B) + rho^2(C)
std::vector<float> encode_sequence(const std::vector<std::vector<float>>& sequence) {
    if (sequence.empty()) return {};
    size_t n = sequence[0].size();
    std::vector<float> result(n, 0.0f);
    std::vector<float> rotated(n);

    for (size_t pos = 0; pos < sequence.size(); ++pos) {
        permute(sequence[pos].data(), rotated.data(), n, static_cast<int>(pos));
        simd::add(result.data(), rotated.data(), result.data(), n);
    }

    // Normalize
    float norm = simd::l2_norm(result.data(), n);
    if (norm > 1e-10f) {
        simd::scale(result.data(), 1.0f / norm, result.data(), n);
    }

    return result;
}

// Decode sequence position: given sequence encoding and item, find position
// by trying successive inverse permutations
int decode_sequence_position(const std::vector<float>& sequence_encoding,
                              const std::vector<float>& item, size_t n_positions) {
    size_t n = sequence_encoding.size();
    float best_sim = -1.0f;
    int best_pos = -1;

    std::vector<float> unrotated(n);
    for (size_t pos = 0; pos < n_positions; ++pos) {
        unpermute(sequence_encoding.data(), unrotated.data(), n, static_cast<int>(pos));
        float sim = simd::cosine_similarity(unrotated.data(), item.data(), n);
        if (sim > best_sim) {
            best_sim = sim;
            best_pos = static_cast<int>(pos);
        }
    }

    return best_pos;
}

// ============================================================
// Cleanup Memory - find nearest stored pattern
// ============================================================
class CleanupMemory {
public:
    explicit CleanupMemory(size_t dimensions) : dimensions_(dimensions) {}

    // Store a clean vector with its label
    void store(const std::string& label, const std::vector<float>& vector) {
        if (vector.size() != dimensions_) {
            throw std::invalid_argument("CleanupMemory: dimension mismatch");
        }
        patterns_[label] = vector;
    }

    // Clean up a noisy vector: find nearest stored pattern
    std::string cleanup(const float* noisy, float* cleaned = nullptr) const {
        if (patterns_.empty()) return "";

        std::string best_label;
        float best_sim = -1.0f;

        for (const auto& [label, pattern] : patterns_) {
            float sim = simd::cosine_similarity(noisy, pattern.data(), dimensions_);
            if (sim > best_sim) {
                best_sim = sim;
                best_label = label;
            }
        }

        if (cleaned && !best_label.empty()) {
            simd::copy(patterns_.at(best_label).data(), cleaned, dimensions_);
        }

        return best_label;
    }

    // Cleanup returning the vector
    std::pair<std::string, std::vector<float>> cleanup_vector(const std::vector<float>& noisy) const {
        std::vector<float> cleaned(dimensions_);
        std::string label = cleanup(noisy.data(), cleaned.data());
        return {label, cleaned};
    }

    // Find top-k nearest patterns
    std::vector<std::pair<std::string, float>> top_k(const float* query, size_t k) const {
        std::vector<std::pair<std::string, float>> scored;
        scored.reserve(patterns_.size());

        for (const auto& [label, pattern] : patterns_) {
            float sim = simd::cosine_similarity(query, pattern.data(), dimensions_);
            scored.emplace_back(label, sim);
        }

        std::partial_sort(scored.begin(),
                          scored.begin() + std::min(k, scored.size()),
                          scored.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });

        if (scored.size() > k) scored.resize(k);
        return scored;
    }

    size_t size() const { return patterns_.size(); }
    bool empty() const { return patterns_.empty(); }

    // Get stored pattern by label
    const std::vector<float>* get(const std::string& label) const {
        auto it = patterns_.find(label);
        if (it == patterns_.end()) return nullptr;
        return &it->second;
    }

private:
    size_t dimensions_;
    std::unordered_map<std::string, std::vector<float>> patterns_;
};

// ============================================================
// Random Vector Generator (for creating basis vectors)
// ============================================================
class VectorFactory {
public:
    explicit VectorFactory(size_t dimensions, uint64_t seed = 42)
        : dimensions_(dimensions), rng_(seed) {}

    // Generate a random bipolar vector (+1/-1)
    std::vector<float> random_bipolar() {
        std::vector<float> v(dimensions_);
        std::bernoulli_distribution dist(0.5);
        for (auto& x : v) {
            x = dist(rng_) ? 1.0f : -1.0f;
        }
        return v;
    }

    // Generate a random unit vector (Gaussian -> normalize)
    std::vector<float> random_unit() {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> v(dimensions_);
        for (auto& x : v) {
            x = dist(rng_);
        }
        float norm = simd::l2_norm(v.data(), dimensions_);
        if (norm > 1e-10f) {
            simd::scale(v.data(), 1.0f / norm, v.data(), dimensions_);
        }
        return v;
    }

    // Generate a nearly-orthogonal set of vectors
    // Uses iterative orthogonalization
    std::vector<std::vector<float>> orthogonal_set(size_t count) {
        std::vector<std::vector<float>> set;
        set.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            auto v = random_unit();

            // Gram-Schmidt orthogonalization against existing vectors
            for (const auto& existing : set) {
                float proj = simd::dot_product(v.data(), existing.data(), dimensions_);
                // v -= proj * existing
                simd::weighted_add(existing.data(), -proj, v.data(), dimensions_);
            }

            // Re-normalize
            float norm = simd::l2_norm(v.data(), dimensions_);
            if (norm > 1e-10f) {
                simd::scale(v.data(), 1.0f / norm, v.data(), dimensions_);
            }

            set.push_back(std::move(v));
        }

        return set;
    }

    size_t dimensions() const { return dimensions_; }

private:
    size_t dimensions_;
    std::mt19937_64 rng_;
};

// ============================================================
// VSA Framework - ties it all together
// ============================================================
class VSAEngine {
public:
    explicit VSAEngine(size_t dimensions = 512, uint64_t seed = 42)
        : dimensions_(dimensions)
        , factory_(dimensions, seed)
        , cleanup_(dimensions) {}

    // Create a named symbol (basis vector)
    std::vector<float> create_symbol(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = symbols_.find(name);
        if (it != symbols_.end()) {
            return it->second;
        }
        auto vec = factory_.random_bipolar();
        symbols_[name] = vec;
        cleanup_.store(name, vec);
        return vec;
    }

    // Bind two symbols
    std::vector<float> bind_symbols(const std::string& a, const std::string& b) {
        auto va = create_symbol(a);
        auto vb = create_symbol(b);
        return bind(va, vb);
    }

    // Bundle symbols into a set
    std::vector<float> bundle_symbols(const std::vector<std::string>& names) {
        std::vector<std::vector<float>> vectors;
        vectors.reserve(names.size());
        for (const auto& name : names) {
            vectors.push_back(create_symbol(name));
        }
        return bundle(vectors);
    }

    // Encode key-value pair: role * filler
    std::vector<float> encode_kv(const std::string& role, const std::vector<float>& filler) {
        auto role_vec = create_symbol(role);
        return bind(role_vec, filler);
    }

    // Decode key-value: retrieve filler for a role
    std::vector<float> decode_kv(const std::vector<float>& memory, const std::string& role) {
        auto role_vec = create_symbol(role);
        // Unbind by multiplying with role (self-inverse for bipolar)
        return bind(memory, role_vec);
    }

    // Cleanup a vector
    std::string cleanup(const std::vector<float>& noisy, std::vector<float>* cleaned = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cleaned) {
            cleaned->resize(dimensions_);
            return cleanup_.cleanup(noisy.data(), cleaned->data());
        }
        return cleanup_.cleanup(noisy.data());
    }

    // Measure similarity between two vectors (wrapper)
    float similarity(const std::vector<float>& a, const std::vector<float>& b) const {
        return simd::cosine_similarity(a.data(), b.data(), dimensions_);
    }

    size_t dimensions() const { return dimensions_; }
    size_t symbol_count() const { std::lock_guard<std::mutex> lock(mutex_); return symbols_.size(); }

private:
    size_t dimensions_;
    VectorFactory factory_;
    CleanupMemory cleanup_;
    std::unordered_map<std::string, std::vector<float>> symbols_;
    mutable std::mutex mutex_;
};

} // namespace neural::vsa
