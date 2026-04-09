// src/memory/hopfield.cpp - Modern Hopfield Network (Transformer Attention)
#include "neural/hopfield.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>
#include <cstring>

namespace neural {

HopfieldLayer::HopfieldLayer(const HopfieldConfig& config)
    : config_(config)
    , scratch_sims_(config.capacity)
    , scratch_weights_(config.capacity)
{
    patterns_.reserve(config.capacity);
}

HopfieldLayer::~HopfieldLayer() = default;

// ---- Store ----
uint64_t HopfieldLayer::store(const std::vector<float>& pattern, const std::string& label,
                               const std::string& source) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pattern.size() != config_.dimensions) {
        throw std::invalid_argument("Pattern dimension mismatch: expected " +
            std::to_string(config_.dimensions) + ", got " + std::to_string(pattern.size()));
    }

    // Evict if at capacity
    while (patterns_.size() >= config_.capacity) {
        evict_internal();
    }

    Pattern p;
    p.data = pattern;
    p.id = next_id_++;
    p.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    p.access_count = 0;
    p.salience = 1.0f;
    p.label = label;
    p.source = source;

    patterns_.push_back(std::move(p));
    return patterns_.back().id;
}

// ---- Attention computation ----
std::vector<float> HopfieldLayer::attention_weights(const float* query) const {
    const size_t n = patterns_.size();
    if (n == 0) return {};

    // Compute similarities: sim(query, pattern_i) for all i
    std::vector<float> scores(n);
    for (size_t i = 0; i < n; ++i) {
        // Cosine similarity via SIMD
        scores[i] = simd::cosine_similarity(query, patterns_[i].data.data(), config_.dimensions);
    }

    // Scale by beta
    for (size_t i = 0; i < n; ++i) {
        scores[i] *= config_.beta;
    }

    // Softmax
    return softmax(scores);
}

std::vector<float> HopfieldLayer::softmax(const std::vector<float>& scores) const {
    if (scores.empty()) return {};

    // Numerical stability: subtract max
    float max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<float> weights(scores.size());

    float sum = 0.0f;
    for (size_t i = 0; i < scores.size(); ++i) {
        weights[i] = std::exp(scores[i] - max_score);
        sum += weights[i];
    }

    float inv_sum = 1.0f / (sum + 1e-10f);
    for (auto& w : weights) {
        w *= inv_sum;
    }

    return weights;
}

void HopfieldLayer::attention_sum(const float* query, float* output) const {
    const size_t n = patterns_.size();
    if (n == 0) {
        simd::zero(output, config_.dimensions);
        return;
    }

    auto weights = attention_weights(query);

    // output = sum_j weights[j] * pattern_j
    // This is the transformer attention weighted sum
    simd::zero(output, config_.dimensions);
    for (size_t j = 0; j < n; ++j) {
        if (weights[j] > 1e-8f) {
            simd::weighted_add(patterns_[j].data.data(), weights[j], output, config_.dimensions);
        }
    }
}

// ---- Retrieve (iterative attention = pattern completion) ----
RetrievalResult HopfieldLayer::retrieve(const std::vector<float>& cue, size_t max_iterations,
                                          float convergence_eps) const {
    if (cue.size() != config_.dimensions) {
        throw std::invalid_argument("Cue dimension mismatch");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    RetrievalResult result;
    result.pattern.resize(config_.dimensions);
    result.converged = false;
    result.iterations = 0;

    if (patterns_.empty()) {
        std::memcpy(result.pattern.data(), cue.data(), config_.dimensions * sizeof(float));
        return result;
    }

    // Initialize with the cue
    std::vector<float> current(config_.dimensions);
    std::memcpy(current.data(), cue.data(), config_.dimensions * sizeof(float));
    std::vector<float> next(config_.dimensions);

    // Iterative attention updates: xi_new = sum_j softmax(beta * sim(xi,xj)) * xj
    // This IS the modern Hopfield update (equivalent to transformer self-attention)
    for (size_t iter = 0; iter < max_iterations; ++iter) {
        attention_sum(current.data(), next.data());

        // Check convergence
        float diff = 0.0f;
        for (size_t i = 0; i < config_.dimensions; ++i) {
            float d = next[i] - current[i];
            diff += d * d;
        }
        diff = std::sqrt(diff);

        current.swap(next);
        result.iterations = iter + 1;

        if (diff < convergence_eps) {
            result.converged = true;
            break;
        }
    }

    // Find closest stored pattern for confidence
    std::memcpy(result.pattern.data(), current.data(), config_.dimensions * sizeof(float));

    float best_sim = -1.0f;
    size_t best_id = 0;
    for (size_t i = 0; i < patterns_.size(); ++i) {
        float sim = simd::cosine_similarity(current.data(), patterns_[i].data.data(), config_.dimensions);
        if (sim > best_sim) {
            best_sim = sim;
            best_id = patterns_[i].id;
        }
    }
    result.confidence = best_sim;
    result.pattern_id = best_id;

    // Compute entropy of final attention distribution
    auto final_weights = attention_weights(current.data());
    float entropy = 0.0f;
    for (float w : final_weights) {
        if (w > 1e-10f) {
            entropy -= w * std::log(w);
        }
    }
    result.entropy = entropy;

    return result;
}

// ---- Batch retrieval ----
BatchRetrievalResult HopfieldLayer::retrieve_batch(const std::vector<std::vector<float>>& cues,
                                                     size_t max_iterations) const {
    BatchRetrievalResult batch_result;
    batch_result.results.resize(cues.size());

    if (cues.empty()) return batch_result;

    // Use parallel threads for batch
    size_t n_threads = std::min(cues.size(), static_cast<size_t>(
        std::max(1u, std::thread::hardware_concurrency())));
    std::vector<std::future<void>> futures;

    auto worker = [this, &cues, &batch_result, max_iterations](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            batch_result.results[i] = retrieve(cues[i], max_iterations);
        }
    };

    size_t chunk = (cues.size() + n_threads - 1) / n_threads;
    for (size_t t = 0; t < n_threads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, cues.size());
        if (start < end) {
            futures.push_back(std::async(std::launch::async, worker, start, end));
        }
    }

    for (auto& f : futures) {
        f.get();
    }

    // Compute mean confidence
    float sum = 0.0f;
    for (const auto& r : batch_result.results) {
        sum += r.confidence;
    }
    batch_result.mean_confidence = sum / batch_result.results.size();

    return batch_result;
}

// ---- Direct lookup (no iteration) ----
RetrievalResult HopfieldLayer::lookup(const std::vector<float>& query) const {
    return retrieve(query, 1);
}

// ---- Capacity management ----
void HopfieldLayer::evict_internal() {
    if (patterns_.empty()) return;

    // Find pattern with lowest salience
    auto it = std::min_element(patterns_.begin(), patterns_.end(),
        [](const Pattern& a, const Pattern& b) { return a.salience < b.salience; });

    patterns_.erase(it);
}

void HopfieldLayer::evict() {
    std::lock_guard<std::mutex> lock(mutex_);
    evict_internal();
}

void HopfieldLayer::update_salience() {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    for (auto& p : patterns_) {
        // Decay existing salience
        p.salience *= config_.decay_rate;

        // Boost from recency (exponential decay from last access)
        double age_seconds = static_cast<double>(now - p.timestamp) / 1e6;
        float recency_boost = std::exp(-age_seconds / 3600.0f);  // 1-hour half-life

        // Boost from access frequency
        float freq_boost = std::log1p(static_cast<float>(p.access_count)) * 0.1f;

        p.salience = std::max(p.salience, recency_boost + freq_boost);
    }
}

const Pattern* HopfieldLayer::get_pattern(uint64_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& p : patterns_) {
        if (p.id == id) return &p;
    }
    return nullptr;
}

bool HopfieldLayer::remove(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(patterns_.begin(), patterns_.end(),
        [id](const Pattern& p) { return p.id == id; });
    if (it != patterns_.end()) {
        patterns_.erase(it);
        return true;
    }
    return false;
}

std::vector<uint64_t> HopfieldLayer::pattern_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint64_t> ids;
    ids.reserve(patterns_.size());
    for (const auto& p : patterns_) {
        ids.push_back(p.id);
    }
    return ids;
}

std::vector<float> HopfieldLayer::attention_matrix() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t n = patterns_.size();
    std::vector<float> matrix(n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            matrix[i * n + j] = simd::cosine_similarity(
                patterns_[i].data.data(), patterns_[j].data.data(), config_.dimensions);
        }
    }

    return matrix;
}

std::vector<std::pair<uint64_t, float>> HopfieldLayer::top_k(const std::vector<float>& query, size_t k) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::pair<uint64_t, float>> scored;
    scored.reserve(patterns_.size());

    for (const auto& p : patterns_) {
        float sim = simd::cosine_similarity(query.data(), p.data.data(), config_.dimensions);
        scored.emplace_back(p.id, sim);
    }

    std::partial_sort(scored.begin(), scored.begin() + std::min(k, scored.size()),
                      scored.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    if (scored.size() > k) scored.resize(k);
    return scored;
}

} // namespace neural
