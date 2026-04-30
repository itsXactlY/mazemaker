// src/memory/consolidation.cpp - Memory Consolidation Engine
#include "mazemaker/memory.h"
#include "mazemaker/hopfield.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <thread>

namespace neural::memory {

// Utility: current time in microseconds
uint64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// MemoryEntry age calculation
double MemoryEntry::age_seconds(uint64_t now_us) const {
    return static_cast<double>(now_us - timestamp) / 1e6;
}

double MemoryEntry::recency_seconds(uint64_t now_us) const {
    return static_cast<double>(now_us - last_accessed) / 1e6;
}

// ============================================================
// Consolidation: Episodic -> Semantic Transfer
// ============================================================

// Exponential decay function
// R(t) = R0 * exp(-t / tau)
// where tau is the time constant (decay_rate controls this)
float exponential_decay(float initial_value, float time_seconds, float decay_rate = 0.001f) {
    return initial_value * std::exp(-decay_rate * time_seconds);
}

// Salience update: combines recency and frequency
// salience = alpha * recency + (1-alpha) * frequency
float compute_salience(const MemoryEntry& entry, uint64_t now) {
    double age = entry.age_seconds(now);
    double recency = entry.recency_seconds(now);

    // Recency score: exponential decay from last access
    float recency_score = static_cast<float>(std::exp(-recency / 3600.0));  // 1-hour half-life

    // Frequency score: log-normalized access count
    float freq_score = std::log1p(static_cast<float>(entry.access_count)) / 10.0f;
    freq_score = std::min(freq_score, 1.0f);

    // Combine with weights
    return 0.6f * recency_score + 0.4f * freq_score;
}

// Connection strength between two memory entries
// Based on embedding similarity and co-activation history
float connection_strength_internal(const MemoryEntry& a, const MemoryEntry& b,
                                    size_t dimensions) {
    if (a.embedding.empty() || b.embedding.empty()) return 0.0f;

    float sim = simd::cosine_similarity(a.embedding.data(), b.embedding.data(), dimensions);

    // Boost from temporal proximity
    double time_diff = std::abs(static_cast<double>(a.timestamp) - static_cast<double>(b.timestamp)) / 1e6;
    float temporal_boost = static_cast<float>(std::exp(-time_diff / 60.0));  // 1-minute decay

    // Boost from shared links
    float link_boost = 0.0f;
    std::unordered_set<uint64_t> a_links(a.linked.begin(), a.linked.end());
    for (uint64_t lid : b.linked) {
        if (a_links.count(lid)) link_boost += 0.1f;
    }
    link_boost = std::min(link_boost, 0.3f);

    return std::max(0.0f, sim * 0.5f + temporal_boost * 0.3f + link_boost);
}

// Self-attention sweep: find related memories via the Hopfield layer
// Returns (memory_a, memory_b, strength) tuples
std::vector<std::tuple<uint64_t, uint64_t, float>>
self_attention_sweep(const std::vector<const MemoryEntry*>& memories,
                     size_t dimensions, float strength_threshold = 0.5f) {
    std::vector<std::tuple<uint64_t, uint64_t, float>> connections;

    if (memories.size() < 2) return connections;

    // Build attention matrix via pairwise similarities
    size_t n = memories.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float strength = connection_strength_internal(*memories[i], *memories[j], dimensions);
            if (strength >= strength_threshold) {
                connections.emplace_back(memories[i]->id, memories[j]->id, strength);
            }
        }
    }

    // Sort by strength descending
    std::sort(connections.begin(), connections.end(),
        [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });

    return connections;
}

// ============================================================
// MemoryManager consolidation methods
// ============================================================

size_t MemoryManager::consolidate(size_t batch_size) {
    uint64_t now = now_us();

    // 1. Get consolidation candidates from episodic memory
    auto candidates = episodic_.candidates_for_consolidation(batch_size);

    if (candidates.empty()) return 0;

    // 2. Run self-attention sweep to find related memories
    auto connections = self_attention_sweep(candidates, dimensions_);

    // 3. Transfer high-salience episodic memories to semantic memory
    size_t transferred = 0;
    std::vector<uint64_t> episodic_ids;

    for (const auto* entry : candidates) {
        if (entry->salience > 0.3f || entry->access_count >= 3) {
            // Transfer to semantic memory
            MemoryEntry semantic_entry = *entry;
            semantic_entry.source = "consolidated";
            semantic_entry.last_accessed = now;
            semantic_entry.decay_factor = 1.0f;

            uint64_t semantic_id = semantic_.store(semantic_entry);
            episodic_ids.push_back(entry->id);

            // Also store in Hopfield layer
            hopfield_.store(entry->embedding, entry->label, "consolidated");

            // Record consolidation event
            ConsolidationEvent event;
            event.timestamp = now;
            event.episodic_ids.push_back(entry->id);
            event.semantic_id = semantic_id;
            event.operation = "transfer";
            event.strength = entry->salience;

            if (consolidation_cb_) {
                consolidation_cb_(event);
            }

            transferred++;
        }
    }

    // 4. Update connections from sweep results
    {
        std::unique_lock<std::shared_mutex> lock(conn_mutex_);
        for (const auto& [id_a, id_b, strength] : connections) {
            connections_[id_a][id_b] = strength;
            connections_[id_b][id_a] = strength;
        }
    }

    // 5. Remove transferred episodic memories
    for (uint64_t id : episodic_ids) {
        episodic_.remove(id);
    }

    // 6. Rebuild semantic clusters
    if (transferred > 0) {
        semantic_.rebuild_clusters();
    }

    last_consolidation_time_ = now;
    return transferred;
}

float MemoryManager::connection_strength(uint64_t id_a, uint64_t id_b) const {
    std::shared_lock<std::shared_mutex> lock(conn_mutex_);
    auto it_a = connections_.find(id_a);
    if (it_a == connections_.end()) return 0.0f;
    auto it_b = it_a->second.find(id_b);
    if (it_b == it_a->second.end()) return 0.0f;
    return it_b->second;
}

std::vector<std::pair<uint64_t, float>> MemoryManager::connections(uint64_t id) const {
    std::shared_lock<std::shared_mutex> lock(conn_mutex_);
    std::vector<std::pair<uint64_t, float>> result;
    auto it = connections_.find(id);
    if (it != connections_.end()) {
        for (const auto& [other_id, strength] : it->second) {
            result.emplace_back(other_id, strength);
        }
    }
    // Sort by strength
    std::sort(result.begin(), result.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    return result;
}

void MemoryManager::apply_decay(float decay_factor) {
    // Decay semantic memory salience
    semantic_.decay_all(decay_factor);

    // Decay Hopfield layer salience
    hopfield_.update_salience();
}

void MemoryManager::maybe_auto_consolidate() {
    if (episodic_.occupancy() > auto_consolidate_threshold_) {
        consolidate(64);
    }
}

} // namespace neural::memory
