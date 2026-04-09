// src/memory/memory_manager.cpp - EpisodicMemory, SemanticMemory, MemoryManager implementations
#include "neural/memory.h"
#include "neural/hopfield.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>

namespace neural::memory {

// ============================================================
// EpisodicMemory Implementation
// ============================================================

EpisodicMemory::EpisodicMemory(size_t capacity, size_t dimensions)
    : capacity_(capacity), dimensions_(dimensions) {
}

uint64_t EpisodicMemory::write(const std::vector<float>& embedding, const std::string& label,
                                const std::string& content) {
    std::unique_lock lock(mutex_);

    // Evict if full (FIFO)
    while (entries_.size() >= capacity_) {
        evict_oldest_internal();
    }

    MemoryEntry entry;
    entry.id = next_id_++;
    entry.embedding = embedding;
    entry.label = label;
    entry.content = content;
    entry.source = "perception";
    entry.timestamp = now_us();
    entry.last_accessed = entry.timestamp;
    entry.access_count = 0;
    entry.salience = 1.0f;
    entry.decay_factor = 1.0f;

    uint64_t id = entry.id;
    id_to_index_[id] = entries_.size();
    entries_.push_back(std::move(entry));

    return id;
}

const MemoryEntry* EpisodicMemory::read(uint64_t id) const {
    std::shared_lock lock(mutex_);
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return nullptr;
    return &entries_[it->second];
}

std::vector<std::pair<uint64_t, float>> EpisodicMemory::search(
    const std::vector<float>& query, size_t k) const {
    std::shared_lock lock(mutex_);

    std::vector<std::pair<uint64_t, float>> scored;
    scored.reserve(entries_.size());

    for (const auto& entry : entries_) {
        if (entry.embedding.empty()) continue;
        float sim = simd::cosine_similarity(query.data(), entry.embedding.data(), dimensions_);
        scored.emplace_back(entry.id, sim);
    }

    std::partial_sort(scored.begin(),
                      scored.begin() + std::min(k, scored.size()),
                      scored.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    if (scored.size() > k) scored.resize(k);
    return scored;
}

std::vector<const MemoryEntry*> EpisodicMemory::candidates_for_consolidation(
    size_t max_count) const {
    std::shared_lock lock(mutex_);

    std::vector<const MemoryEntry*> candidates;
    candidates.reserve(entries_.size());

    for (const auto& entry : entries_) {
        candidates.push_back(&entry);
    }

    // Sort by: access_count (desc), then age (oldest first)
    uint64_t now = now_us();
    std::sort(candidates.begin(), candidates.end(),
        [now](const MemoryEntry* a, const MemoryEntry* b) {
            if (a->access_count != b->access_count)
                return a->access_count > b->access_count;
            return a->timestamp < b->timestamp;
        });

    if (candidates.size() > max_count) candidates.resize(max_count);
    return candidates;
}

std::optional<MemoryEntry> EpisodicMemory::evict_oldest() {
    std::unique_lock lock(mutex_);
    return evict_oldest_internal();
}

std::optional<MemoryEntry> EpisodicMemory::evict_oldest_internal() {
    if (entries_.empty()) return std::nullopt;

    MemoryEntry oldest = std::move(entries_.front());
    id_to_index_.erase(oldest.id);

    // Shift remaining entries
    entries_.pop_front();

    // Rebuild index map
    id_to_index_.clear();
    for (size_t i = 0; i < entries_.size(); ++i) {
        id_to_index_[entries_[i].id] = i;
    }

    return oldest;
}

bool EpisodicMemory::remove(uint64_t id) {
    std::unique_lock lock(mutex_);
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return false;

    size_t idx = it->second;
    entries_.erase(entries_.begin() + idx);
    id_to_index_.erase(it);

    // Rebuild index for entries after removed one
    for (size_t i = idx; i < entries_.size(); ++i) {
        id_to_index_[entries_[i].id] = i;
    }

    return true;
}

void EpisodicMemory::touch(uint64_t id) {
    std::unique_lock lock(mutex_);
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return;

    auto& entry = entries_[it->second];
    entry.last_accessed = now_us();
    entry.access_count++;
}

// ============================================================
// SemanticMemory Implementation
// ============================================================

SemanticMemory::SemanticMemory(size_t dimensions, size_t max_clusters)
    : dimensions_(dimensions), max_clusters_(max_clusters) {
    clusters_.reserve(max_clusters);
}

uint64_t SemanticMemory::store(const MemoryEntry& entry) {
    std::unique_lock lock(mutex_);
    uint64_t id = next_id_++;
    MemoryEntry copy = entry;
    copy.id = id;
    entries_[id] = std::move(copy);

    // Assign to cluster
    assign_to_cluster(id, entries_[id].embedding);

    return id;
}

const MemoryEntry* SemanticMemory::read(uint64_t id) const {
    std::shared_lock lock(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

std::vector<std::pair<uint64_t, float>> SemanticMemory::search(
    const std::vector<float>& query, size_t k) const {
    std::shared_lock lock(mutex_);

    std::vector<std::pair<uint64_t, float>> scored;
    scored.reserve(entries_.size());

    for (const auto& [id, entry] : entries_) {
        if (entry.embedding.empty()) continue;
        float sim = simd::cosine_similarity(query.data(), entry.embedding.data(), dimensions_);
        scored.emplace_back(id, sim);
    }

    std::partial_sort(scored.begin(),
                      scored.begin() + std::min(k, scored.size()),
                      scored.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    if (scored.size() > k) scored.resize(k);
    return scored;
}

void SemanticMemory::assign_to_cluster(uint64_t entry_id, const std::vector<float>& embedding) {
    if (clusters_.empty() || clusters_.size() < max_clusters_) {
        // Create new cluster
        Cluster cluster;
        cluster.id = clusters_.size();
        cluster.centroid = embedding;
        cluster.member_ids.push_back(entry_id);
        cluster.coherence = 1.0f;
        cluster.created = now_us();
        cluster.last_updated = cluster.created;
        clusters_.push_back(std::move(cluster));
        return;
    }

    // Find nearest cluster
    size_t best_cluster = 0;
    float best_sim = -1.0f;
    for (size_t i = 0; i < clusters_.size(); ++i) {
        if (clusters_[i].centroid.empty()) continue;
        float sim = simd::cosine_similarity(embedding.data(), clusters_[i].centroid.data(), dimensions_);
        if (sim > best_sim) {
            best_sim = sim;
            best_cluster = i;
        }
    }

    clusters_[best_cluster].member_ids.push_back(entry_id);
    update_cluster_centroid(clusters_[best_cluster]);
}

void SemanticMemory::update_cluster_centroid(Cluster& cluster) {
    if (cluster.member_ids.empty()) return;

    std::vector<float> centroid(dimensions_, 0.0f);
    size_t count = 0;

    for (uint64_t id : cluster.member_ids) {
        auto it = entries_.find(id);
        if (it != entries_.end() && !it->second.embedding.empty()) {
            simd::add(centroid.data(), it->second.embedding.data(), centroid.data(), dimensions_);
            count++;
        }
    }

    if (count > 0) {
        float norm = simd::l2_norm(centroid.data(), dimensions_);
        if (norm > 1e-10f) {
            simd::scale(centroid.data(), 1.0f / norm, centroid.data(), dimensions_);
        }
        cluster.centroid = std::move(centroid);
        cluster.last_updated = now_us();

        // Compute coherence (avg pairwise similarity)
        if (cluster.member_ids.size() > 1) {
            float total_sim = 0.0f;
            size_t pairs = 0;
            for (size_t i = 0; i < std::min(cluster.member_ids.size(), size_t(10)); ++i) {
                auto it_a = entries_.find(cluster.member_ids[i]);
                if (it_a == entries_.end()) continue;
                for (size_t j = i + 1; j < std::min(cluster.member_ids.size(), size_t(10)); ++j) {
                    auto it_b = entries_.find(cluster.member_ids[j]);
                    if (it_b == entries_.end()) continue;
                    total_sim += simd::cosine_similarity(
                        it_a->second.embedding.data(),
                        it_b->second.embedding.data(), dimensions_);
                    pairs++;
                }
            }
            cluster.coherence = pairs > 0 ? total_sim / pairs : 0.0f;
        }
    }
}

void SemanticMemory::rebuild_clusters() {
    clusters_.clear();

    // Reassign all entries
    for (const auto& [id, entry] : entries_) {
        assign_to_cluster(id, entry.embedding);
    }
}

uint64_t SemanticMemory::merge(uint64_t id_a, uint64_t id_b) {
    std::unique_lock lock(mutex_);

    auto it_a = entries_.find(id_a);
    auto it_b = entries_.find(id_b);
    if (it_a == entries_.end() || it_b == entries_.end()) return 0;

    // Create merged entry
    MemoryEntry merged;
    merged.id = next_id_++;
    merged.timestamp = std::min(it_a->second.timestamp, it_b->second.timestamp);
    merged.last_accessed = now_us();
    merged.access_count = it_a->second.access_count + it_b->second.access_count;
    merged.salience = std::max(it_a->second.salience, it_b->second.salience);
    merged.source = "merged";
    merged.label = it_a->second.label + "+" + it_b->second.label;

    // Average embeddings
    merged.embedding.resize(dimensions_);
    for (size_t i = 0; i < dimensions_; ++i) {
        merged.embedding[i] = (it_a->second.embedding[i] + it_b->second.embedding[i]) * 0.5f;
    }

    // Combine links
    merged.linked = it_a->second.linked;
    merged.linked.insert(merged.linked.end(), it_b->second.linked.begin(), it_b->second.linked.end());
    merged.linked.push_back(id_a);
    merged.linked.push_back(id_b);

    uint64_t new_id = merged.id;
    entries_[new_id] = std::move(merged);

    // Remove originals
    entries_.erase(it_a);
    entries_.erase(it_b);

    // Rebuild clusters
    rebuild_clusters();

    return new_id;
}

void SemanticMemory::decay_all(float decay_factor) {
    std::unique_lock lock(mutex_);
    for (auto& [id, entry] : entries_) {
        entry.salience *= decay_factor;
        entry.decay_factor *= decay_factor;
    }
}

// ============================================================
// MemoryManager Implementation
// ============================================================

MemoryManager::MemoryManager(size_t dimensions, size_t episodic_capacity, size_t semantic_max_clusters)
    : episodic_(episodic_capacity, dimensions)
    , semantic_(dimensions, semantic_max_clusters)
    , hopfield_(HopfieldConfig{.dimensions = dimensions, .capacity = 1024})
    , dimensions_(dimensions) {
}

uint64_t MemoryManager::remember(const std::vector<float>& embedding, const std::string& label,
                                   const std::string& content) {
    uint64_t id = episodic_.write(embedding, label, content);

    // Also store in Hopfield for fast associative retrieval
    hopfield_.store(embedding, label, "episodic");

    // Auto-consolidate if episodic memory is filling up
    maybe_auto_consolidate();

    return id;
}

std::vector<std::pair<uint64_t, float>> MemoryManager::recall(
    const std::vector<float>& query, size_t k) const {
    // Search both tiers and merge results
    auto episodic_results = episodic_.search(query, k);
    auto semantic_results = semantic_.search(query, k);

    // Also check Hopfield layer
    auto hopfield_top = hopfield_.top_k(query, k);

    // Merge and deduplicate, keeping highest similarity
    std::unordered_map<uint64_t, float> merged;
    for (const auto& [id, sim] : episodic_results) {
        merged[id] = std::max(merged[id], sim);
    }
    for (const auto& [id, sim] : semantic_results) {
        merged[id] = std::max(merged[id], sim);
    }

    // Convert to vector and sort
    std::vector<std::pair<uint64_t, float>> results(merged.begin(), merged.end());
    std::partial_sort(results.begin(),
                      results.begin() + std::min(k, results.size()),
                      results.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    if (results.size() > k) results.resize(k);
    return results;
}

const MemoryEntry* MemoryManager::read(uint64_t id) const {
    // Try episodic first
    if (const auto* entry = episodic_.read(id)) {
        return entry;
    }
    // Then semantic
    return semantic_.read(id);
}

} // namespace neural::memory
