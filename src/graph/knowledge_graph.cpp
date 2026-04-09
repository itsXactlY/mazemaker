// neural/graph/knowledge_graph.cpp - Knowledge Graph Implementation
#include "neural/graph.h"
#include "neural/simd.h"
#include <cassert>

namespace neural {
namespace graph {

// ============================================================================
// Node Operations
// ============================================================================

uint64_t KnowledgeGraph::add_node(NodeType type, const std::string& label,
                                   uint64_t memory_id) {
    std::unique_lock lock(mutex_);
    uint64_t id = next_node_id_++;
    Node node;
    node.id = id;
    node.type = type;
    node.memory_id = memory_id;
    node.label = label;
    node.created_at = 0;  // Will be set by caller if needed
    nodes_[id] = std::move(node);
    adjacency_[id] = {};  // Initialize adjacency list
    return id;
}

bool KnowledgeGraph::remove_node(uint64_t node_id) {
    std::unique_lock lock(mutex_);
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) return false;
    
    // Remove all edges involving this node
    adjacency_.erase(node_id);
    for (auto& [id, edges] : adjacency_) {
        edges.erase(
            std::remove_if(edges.begin(), edges.end(),
                [node_id](const Edge& e) { return e.target_id == node_id; }),
            edges.end()
        );
    }
    
    nodes_.erase(it);
    return true;
}

Node* KnowledgeGraph::get_node(uint64_t node_id) {
    std::shared_lock lock(mutex_);
    auto it = nodes_.find(node_id);
    return it != nodes_.end() ? &it->second : nullptr;
}

const Node* KnowledgeGraph::get_node(uint64_t node_id) const {
    std::shared_lock lock(mutex_);
    auto it = nodes_.find(node_id);
    return it != nodes_.end() ? &it->second : nullptr;
}

bool KnowledgeGraph::set_embedding(uint64_t node_id, const std::vector<float>& embedding) {
    std::unique_lock lock(mutex_);
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) return false;
    it->second.embedding = embedding;
    return true;
}

// ============================================================================
// Edge Operations
// ============================================================================

bool KnowledgeGraph::add_edge(uint64_t source, uint64_t target,
                               EdgeType type, float weight) {
    std::unique_lock lock(mutex_);
    if (nodes_.find(source) == nodes_.end() ||
        nodes_.find(target) == nodes_.end()) return false;
    
    // Check for duplicate
    for (const auto& edge : adjacency_[source]) {
        if (edge.target_id == target && edge.type == type) {
            return false;  // Already exists
        }
    }
    
    Edge edge;
    edge.source_id = source;
    edge.target_id = target;
    edge.type = type;
    edge.weight = weight;
    edge.created_at = 0;
    edge.last_activated = 0;
    edge.activation_count = 0;
    
    adjacency_[source].push_back(edge);
    
    // Add reverse edge for undirected traversal
    Edge reverse = edge;
    reverse.source_id = target;
    reverse.target_id = source;
    adjacency_[target].push_back(reverse);
    
    return true;
}

bool KnowledgeGraph::remove_edge(uint64_t source, uint64_t target, EdgeType type) {
    std::unique_lock lock(mutex_);
    
    auto remove_from = [&](uint64_t from, uint64_t to) {
        auto& edges = adjacency_[from];
        edges.erase(
            std::remove_if(edges.begin(), edges.end(),
                [to, type](const Edge& e) {
                    return e.target_id == to && e.type == type;
                }),
            edges.end()
        );
    };
    
    remove_from(source, target);
    remove_from(target, source);
    return true;
}

bool KnowledgeGraph::update_edge_weight(uint64_t source, uint64_t target,
                                         EdgeType type, float new_weight) {
    std::unique_lock lock(mutex_);
    bool found = false;
    
    auto update = [&](uint64_t from, uint64_t to) {
        for (auto& edge : adjacency_[from]) {
            if (edge.target_id == to && edge.type == type) {
                edge.weight = std::clamp(new_weight, 0.0f, 1.0f);
                found = true;
            }
        }
    };
    
    update(source, target);
    update(target, source);
    return found;
}

void KnowledgeGraph::hebbian_strengthen(uint64_t a, uint64_t b, float delta) {
    std::unique_lock lock(mutex_);
    
    auto strengthen = [&](uint64_t from, uint64_t to) {
        for (auto& edge : adjacency_[from]) {
            if (edge.target_id == to) {
                edge.weight = std::min(1.0f, edge.weight + delta);
                edge.activation_count++;
                edge.last_activated = 0;
            }
        }
    };
    
    strengthen(a, b);
    strengthen(b, a);
}

std::vector<Edge> KnowledgeGraph::get_edges(uint64_t node_id) const {
    std::shared_lock lock(mutex_);
    auto it = adjacency_.find(node_id);
    if (it == adjacency_.end()) return {};
    return it->second;
}

std::vector<std::pair<uint64_t, float>> KnowledgeGraph::neighbors(uint64_t node_id) const {
    std::shared_lock lock(mutex_);
    std::vector<std::pair<uint64_t, float>> result;
    auto it = adjacency_.find(node_id);
    if (it == adjacency_.end()) return result;
    
    for (const auto& edge : it->second) {
        result.emplace_back(edge.target_id, edge.weight);
    }
    
    // Sort by weight descending
    std::sort(result.begin(), result.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return result;
}

// ============================================================================
// Spreading Activation
// ============================================================================

std::vector<TraversalResult> KnowledgeGraph::spread_activation(
    uint64_t seed, float decay, float threshold, int max_depth) const
{
    std::shared_lock lock(mutex_);
    
    std::vector<TraversalResult> results;
    std::unordered_map<uint64_t, float> activation;
    std::unordered_map<uint64_t, int> depth;
    std::unordered_map<uint64_t, std::vector<uint64_t>> paths;
    
    // Priority queue: (activation, node_id)
    using PQEntry = std::pair<float, uint64_t>;
    std::priority_queue<PQEntry> pq;
    
    activation[seed] = 1.0f;
    depth[seed] = 0;
    paths[seed] = {seed};
    pq.push({1.0f, seed});
    
    while (!pq.empty()) {
        auto [act, current] = pq.top();
        pq.pop();
        
        if (act < threshold) continue;
        if (depth[current] >= max_depth) continue;
        
        auto it = adjacency_.find(current);
        if (it == adjacency_.end()) continue;
        
        for (const auto& edge : it->second) {
            float propagated = act * edge.weight * decay;
            
            if (propagated < threshold) continue;
            
            auto existing = activation.find(edge.target_id);
            if (existing == activation.end() || propagated > existing->second) {
                activation[edge.target_id] = propagated;
                depth[edge.target_id] = depth[current] + 1;
                
                paths[edge.target_id] = paths[current];
                paths[edge.target_id].push_back(edge.target_id);
                
                pq.push({propagated, edge.target_id});
            }
        }
    }
    
    // Convert to results (skip seed)
    for (const auto& [node_id, act] : activation) {
        if (node_id == seed) continue;
        results.push_back({
            node_id, act, depth[node_id], paths[node_id]
        });
    }
    
    // Sort by activation descending
    std::sort(results.begin(), results.end(),
        [](const TraversalResult& a, const TraversalResult& b) {
            return a.activation > b.activation;
        });
    
    return results;
}

std::vector<TraversalResult> KnowledgeGraph::spread_activation_multi(
    const std::vector<uint64_t>& seeds,
    const std::vector<float>& seed_weights,
    float decay, float threshold, int max_depth) const
{
    std::shared_lock lock(mutex_);
    
    std::unordered_map<uint64_t, float> activation;
    std::unordered_map<uint64_t, int> depths;
    
    // Initialize seeds
    for (size_t i = 0; i < seeds.size(); ++i) {
        float w = (i < seed_weights.size()) ? seed_weights[i] : 1.0f;
        activation[seeds[i]] = w;
        depths[seeds[i]] = 0;
    }
    
    // BFS spreading
    std::queue<uint64_t> queue;
    for (uint64_t seed : seeds) queue.push(seed);
    
    while (!queue.empty()) {
        uint64_t current = queue.front();
        queue.pop();
        
        float current_act = activation[current];
        int current_depth = depths[current];
        
        if (current_depth >= max_depth || current_act < threshold) continue;
        
        auto it = adjacency_.find(current);
        if (it == adjacency_.end()) continue;
        
        for (const auto& edge : it->second) {
            float propagated = current_act * edge.weight * decay;
            if (propagated < threshold) continue;
            
            auto existing = activation.find(edge.target_id);
            if (existing == activation.end()) {
                activation[edge.target_id] = propagated;
                depths[edge.target_id] = current_depth + 1;
                queue.push(edge.target_id);
            } else if (propagated > existing->second) {
                existing->second = propagated;
                depths[edge.target_id] = current_depth + 1;
                queue.push(edge.target_id);
            }
        }
    }
    
    // Build results
    std::vector<TraversalResult> results;
    std::unordered_set<uint64_t> seed_set(seeds.begin(), seeds.end());
    
    for (const auto& [node_id, act] : activation) {
        if (seed_set.count(node_id)) continue;
        results.push_back({node_id, act, depths[node_id], {}});
    }
    
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.activation > b.activation; });
    
    return results;
}

// ============================================================================
// Shortest Path
// ============================================================================

std::optional<std::vector<uint64_t>> KnowledgeGraph::shortest_path(
    uint64_t source, uint64_t target) const
{
    std::shared_lock lock(mutex_);
    
    if (nodes_.find(source) == nodes_.end() ||
        nodes_.find(target) == nodes_.end()) return std::nullopt;
    
    std::queue<uint64_t> queue;
    std::unordered_map<uint64_t, uint64_t> parent;
    
    queue.push(source);
    parent[source] = source;
    
    while (!queue.empty()) {
        uint64_t current = queue.front();
        queue.pop();
        
        if (current == target) {
            // Reconstruct path
            std::vector<uint64_t> path;
            uint64_t node = target;
            while (node != source) {
                path.push_back(node);
                node = parent[node];
            }
            path.push_back(source);
            std::reverse(path.begin(), path.end());
            return path;
        }
        
        auto it = adjacency_.find(current);
        if (it == adjacency_.end()) continue;
        
        for (const auto& edge : it->second) {
            if (parent.find(edge.target_id) == parent.end()) {
                parent[edge.target_id] = current;
                queue.push(edge.target_id);
            }
        }
    }
    
    return std::nullopt;  // No path found
}

std::vector<uint64_t> KnowledgeGraph::subgraph(uint64_t center, int radius) const {
    std::shared_lock lock(mutex_);
    
    std::vector<uint64_t> result;
    std::unordered_set<uint64_t> visited;
    std::queue<std::pair<uint64_t, int>> queue;
    
    queue.push({center, 0});
    visited.insert(center);
    
    while (!queue.empty()) {
        auto [current, depth] = queue.front();
        queue.pop();
        
        result.push_back(current);
        
        if (depth >= radius) continue;
        
        auto it = adjacency_.find(current);
        if (it == adjacency_.end()) continue;
        
        for (const auto& edge : it->second) {
            if (visited.insert(edge.target_id).second) {
                queue.push({edge.target_id, depth + 1});
            }
        }
    }
    
    return result;
}

// ============================================================================
// Link Prediction
// ============================================================================

std::vector<ConnectionPrediction> KnowledgeGraph::predict_links(
    size_t max_results) const
{
    std::shared_lock lock(mutex_);
    std::vector<ConnectionPrediction> predictions;
    
    // For each pair of non-connected nodes, compute scores
    std::vector<uint64_t> node_ids;
    for (const auto& [id, _] : nodes_) node_ids.push_back(id);
    
    for (size_t i = 0; i < node_ids.size() && predictions.size() < max_results * 2; ++i) {
        for (size_t j = i + 1; j < node_ids.size() && predictions.size() < max_results * 2; ++j) {
            uint64_t a = node_ids[i], b = node_ids[j];
            
            // Skip if already connected
            bool connected = false;
            auto it = adjacency_.find(a);
            if (it != adjacency_.end()) {
                for (const auto& edge : it->second) {
                    if (edge.target_id == b) { connected = true; break; }
                }
            }
            if (connected) continue;
            
            float cn = common_neighbors_score(a, b);
            float aa = adamic_adar_score(a, b);
            float emb = embedding_similarity(a, b);
            
            // Combined score
            float score = 0.3f * cn + 0.4f * aa + 0.3f * emb;
            
            if (score > 0.1f) {
                std::string method;
                if (aa >= cn && aa >= emb) method = "adamic_adar";
                else if (emb >= cn) method = "embedding";
                else method = "common_neighbors";
                
                predictions.push_back({a, b, score, method});
            }
        }
    }
    
    // Sort by confidence descending, take top K
    std::sort(predictions.begin(), predictions.end(),
        [](const auto& a, const auto& b) { return a.confidence > b.confidence; });
    
    if (predictions.size() > max_results)
        predictions.resize(max_results);
    
    return predictions;
}

std::vector<ConnectionPrediction> KnowledgeGraph::predict_links_for(
    uint64_t node_id, size_t max_results) const
{
    std::shared_lock lock(mutex_);
    std::vector<ConnectionPrediction> predictions;
    
    if (nodes_.find(node_id) == nodes_.end()) return predictions;
    
    // Get existing neighbors
    auto existing_neighbors = get_neighbor_set(node_id);
    existing_neighbors.insert(node_id);
    
    for (const auto& [other_id, _] : nodes_) {
        if (existing_neighbors.count(other_id)) continue;
        
        float cn = common_neighbors_score(node_id, other_id);
        float aa = adamic_adar_score(node_id, other_id);
        float emb = embedding_similarity(node_id, other_id);
        
        float score = 0.3f * cn + 0.4f * aa + 0.3f * emb;
        
        if (score > 0.1f) {
            std::string method;
            if (aa >= cn && aa >= emb) method = "adamic_adar";
            else if (emb >= cn) method = "embedding";
            else method = "common_neighbors";
            
            predictions.push_back({node_id, other_id, score, method});
        }
    }
    
    std::sort(predictions.begin(), predictions.end(),
        [](const auto& a, const auto& b) { return a.confidence > b.confidence; });
    
    if (predictions.size() > max_results)
        predictions.resize(max_results);
    
    return predictions;
}

// ============================================================================
// Metrics
// ============================================================================

size_t KnowledgeGraph::edge_count() const {
    std::shared_lock lock(mutex_);
    size_t count = 0;
    for (const auto& [_, edges] : adjacency_) count += edges.size();
    return count / 2;  // Each edge counted twice (undirected)
}

float KnowledgeGraph::density() const {
    std::shared_lock lock(mutex_);
    size_t n = nodes_.size();
    if (n < 2) return 0.0f;
    size_t e = 0;
    for (const auto& [_, edges] : adjacency_) e += edges.size();
    e /= 2;
    return static_cast<float>(e) / (n * (n - 1) / 2.0f);
}

KnowledgeGraph::Stats KnowledgeGraph::get_stats() const {
    std::shared_lock lock(mutex_);
    Stats stats;
    stats.nodes = nodes_.size();
    stats.edges = 0;
    stats.max_centrality = 0.0f;
    stats.most_central_node = 0;
    
    float total_degree = 0;
    for (const auto& [id, node] : nodes_) {
        float deg = 0;
        auto it = adjacency_.find(id);
        if (it != adjacency_.end()) deg = static_cast<float>(it->second.size());
        total_degree += deg;
        stats.edges += it != adjacency_.end() ? it->second.size() : 0;
        
        if (node.centrality > stats.max_centrality) {
            stats.max_centrality = node.centrality;
            stats.most_central_node = id;
        }
    }
    stats.edges /= 2;
    stats.avg_degree = nodes_.empty() ? 0.0f : total_degree / nodes_.size();
    stats.density = density();
    
    return stats;
}

// ============================================================================
// Decay & Pruning
// ============================================================================

void KnowledgeGraph::decay_edges(float decay_factor) {
    std::unique_lock lock(mutex_);
    for (auto& [_, edges] : adjacency_) {
        for (auto& edge : edges) {
            edge.weight *= decay_factor;
        }
    }
}

void KnowledgeGraph::prune_weak_edges(float threshold) {
    std::unique_lock lock(mutex_);
    for (auto& [_, edges] : adjacency_) {
        edges.erase(
            std::remove_if(edges.begin(), edges.end(),
                [threshold](const Edge& e) { return e.weight < threshold; }),
            edges.end()
        );
    }
}

// ============================================================================
// Iteration
// ============================================================================

void KnowledgeGraph::for_each_node(std::function<void(const Node&)> fn) const {
    std::shared_lock lock(mutex_);
    for (const auto& [_, node] : nodes_) fn(node);
}

void KnowledgeGraph::for_each_edge(std::function<void(const Edge&)> fn) const {
    std::shared_lock lock(mutex_);
    std::unordered_set<std::string> seen;
    for (const auto& [src, edges] : adjacency_) {
        for (const auto& edge : edges) {
            std::string key = std::to_string(std::min(src, edge.target_id)) + ":" +
                              std::to_string(std::max(src, edge.target_id));
            if (seen.insert(key).second) fn(edge);
        }
    }
}

// ============================================================================
// Internal Helpers
// ============================================================================

std::unordered_set<uint64_t> KnowledgeGraph::get_neighbor_set(uint64_t node_id) const {
    std::unordered_set<uint64_t> result;
    auto it = adjacency_.find(node_id);
    if (it != adjacency_.end()) {
        for (const auto& edge : it->second) result.insert(edge.target_id);
    }
    return result;
}

float KnowledgeGraph::common_neighbors_score(uint64_t a, uint64_t b) const {
    auto na = get_neighbor_set(a);
    auto nb = get_neighbor_set(b);
    size_t common = 0;
    for (uint64_t id : na) {
        if (nb.count(id)) common++;
    }
    // Normalize by max possible
    size_t max_possible = std::min(na.size(), nb.size());
    return max_possible > 0 ? static_cast<float>(common) / max_possible : 0.0f;
}

float KnowledgeGraph::adamic_adar_score(uint64_t a, uint64_t b) const {
    auto na = get_neighbor_set(a);
    auto nb = get_neighbor_set(b);
    float score = 0.0f;
    
    for (uint64_t common : na) {
        if (!nb.count(common)) continue;
        // Weight = 1/log(degree(common))
        auto it = adjacency_.find(common);
        if (it != adjacency_.end() && it->second.size() > 1) {
            score += 1.0f / std::log(static_cast<float>(it->second.size()));
        }
    }
    return score;
}

float KnowledgeGraph::jaccard_score(uint64_t a, uint64_t b) const {
    auto na = get_neighbor_set(a);
    auto nb = get_neighbor_set(b);
    size_t intersection = 0;
    for (uint64_t id : na) {
        if (nb.count(id)) intersection++;
    }
    size_t union_size = na.size() + nb.size() - intersection;
    return union_size > 0 ? static_cast<float>(intersection) / union_size : 0.0f;
}

float KnowledgeGraph::embedding_similarity(uint64_t a, uint64_t b) const {
    auto it_a = nodes_.find(a);
    auto it_b = nodes_.find(b);
    if (it_a == nodes_.end() || it_b == nodes_.end()) return 0.0f;
    if (!it_a->second.has_embedding() || !it_b->second.has_embedding()) return 0.0f;
    
    return simd::cosine_similarity(
        it_a->second.embedding.data(),
        it_b->second.embedding.data(),
        it_a->second.embedding.size()
    );
}

void KnowledgeGraph::update_centrality() {
    std::unique_lock lock(mutex_);
    
    // Simple degree centrality (normalized)
    float max_deg = 0;
    std::unordered_map<uint64_t, float> degrees;
    
    for (const auto& [id, edges] : adjacency_) {
        degrees[id] = static_cast<float>(edges.size());
        max_deg = std::max(max_deg, degrees[id]);
    }
    
    for (auto& [id, node] : nodes_) {
        node.centrality = max_deg > 0 ? degrees[id] / max_deg : 0.0f;
    }
}

} // namespace graph
} // namespace neural
