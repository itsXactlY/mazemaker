#pragma once
// neural/graph.h - Knowledge Graph with Spreading Activation
// Part of Neural Memory Adapter

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <functional>
#include <optional>
#include <cmath>
#include <shared_mutex>
#include <mutex>
#include <numeric>

namespace neural {
namespace graph {

// ============================================================================
// Types
// ============================================================================

enum class NodeType : uint8_t {
    Entity    = 0,
    Event     = 1,
    Concept   = 2,
    Memory    = 3,
    Procedure = 4
};

enum class EdgeType : uint8_t {
    Similar    = 0,  // Content similarity
    Causal     = 1,  // A caused B
    Temporal   = 2,  // A happened before B
    Associative= 3,  // Co-activated
    Semantic   = 4,  // Meaning relationship
    Inferred   = 5   // Discovered by link prediction
};

struct Node {
    uint64_t id;
    NodeType type;
    uint64_t memory_id;         // Reference to vector store
    std::vector<float> embedding; // Cached embedding (optional)
    float centrality = 0.0f;    // Degree/PageRank centrality
    float activation = 0.0f;    // Current spreading activation
    uint64_t created_at = 0;
    uint64_t last_activated = 0;
    std::string label;
    
    bool has_embedding() const { return !embedding.empty(); }
};

struct Edge {
    uint64_t source_id;
    uint64_t target_id;
    EdgeType type;
    float weight;               // Connection strength [0, 1]
    uint64_t created_at;
    uint64_t last_activated;
    uint32_t activation_count;  // How many times this edge was traversed
    
    // For priority queue in spreading activation
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

struct TraversalResult {
    uint64_t node_id;
    float activation;
    int depth;
    std::vector<uint64_t> path;
};

struct ConnectionPrediction {
    uint64_t source_id;
    uint64_t target_id;
    float confidence;
    std::string method;  // "common_neighbors", "adamic_adar", "embedding"
};

// ============================================================================
// Knowledge Graph
// ============================================================================

class KnowledgeGraph {
public:
    KnowledgeGraph() = default;
    ~KnowledgeGraph() = default;
    
    // --- Node Operations ---
    
    uint64_t add_node(NodeType type, const std::string& label = "",
                      uint64_t memory_id = 0);
    
    bool remove_node(uint64_t node_id);
    
    Node* get_node(uint64_t node_id);
    const Node* get_node(uint64_t node_id) const;
    
    bool set_embedding(uint64_t node_id, const std::vector<float>& embedding);
    
    void update_centrality();  // Recompute all centralities
    
    // --- Edge Operations ---
    
    bool add_edge(uint64_t source, uint64_t target, EdgeType type,
                  float weight = 0.5f);
    
    bool remove_edge(uint64_t source, uint64_t target, EdgeType type);
    
    bool update_edge_weight(uint64_t source, uint64_t target,
                            EdgeType type, float new_weight);
    
    // Hebbian: strengthen edge when nodes are co-activated
    void hebbian_strengthen(uint64_t a, uint64_t b, float delta = 0.01f);
    
    std::vector<Edge> get_edges(uint64_t node_id) const;
    std::vector<std::pair<uint64_t, float>> neighbors(uint64_t node_id) const;
    
    // --- Traversal ---
    
    // Spreading activation: start from seed, propagate with decay
    std::vector<TraversalResult> spread_activation(
        uint64_t seed, float decay = 0.85f,
        float threshold = 0.01f, int max_depth = 5) const;
    
    // Multi-seed spreading
    std::vector<TraversalResult> spread_activation_multi(
        const std::vector<uint64_t>& seeds,
        const std::vector<float>& seed_weights,
        float decay = 0.85f, float threshold = 0.01f,
        int max_depth = 5) const;
    
    // BFS shortest path
    std::optional<std::vector<uint64_t>> shortest_path(
        uint64_t source, uint64_t target) const;
    
    // Subgraph extraction
    std::vector<uint64_t> subgraph(uint64_t center, int radius) const;
    
    // --- Link Prediction ---
    
    // Find likely new connections
    std::vector<ConnectionPrediction> predict_links(size_t max_results = 20) const;
    
    // Predict links for a specific node
    std::vector<ConnectionPrediction> predict_links_for(
        uint64_t node_id, size_t max_results = 10) const;
    
    // --- Metrics ---
    
    size_t node_count() const { return nodes_.size(); }
    size_t edge_count() const;
    
    float density() const;  // edges / possible_edges
    
    struct Stats {
        size_t nodes;
        size_t edges;
        float density;
        float avg_degree;
        float max_centrality;
        uint64_t most_central_node;
    };
    Stats get_stats() const;
    
    // --- Decay ---
    
    // Weaken old/unused connections
    void decay_edges(float decay_factor = 0.999f);
    
    // Remove very weak edges
    void prune_weak_edges(float threshold = 0.01f);
    
    // --- Bulk Operations ---
    
    size_t node_count_estimate() const { return nodes_.size(); }
    
    // Iterate all nodes
    void for_each_node(std::function<void(const Node&)> fn) const;
    
    // Iterate all edges
    void for_each_edge(std::function<void(const Edge&)> fn) const;

private:
    // Adjacency: node_id -> list of (neighbor_id, edge)
    std::unordered_map<uint64_t, std::vector<Edge>> adjacency_;
    std::unordered_map<uint64_t, Node> nodes_;
    
    mutable std::shared_mutex mutex_;
    uint64_t next_node_id_ = 1;
    
    // Internal helpers
    float common_neighbors_score(uint64_t a, uint64_t b) const;
    float adamic_adar_score(uint64_t a, uint64_t b) const;
    float jaccard_score(uint64_t a, uint64_t b) const;
    float embedding_similarity(uint64_t a, uint64_t b) const;
    
    std::unordered_set<uint64_t> get_neighbor_set(uint64_t node_id) const;
};

} // namespace graph
} // namespace neural
