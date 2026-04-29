// tests/test_graph.cpp - Knowledge graph tests
#include "mazemaker/graph.h"
#include <cassert>
#include <iostream>

using namespace neural::graph;

void test_add_remove() {
    KnowledgeGraph g;
    auto n1 = g.add_node(NodeType::Memory, "node1");
    auto n2 = g.add_node(NodeType::Memory, "node2");
    auto n3 = g.add_node(NodeType::Concept, "node3");

    assert(g.node_count() == 3);

    g.add_edge(n1, n2, EdgeType::Similar, 0.8f);
    g.add_edge(n2, n3, EdgeType::Causal, 0.5f);

    assert(g.edge_count() == 2);

    g.remove_node(n2);
    assert(g.node_count() == 2);
    assert(g.edge_count() == 0);

    std::cout << "  add_remove: OK\n";
}

void test_spreading_activation() {
    KnowledgeGraph g;
    auto n1 = g.add_node(NodeType::Memory, "A");
    auto n2 = g.add_node(NodeType::Memory, "B");
    auto n3 = g.add_node(NodeType::Memory, "C");
    auto n4 = g.add_node(NodeType::Memory, "D");

    g.add_edge(n1, n2, EdgeType::Similar, 0.9f);
    g.add_edge(n2, n3, EdgeType::Similar, 0.8f);
    g.add_edge(n3, n4, EdgeType::Similar, 0.7f);

    auto activated = g.spread_activation(n1, 0.85f, 0.01f, 5);
    assert(!activated.empty());

    // n2 should be most activated (direct neighbor)
    assert(activated[0].node_id == n2);
    assert(activated[0].activation > activated[1].activation);

    std::cout << "  spreading_activation: OK (" << activated.size() << " nodes activated)\n";
}

void test_shortest_path() {
    KnowledgeGraph g;
    auto n1 = g.add_node(NodeType::Memory, "A");
    auto n2 = g.add_node(NodeType::Memory, "B");
    auto n3 = g.add_node(NodeType::Memory, "C");

    g.add_edge(n1, n2, EdgeType::Similar, 1.0f);
    g.add_edge(n2, n3, EdgeType::Similar, 1.0f);

    auto path = g.shortest_path(n1, n3);
    assert(path.has_value());
    assert(path->size() == 3);
    assert((*path)[0] == n1);
    assert((*path)[2] == n3);

    std::cout << "  shortest_path: OK\n";
}

void test_hebbian() {
    KnowledgeGraph g;
    auto n1 = g.add_node(NodeType::Memory, "A");
    auto n2 = g.add_node(NodeType::Memory, "B");

    g.add_edge(n1, n2, EdgeType::Associative, 0.5f);

    // Strengthen
    g.hebbian_strengthen(n1, n2, 0.1f);
    g.hebbian_strengthen(n1, n2, 0.1f);

    auto edges = g.get_edges(n1);
    assert(!edges.empty());
    assert(edges[0].weight > 0.5f);

    std::cout << "  hebbian: OK (weight=" << edges[0].weight << ")\n";
}

void test_link_prediction() {
    KnowledgeGraph g;
    auto n1 = g.add_node(NodeType::Memory, "A");
    auto n2 = g.add_node(NodeType::Memory, "B");
    auto n3 = g.add_node(NodeType::Memory, "C");
    auto n4 = g.add_node(NodeType::Memory, "D");

    // A-B, B-C connected, predict A-C
    g.add_edge(n1, n2, EdgeType::Similar, 0.9f);
    g.add_edge(n2, n3, EdgeType::Similar, 0.9f);

    auto predictions = g.predict_links_for(n1, 5);
    // n3 should be a predicted link (common neighbor: n2)
    bool found = false;
    for (const auto& p : predictions) {
        if (p.target_id == n3) { found = true; break; }
    }
    assert(found);

    std::cout << "  link_prediction: OK\n";
}

int main() {
    std::cout << "=== Graph Tests ===\n";
    test_add_remove();
    test_spreading_activation();
    test_shortest_path();
    test_hebbian();
    test_link_prediction();
    std::cout << "\nAll tests passed.\n";
    return 0;
}
