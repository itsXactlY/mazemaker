#!/usr/bin/env python3
"""
demo.py - Neural Memory Adapter End-to-End Demo
Shows storing memories, retrieval, spreading activation, and knowledge graph.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from memory_client import NeuralMemory
import time

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_result(i, r):
    print(f"  {i}. [{r['id']}] {r['label']}")
    print(f"     Similarity: {r['similarity']}")
    if r.get('connections'):
        conns = ", ".join(f"{c['label'][:30]}({c['weight']:.2f})" for c in r['connections'][:3])
        print(f"     Connected to: {conns}")
    if r.get('content') and r['content'] != r['label']:
        print(f"     Content: {r['content'][:80]}")

def main():
    print_header("Neural Memory Adapter - Live Demo")
    
    # Use fresh DB for demo
    db_path = os.path.expanduser("~/.neural_memory/demo.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    mem = NeuralMemory(db_path=db_path)
    print(f"\nBackend: {mem.embedder.backend.__class__.__name__} ({mem.dim}d)")
    
    # ========================================================================
    # Phase 1: Store memories
    # ========================================================================
    print_header("Phase 1: Storing Memories")
    
    facts = [
        ("Person", "My name is Max, I'm 30 years old and live in Berlin"),
        ("Pet", "I have a dog named Lou, she's a Chihuahua mix"),
        ("Pet", "Lou loves to play with her squeaky toy elephant"),
        ("Pet", "I also have 5 other Chihuahuas - it's a big pack"),
        ("Work", "I'm working on BTQuant, a quantitative trading platform"),
        ("Work", "BTQuant uses Microsoft SQL Server for data storage"),
        ("Work", "The trading system connects to Interactive Brokers via JRR"),
        ("Tech", "I'm building a neural memory adapter in C++ with AVX2 SIMD"),
        ("Tech", "The memory system uses Modern Hopfield Networks"),
        ("Tech", "Hopfield Networks are mathematically equivalent to transformer attention"),
        ("Tech", "I use HotSpine for real-time market data streaming"),
        ("Home", "Looking for a freestanding house to rent in Germany"),
        ("Home", "Need a place where all my dogs are allowed"),
        ("Home", "Budget is up to 1300 euros warm rent"),
        ("Home", "Want a house in true isolation, no neighbors nearby"),
        ("Project", "Also working on a DayZ mod called VanillaPPMap"),
        ("Project", "DayZ modding uses Enforce Script, similar to C++"),
        ("Project", "LBMaster handles admin functionality for the server"),
        ("Music", "I play trumpet and have a trombone named Trompetenkopf"),
        ("Food", "My favorite food is Thai curry with extra chili"),
    ]
    
    # Pre-train embedder on all texts for consistent embeddings
    print("\n  Pre-training embedder on corpus...")
    all_texts = [text for _, text in facts]
    if hasattr(mem.embedder.backend, 'fit'):
        mem.embedder.backend.fit(all_texts)
        mem.embedder.backend._trained = True
    print(f"  Embedder trained: {mem.embedder.backend.__class__.__name__}")
    
    ids = []
    for label, text in facts:
        t0 = time.time()
        mid = mem.remember(text, label=label)
        dt = (time.time() - t0) * 1000
        ids.append(mid)
        print(f"  [{mid}] {label}: {text[:60]}... ({dt:.1f}ms)")
    
    print(f"\nStored {len(ids)} memories")
    
    # ========================================================================
    # Phase 2: Retrieval
    # ========================================================================
    print_header("Phase 2: Retrieval - Natural Language Queries")
    
    queries = [
        "What kind of pet does Max have?",
        "What is BTQuant?",
        "Where does Max want to live?",
        "Tell me about the neural memory project",
        "What musical instrument does Max play?",
    ]
    
    for query in queries:
        print(f"\n  Query: \"{query}\"")
        results = mem.recall(query, k=3)
        for i, r in enumerate(results, 1):
            print_result(i, r)
    
    # ========================================================================
    # Phase 3: Spreading Activation ("Thinking")
    # ========================================================================
    print_header("Phase 3: Spreading Activation - 'Thinking'")
    
    # Find the "Lou" memory
    lou_results = mem.recall("Lou dog", k=1)
    if lou_results:
        lou_id = lou_results[0]['id']
        print(f"\n  Starting from: [{lou_id}] {lou_results[0]['label']}")
        print(f"  Lou's content: {lou_results[0]['content']}")
        
        thoughts = mem.think(lou_id, depth=3, decay=0.85)
        print(f"\n  Spreading activation (depth=3, decay=0.85):")
        for i, t in enumerate(thoughts[:8], 1):
            print(f"    {i}. [{t['id']}] {t['label']} (activation: {t['activation']})")
    
    # Think from BTQuant
    btq_results = mem.recall("BTQuant trading", k=1)
    if btq_results:
        btq_id = btq_results[0]['id']
        print(f"\n  Starting from: [{btq_id}] {btq_results[0]['label']}")
        thoughts = mem.think(btq_id, depth=2, decay=0.9)
        print(f"\n  Connected thoughts:")
        for i, t in enumerate(thoughts[:5], 1):
            print(f"    {i}. [{t['id']}] {t['label']} (activation: {t['activation']})")
    
    # ========================================================================
    # Phase 4: Knowledge Graph
    # ========================================================================
    print_header("Phase 4: Knowledge Graph")
    
    graph = mem.graph()
    print(f"\n  Nodes: {graph['nodes']}")
    print(f"  Edges: {graph['edges']}")
    print(f"\n  Top connections (strongest):")
    for i, edge in enumerate(graph['top_edges'][:8], 1):
        from_mem = mem.store.get(edge['from'])
        to_mem = mem.store.get(edge['to'])
        if from_mem and to_mem:
            print(f"    {i}. [{edge['from']}] {from_mem['label'][:30]} <--{edge['weight']:.3f}--> [{edge['to']}] {to_mem['label'][:30]}")
    
    # ========================================================================
    # Phase 5: Cross-Domain Discovery
    # ========================================================================
    print_header("Phase 5: Cross-Domain Connection Discovery")
    
    print("\n  These memories were automatically connected by the system:")
    print("  (The system found relationships across different topics)\n")
    
    # Show connections for a few interesting nodes
    for query in ["neural memory", "Chihuahuas", "house rent"]:
        results = mem.recall(query, k=1)
        if results:
            mid = results[0]['id']
            conns = mem.connections(mid)
            if conns:
                print(f"  [{mid}] {results[0]['label']}:")
                for c in conns[:4]:
                    print(f"    -> [{c['id']}] {c['label'][:50]} ({c['weight']:.3f}, {c['type']})")
                print()
    
    # ========================================================================
    # Stats
    # ========================================================================
    print_header("Final Stats")
    
    stats = mem.stats()
    print(f"\n  Memories: {stats['memories']}")
    print(f"  Connections: {stats['connections']}")
    print(f"  Embedding: {stats['embedding_dim']}d ({stats['embedding_backend']})")
    print(f"  DB: {db_path}")
    
    mem.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
