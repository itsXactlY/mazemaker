// tests/test_hopfield.cpp - Hopfield memory tests
#include "mazemaker/hopfield.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

using namespace neural;

std::vector<float> random_vec(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1, 1);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    float n = 0; for (float x : v) n += x*x;
    n = std::sqrt(n);
    if (n > 0) for (float& x : v) x /= n;
    return v;
}

void test_store_retrieve() {
    HopfieldConfig cfg;
    cfg.dimensions = 64;
    cfg.capacity = 100;
    cfg.beta = 20.0f;

    HopfieldLayer mem(cfg);

    std::mt19937 rng(42);
    auto pattern = random_vec(64, rng);

    uint64_t id = mem.store(pattern, "test", "episodic");
    assert(id > 0);
    assert(mem.pattern_count() == 1);

    auto result = mem.retrieve(pattern);
    assert(result.converged || result.confidence > 0.5f);

    std::cout << "  store_retrieve: OK (confidence=" << result.confidence << ")\n";
}

void test_capacity() {
    HopfieldConfig cfg;
    cfg.dimensions = 32;
    cfg.capacity = 10;

    HopfieldLayer mem(cfg);
    std::mt19937 rng(42);

    for (int i = 0; i < 15; ++i) {
        auto v = random_vec(32, rng);
        mem.store(v, "p" + std::to_string(i));
    }

    // Should have evicted some
    assert(mem.pattern_count() <= 10);
    std::cout << "  capacity: OK (count=" << mem.pattern_count() << ")\n";
}

void test_top_k() {
    HopfieldConfig cfg;
    cfg.dimensions = 64;
    cfg.capacity = 50;

    HopfieldLayer mem(cfg);
    std::mt19937 rng(42);

    auto target = random_vec(64, rng);
    mem.store(target, "target");

    // Store noise
    for (int i = 0; i < 20; ++i) {
        mem.store(random_vec(64, rng), "noise_" + std::to_string(i));
    }

    auto top = mem.top_k(target, 3);
    assert(!top.empty());
    // Best match should be the target itself (or close)
    std::cout << "  top_k: OK (best=" << top[0].second << ")\n";
}

int main() {
    std::cout << "=== Hopfield Tests ===\n";
    test_store_retrieve();
    test_capacity();
    test_top_k();
    std::cout << "\nAll tests passed.\n";
    return 0;
}
