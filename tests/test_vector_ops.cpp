// tests/test_vector_ops.cpp - Basic vector ops tests
#include "mazemaker/simd.h"
#include "mazemaker/vector.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

using namespace neural;

bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_dot_product() {
    float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[] = {8, 7, 6, 5, 4, 3, 2, 1};
    float expected = 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1; // 120
    float result = simd::dot_product(a, b, 8);
    assert(approx_eq(result, expected));
    std::cout << "  dot_product: OK\n";
}

void test_cosine_similarity() {
    float a[] = {1, 0, 0, 0, 0, 0, 0, 0};
    float b[] = {1, 0, 0, 0, 0, 0, 0, 0};
    assert(approx_eq(simd::cosine_similarity(a, b, 8), 1.0f));

    float c[] = {0, 1, 0, 0, 0, 0, 0, 0};
    assert(approx_eq(simd::cosine_similarity(a, c, 8), 0.0f));

    float d[] = {-1, 0, 0, 0, 0, 0, 0, 0};
    assert(approx_eq(simd::cosine_similarity(a, d, 8), -1.0f));
    std::cout << "  cosine_similarity: OK\n";
}

void test_l2_norm() {
    float a[] = {3, 4, 0, 0, 0, 0, 0, 0};
    assert(approx_eq(simd::l2_norm(a, 8), 5.0f));
    std::cout << "  l2_norm: OK\n";
}

void test_normalize() {
    float a[] = {3, 4, 0, 0, 0, 0, 0, 0};
    simd::normalize(a, 8);
    assert(approx_eq(simd::l2_norm(a, 8), 1.0f));
    assert(approx_eq(a[0], 0.6f));
    assert(approx_eq(a[1], 0.8f));
    std::cout << "  normalize: OK\n";
}

void test_hadamard() {
    float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[] = {2, 3, 4, 5, 6, 7, 8, 9};
    float c[8];
    simd::hadamard(a, b, c, 8);
    for (int i = 0; i < 8; ++i) {
        assert(approx_eq(c[i], a[i] * b[i]));
    }
    std::cout << "  hadamard: OK\n";
}

void test_batch_cosine() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1, 1);
    const size_t dim = 768, count = 100;

    std::vector<float> query(dim), vectors(count * dim), results(count);
    for (auto& x : query) x = dist(rng);
    for (auto& x : vectors) x = dist(rng);

    simd::batch_cosine_similarity(query.data(), vectors.data(), count, dim, results.data());

    // Verify first result matches single computation
    float single = simd::cosine_similarity(query.data(), vectors.data(), dim);
    assert(approx_eq(results[0], single, 1e-4f));
    std::cout << "  batch_cosine_similarity: OK\n";
}

void test_vector32f() {
    Vector32f v(8, 0.0f);
    assert(v.dim() == 8);
    assert(v.norm() == 0.0f);

    Vector32f a({1, 2, 3, 4, 5, 6, 7, 8});
    Vector32f b({8, 7, 6, 5, 4, 3, 2, 1});
    assert(approx_eq(a.dot(b), 120.0f));
    std::cout << "  Vector32f: OK\n";
}

int main() {
    std::cout << "=== Vector Ops Tests ===\n";
    test_dot_product();
    test_cosine_similarity();
    test_l2_norm();
    test_normalize();
    test_hadamard();
    test_batch_cosine();
    test_vector32f();
    std::cout << "\nAll tests passed.\n";
    return 0;
}
