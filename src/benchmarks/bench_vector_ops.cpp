// benchmarks/bench_vector_ops.cpp - Vector Operations Benchmark
#include "neural/simd.h"
#include "neural/vector.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstring>

using namespace neural;
using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    std::string name;
    double ops_per_sec;
    double ns_per_op;
    double speedup_vs_scalar;
};

// ============================================================================
// Benchmark Utilities
// ============================================================================

std::vector<float> random_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<std::vector<float>> random_vectors(size_t count, size_t dim, std::mt19937& rng) {
    std::vector<std::vector<float>> result(count);
    for (auto& v : result) v = random_vector(dim, rng);
    return result;
}

template<typename Func>
double bench_op(Func&& op, size_t iterations, const std::string& warmup_name = "") {
    // Warmup
    for (size_t i = 0; i < std::min(iterations / 10, (size_t)1000); ++i) op();
    
    auto start = Clock::now();
    for (size_t i = 0; i < iterations; ++i) op();
    auto end = Clock::now();
    
    double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return us;
}

// ============================================================================
// Scalar Baselines
// ============================================================================

float scalar_dot(const float* a, const float* b, size_t n) {
    float sum = 0;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

float scalar_cosine(const float* a, const float* b, size_t n) {
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0 ? dot / denom : 0;
}

float scalar_l2_dist(const float* a, const float* b, size_t n) {
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

// ============================================================================
// Benchmarks
// ============================================================================

void bench_dot_product(const std::vector<size_t>& dims) {
    std::cout << "\n=== DOT PRODUCT ===\n";
    std::cout << std::setw(8) << "Dim"
              << std::setw(15) << "Scalar (ns)"
              << std::setw(15) << "SIMD (ns)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    std::mt19937 rng(42);
    
    for (size_t dim : dims) {
        auto a = random_vector(dim, rng);
        auto b = random_vector(dim, rng);
        size_t iters = std::max((size_t)100000, 10000000 / dim);
        
        // Scalar
        double scalar_ns = bench_op([&]() {
            volatile float r = scalar_dot(a.data(), b.data(), dim);
            (void)r;
        }, iters);
        
        // SIMD
        double simd_ns = bench_op([&]() {
            volatile float r = simd::dot_product(a.data(), b.data(), dim);
            (void)r;
        }, iters);
        
        double scalar_per_op = scalar_ns / iters;
        double simd_per_op = simd_ns / iters;
        double speedup = scalar_per_op / simd_per_op;
        
        std::cout << std::setw(8) << dim
                  << std::setw(15) << std::fixed << std::setprecision(2) << scalar_per_op
                  << std::setw(15) << simd_per_op
                  << std::setw(11) << std::setprecision(1) << speedup << "x\n";
    }
}

void bench_cosine_similarity(const std::vector<size_t>& dims) {
    std::cout << "\n=== COSINE SIMILARITY ===\n";
    std::cout << std::setw(8) << "Dim"
              << std::setw(15) << "Scalar (ns)"
              << std::setw(15) << "SIMD (ns)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    std::mt19937 rng(42);
    
    for (size_t dim : dims) {
        auto a = random_vector(dim, rng);
        auto b = random_vector(dim, rng);
        size_t iters = std::max((size_t)100000, 10000000 / dim);
        
        double scalar_ns = bench_op([&]() {
            volatile float r = scalar_cosine(a.data(), b.data(), dim);
            (void)r;
        }, iters);
        
        double simd_ns = bench_op([&]() {
            volatile float r = simd::cosine_similarity(a.data(), b.data(), dim);
            (void)r;
        }, iters);
        
        double scalar_per_op = scalar_ns / iters;
        double simd_per_op = simd_ns / iters;
        double speedup = scalar_per_op / simd_per_op;
        
        std::cout << std::setw(8) << dim
                  << std::setw(15) << std::fixed << std::setprecision(2) << scalar_per_op
                  << std::setw(15) << simd_per_op
                  << std::setw(11) << std::setprecision(1) << speedup << "x\n";
    }
}

void bench_batch_similarity(size_t dim, const std::vector<size_t>& batch_sizes) {
    std::cout << "\n=== BATCH COSINE SIMILARITY (dim=" << dim << ") ===\n";
    std::cout << std::setw(10) << "Batch"
              << std::setw(15) << "Scalar (us)"
              << std::setw(15) << "SIMD (us)"
              << std::setw(15) << "SIMD+OMP (us)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(67, '-') << "\n";
    
    std::mt19937 rng(42);
    auto query = random_vector(dim, rng);
    
    for (size_t batch : batch_sizes) {
        auto vectors = random_vectors(batch, dim, rng);
        
        // Flatten for SIMD batch
        std::vector<float> flat(batch * dim);
        for (size_t i = 0; i < batch; ++i) {
            std::memcpy(flat.data() + i * dim, vectors[i].data(), dim * sizeof(float));
        }
        
        size_t iters = std::max((size_t)100, 10000 / batch);
        
        // Scalar
        double scalar_us = bench_op([&]() {
            for (size_t i = 0; i < batch; ++i) {
                volatile float r = scalar_cosine(query.data(), vectors[i].data(), dim);
                (void)r;
            }
        }, iters) / 1000.0;
        
        // SIMD sequential
        std::vector<float> results(batch);
        double simd_us = bench_op([&]() {
            for (size_t i = 0; i < batch; ++i) {
                results[i] = simd::cosine_similarity(query.data(), flat.data() + i * dim, dim);
            }
        }, iters) / 1000.0;
        
        // SIMD + OpenMP
        double simd_omp_us = bench_op([&]() {
            simd::batch_cosine_similarity(query.data(), flat.data(), batch, dim, results.data());
        }, iters) / 1000.0;
        
        double speedup = scalar_us / simd_omp_us;
        
        std::cout << std::setw(10) << batch
                  << std::setw(15) << std::fixed << std::setprecision(2) << scalar_us
                  << std::setw(15) << simd_us
                  << std::setw(15) << simd_omp_us
                  << std::setw(11) << std::setprecision(1) << speedup << "x\n";
    }
}

void bench_normalize(const std::vector<size_t>& dims) {
    std::cout << "\n=== L2 NORMALIZE ===\n";
    std::cout << std::setw(8) << "Dim"
              << std::setw(15) << "Scalar (ns)"
              << std::setw(15) << "SIMD (ns)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    std::mt19937 rng(42);
    
    for (size_t dim : dims) {
        auto a = random_vector(dim, rng);
        size_t iters = std::max((size_t)100000, 10000000 / dim);
        
        // Scalar
        auto a_copy = a;
        double scalar_ns = bench_op([&]() {
            std::memcpy(a_copy.data(), a.data(), dim * sizeof(float));
            float norm = 0;
            for (size_t i = 0; i < dim; ++i) norm += a_copy[i] * a_copy[i];
            norm = std::sqrt(norm);
            if (norm > 0) for (size_t i = 0; i < dim; ++i) a_copy[i] /= norm;
        }, iters);
        
        // SIMD
        a_copy = a;
        double simd_ns = bench_op([&]() {
            std::memcpy(a_copy.data(), a.data(), dim * sizeof(float));
            simd::normalize(a_copy.data(), dim);
        }, iters);
        
        double scalar_per_op = scalar_ns / iters;
        double simd_per_op = simd_ns / iters;
        double speedup = scalar_per_op / simd_per_op;
        
        std::cout << std::setw(8) << dim
                  << std::setw(15) << std::fixed << std::setprecision(2) << scalar_per_op
                  << std::setw(15) << simd_per_op
                  << std::setw(11) << std::setprecision(1) << speedup << "x\n";
    }
}

void bench_throughput() {
    std::cout << "\n=== THROUGHPUT (768-dim vectors) ===\n";
    
    const size_t dim = 768;
    const size_t count = 1000000;
    std::mt19937 rng(42);
    
    auto query = random_vector(dim, rng);
    auto vectors = random_vectors(count, dim, rng);
    
    // Flatten
    std::vector<float> flat(count * dim);
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(flat.data() + i * dim, vectors[i].data(), dim * sizeof(float));
    }
    
    std::vector<float> results(count);
    
    // SIMD batch
    auto start = Clock::now();
    simd::batch_cosine_similarity(query.data(), flat.data(), count, dim, results.data());
    auto end = Clock::now();
    
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double ops_per_sec = count / (ms / 1000.0);
    
    std::cout << "  " << count << " cosine similarities (" << dim << "d)\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(1) << ms << " ms\n";
    std::cout << "  Throughput: " << std::setprecision(0) << ops_per_sec / 1e6 << "M ops/sec\n";
    std::cout << "  Per-op: " << std::setprecision(0) << (ms * 1e6) / count << " ns\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "==============================================\n";
    std::cout << "  Neural Memory Adapter - Vector Ops Benchmark\n";
    std::cout << "==============================================\n";
    
    // Check SIMD availability
    std::cout << "\nSIMD Support:\n";
    #ifdef HAS_AVX2
    std::cout << "  AVX2: YES\n";
    #else
    std::cout << "  AVX2: NO\n";
    #endif
    #ifdef HAS_AVX512
    std::cout << "  AVX-512: YES\n";
    #else
    std::cout << "  AVX-512: NO\n";
    #endif
    #ifdef HAS_OPENMP
    std::cout << "  OpenMP: YES\n";
    #else
    std::cout << "  OpenMP: NO\n";
    #endif
    
    std::vector<size_t> dims = {64, 128, 256, 384, 512, 768, 1024, 1536};
    
    bench_dot_product(dims);
    bench_cosine_similarity(dims);
    bench_normalize(dims);
    bench_batch_similarity(768, {10, 100, 1000, 10000, 100000});
    bench_throughput();
    
    std::cout << "\nDone.\n";
    return 0;
}
