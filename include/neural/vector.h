// include/neural/vector.h - Vector types using actual SIMD API
#pragma once

#include "neural/simd.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <memory>

namespace neural {

// ============================================================================
// Aligned allocation helpers
// ============================================================================

template<typename T>
T* aligned_alloc_t(size_t count) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 32, count * sizeof(T)) != 0) {
        return nullptr;
    }
    return static_cast<T*>(ptr);
}

template<typename T>
void aligned_free_t(T* ptr) {
    free(ptr);
}

// ============================================================================
// Vector32f - Float32 vector with SIMD operations
// ============================================================================

class Vector32f {
public:
    Vector32f() : data_(nullptr), dim_(0), owns_(false) {}

    explicit Vector32f(size_t dim) : dim_(dim), owns_(true) {
        data_ = aligned_alloc_t<float>(dim);
        std::memset(data_, 0, dim * sizeof(float));
    }

    Vector32f(size_t dim, float fill) : dim_(dim), owns_(true) {
        data_ = aligned_alloc_t<float>(dim);
        std::fill_n(data_, dim, fill);
    }

    Vector32f(std::initializer_list<float> init) : dim_(init.size()), owns_(true) {
        data_ = aligned_alloc_t<float>(dim_);
        std::copy(init.begin(), init.end(), data_);
    }

    Vector32f(const float* src, size_t dim) : dim_(dim), owns_(true) {
        data_ = aligned_alloc_t<float>(dim);
        std::memcpy(data_, src, dim * sizeof(float));
    }

    ~Vector32f() { if (owns_ && data_) aligned_free_t(data_); }

    Vector32f(const Vector32f& o) : dim_(o.dim_), owns_(true) {
        data_ = aligned_alloc_t<float>(dim_);
        std::memcpy(data_, o.data_, dim_ * sizeof(float));
    }

    Vector32f& operator=(const Vector32f& o) {
        if (this != &o) {
            if (owns_ && data_) aligned_free_t(data_);
            dim_ = o.dim_;
            owns_ = true;
            data_ = aligned_alloc_t<float>(dim_);
            std::memcpy(data_, o.data_, dim_ * sizeof(float));
        }
        return *this;
    }

    // Access
    size_t dim() const { return dim_; }
    float* data() { return data_; }
    const float* data() const { return data_; }

    float operator[](size_t i) const { return data_[i]; }
    float& operator[](size_t i) { return data_[i]; }

    // SIMD operations
    float dot(const Vector32f& o) const {
        return simd::dot_product(data_, o.data_, dim_);
    }

    float cosine_similarity(const Vector32f& o) const {
        return simd::cosine_similarity(data_, o.data_, dim_);
    }

    float norm() const {
        return simd::l2_norm(data_, dim_);
    }

    void normalize() {
        simd::normalize(data_, dim_);
    }

    // Arithmetic
    Vector32f operator+(const Vector32f& o) const {
        Vector32f r(dim_);
        simd::add(data_, o.data_, r.data_, dim_);
        return r;
    }

    Vector32f operator*(const Vector32f& o) const {  // Hadamard
        Vector32f r(dim_);
        simd::hadamard(data_, o.data_, r.data_, dim_);
        return r;
    }

    Vector32f operator*(float s) const {
        Vector32f r(dim_);
        simd::scale(data_, s, r.data_, dim_);
        return r;
    }

    void zero() { simd::zero(data_, dim_); }

    bool empty() const { return dim_ == 0; }

    friend std::ostream& operator<<(std::ostream& os, const Vector32f& v) {
        os << "[";
        size_t show = std::min(v.dim_, (size_t)8);
        for (size_t i = 0; i < show; ++i) {
            if (i > 0) os << ", ";
            os << v.data_[i];
        }
        if (v.dim_ > show) os << ", ...";
        os << "] (" << v.dim_ << "d)";
        return os;
    }

private:
    float* data_;
    size_t dim_;
    bool owns_;
};

// ============================================================================
// Batch operations
// ============================================================================

inline std::vector<float> batch_cosine_similarity(
    const Vector32f& query,
    const std::vector<Vector32f>& vectors)
{
    size_t n = vectors.size();
    std::vector<float> results(n);
    for (size_t i = 0; i < n; ++i) {
        results[i] = query.cosine_similarity(vectors[i]);
    }
    return results;
}

inline std::vector<float> batch_cosine_similarity_contiguous(
    const float* query,
    const float* vectors,
    size_t count,
    size_t dim)
{
    std::vector<float> results(count);
    simd::batch_cosine_similarity(query, vectors, count, dim, results.data());
    return results;
}

} // namespace neural
