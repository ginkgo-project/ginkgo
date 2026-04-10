// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/sketch/count_sketch_kernels.hpp"

#include <random>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
namespace count_sketch {


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const DefaultExecutor> exec,
              size_type sketch_size, array<IndexType>& hash_map,
              array<ValueType>& signs, uint64 seed)
{
    auto input_size = hash_map.get_size();
    auto hash_data = hash_map.get_data();
    auto sign_data = signs.get_data();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<IndexType> hash_dist(
        0, static_cast<IndexType>(sketch_size - 1));
    std::bernoulli_distribution sign_dist(0.5);
    for (size_type i = 0; i < input_size; ++i) {
        hash_data[i] = hash_dist(rng);
        sign_data[i] = sign_dist(rng) ? one<ValueType>() : -one<ValueType>();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COUNT_SKETCH_GENERATE);


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const array<IndexType>& hash_map, const array<ValueType>& signs,
           matrix::view::dense<const ValueType> b,
           matrix::view::dense<ValueType> x)
{
    auto input_size = hash_map.get_size();
    auto num_cols = b.size[1];
    auto hash_data = hash_map.get_const_data();
    auto sign_data = signs.get_const_data();
    // Zero output
    for (size_type row = 0; row < x.size[0]; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            x(row, col) = zero<ValueType>();
        }
    }
    // Scatter-add
    for (size_type i = 0; i < input_size; ++i) {
        auto target_row = hash_data[i];
        auto sign = sign_data[i];
        for (size_type col = 0; col < num_cols; ++col) {
            x(target_row, col) += sign * b(i, col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COUNT_SKETCH_APPLY);


template <typename ValueType, typename IndexType>
void rapply(std::shared_ptr<const DefaultExecutor> exec,
            const array<IndexType>& hash_map, const array<ValueType>& signs,
            matrix::view::dense<const ValueType> b,
            matrix::view::dense<ValueType> x)
{
    auto input_size = hash_map.get_size();
    auto num_rows = b.size[0];
    auto hash_data = hash_map.get_const_data();
    auto sign_data = signs.get_const_data();
    // Zero output
    for (size_type row = 0; row < x.size[0]; ++row) {
        for (size_type col = 0; col < x.size[1]; ++col) {
            x(row, col) = zero<ValueType>();
        }
    }
    // Gather-accumulate: x[:, j] = sum_{i: hash[i]==j} sign[i] * b[:, i]
    // input_size = m (columns of b = rows of S), b is (n x m), x is (n x k)
    for (size_type i = 0; i < input_size; ++i) {
        auto target_col = hash_data[i];
        auto sign = sign_data[i];
        for (size_type row = 0; row < num_rows; ++row) {
            x(row, target_col) += sign * b(row, i);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COUNT_SKETCH_RAPPLY);


}  // namespace count_sketch
}  // namespace reference
}  // namespace kernels
}  // namespace gko
