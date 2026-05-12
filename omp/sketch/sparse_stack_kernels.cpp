// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/sketch/sparse_stack_kernels.hpp"

#include <random>

#include <omp.h>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

namespace gko {
namespace kernels {
namespace omp {
namespace sparse_stack {


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const DefaultExecutor> exec,
              size_type sketch_size, size_type input_size, size_type zeta,
              array<IndexType>& hash_map, array<ValueType>& signs, uint64 seed)
{
    auto hash_data = hash_map.get_data();
    auto sign_data = signs.get_data();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<IndexType> hash_dist(
        0, static_cast<IndexType>(std::max(static_cast<IndexType>(sketch_size - 1), static_cast<IndexType>(0))));
    std::bernoulli_distribution sign_dist(0.5);
    for (size_type i = 0; i < input_size * zeta; ++i) {
        hash_data[i] = hash_dist(rng);
        sign_data[i] = sign_dist(rng) ? one<ValueType>() : -one<ValueType>();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSE_STACK_GENERATE);

template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const DefaultExecutor> exec, size_type zeta,
           const array<IndexType>& hash_map, const array<ValueType>& signs,
           matrix::view::dense<const ValueType> b,
           matrix::view::dense<ValueType> x)
{
    auto input_size = hash_map.get_size() / zeta;
    auto num_cols = b.size[1];
    auto hash_data = hash_map.get_const_data();
    auto sign_data = signs.get_const_data();

#pragma omp parallel for
    for (size_type row = 0; row < x.size[0]; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            x(row, col) = zero<ValueType>();
        }
    }

    for (size_type i = 0; i < input_size; ++i) {
        for (size_type z = 0; z < zeta; ++z) {
            auto idx = i * zeta + z;
            auto target_row = hash_data[idx];
            auto sign = sign_data[idx];
            for (size_type col = 0; col < num_cols; ++col) {
                x(target_row, col) += sign * b(i, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSE_STACK_APPLY);

template <typename ValueType, typename IndexType>
void rapply(std::shared_ptr<const DefaultExecutor> exec, size_type zeta,
            const array<IndexType>& hash_map, const array<ValueType>& signs,
            matrix::view::dense<const ValueType> b,
            matrix::view::dense<ValueType> x)
{
    auto input_size = hash_map.get_size() / zeta;
    auto num_rows = b.size[0];
    auto hash_data = hash_map.get_const_data();
    auto sign_data = signs.get_const_data();

#pragma omp parallel for
    for (size_type row = 0; row < x.size[0]; ++row) {
        for (size_type col = 0; col < x.size[1]; ++col) {
            x(row, col) = zero<ValueType>();
        }
    }

    for (size_type i = 0; i < input_size; ++i) {
        for (size_type z = 0; z < zeta; ++z) {
            auto idx = i * zeta + z;
            auto target_col = hash_data[idx];
            auto sign = sign_data[idx];
#pragma omp parallel for
            for (size_type row = 0; row < num_rows; ++row) {
                x(row, target_col) += sign * b(row, i);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSE_STACK_RAPPLY);


}  // namespace sparse_stack
}  // namespace omp
}  // namespace kernels
}  // namespace gko
