// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>

#include "../../kernel_tags.hpp"
#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_template {
namespace batch_single_kernels {


template <typename ValueType, typename IndexType>
__device__ __forceinline__ void simple_apply_impl(
    const batch::matrix::csr::batch_item<const ValueType, IndexType>& mat,
    const batch::multi_vector::batch_item<const ValueType> b,
    batch::multi_vector::batch_item<ValueType> x)
{
    const auto num_rows = mat.num_rows;
    const auto val = mat.values;
    const auto col = mat.col_idxs;
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        auto temp = zero<ValueType>();
        for (auto nnz = mat.row_ptrs[row]; nnz < mat.row_ptrs[row + 1]; nnz++) {
            const auto col_idx = col[nnz];
            temp += val[nnz] * b.values[col_idx];
        }
        x.values[row] = temp;
    }
}


template <typename ValueType, typename IndexType>
__device__ __forceinline__ void advanced_apply_impl(
    const ValueType alpha,
    const batch::matrix::csr::batch_item<const ValueType, IndexType> mat,
    const batch::multi_vector::batch_item<const ValueType> b,
    const ValueType beta, const batch::multi_vector::batch_item<ValueType> x)
{
    const auto num_rows = mat.num_rows;
    const auto val = mat.values;
    const auto col = mat.col_idxs;
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        auto temp = zero<ValueType>();
        for (auto nnz = mat.row_ptrs[row]; nnz < mat.row_ptrs[row + 1]; nnz++) {
            const auto col_idx = col[nnz];
            temp += alpha * val[nnz] * b.values[col_idx];
        }
        x.values[row] = temp + beta * x.values[row];
    }
}

struct simple_apply_fn {
    template <typename ValueType, typename IndexType>
    __device__ __forceinline__ void operator()(
        const batch::matrix::csr::batch_item<const ValueType, IndexType> a,
        const batch::multi_vector::batch_item<const ValueType> b,
        batch::multi_vector::batch_item<ValueType> c) const
    {
        simple_apply_impl(a, b, c);
    }

    template <typename T, typename ValueType>
    __device__ __forceinline__ void operator()(
        const T a, const batch::multi_vector::batch_item<const ValueType> b,
        batch::multi_vector::batch_item<ValueType> c) const
    {
        simple_apply(a, b, c, device_kernel{});
    }
};

GKO_CPO_STORAGE constexpr simple_apply_fn simple_apply{};


struct advanced_apply_fn {
    template <typename ValueType, typename IndexType>
    __device__ __forceinline__ void operator()(
        const ValueType alpha,
        const batch::matrix::csr::batch_item<const ValueType, IndexType> a,
        const batch::multi_vector::batch_item<const ValueType> b,
        const ValueType beta,
        batch::multi_vector::batch_item<ValueType> c) const
    {
        advanced_apply_impl(alpha, a, b, beta, c);
    }

    template <typename T, typename ValueType>
    __device__ __forceinline__ void operator()(
        const ValueType alpha, const T& a,
        const batch::multi_vector::batch_item<const ValueType> b,
        const ValueType beta,
        batch::multi_vector::batch_item<ValueType> c) const
    {
        advanced_apply(alpha, a, b, beta, c, device_kernel{});
    }
};

GKO_CPO_STORAGE constexpr advanced_apply_fn advanced_apply{};


}  // namespace batch_single_kernels
}  // namespace batch_template
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
