// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_MATRIX_BATCH_CSR_KERNELS_HPP_
#define GKO_COMMON_CUDA_HIP_MATRIX_BATCH_CSR_KERNELS_HPP_


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>

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
namespace batch_single_kernels {


// The values of the batch items below are pointers or arithmetic() ranges, and
// the functions compute in the type the elements of the result read as.


template <typename ValueType, typename IndexType, typename MatValues,
          typename BValues, typename XValues>
__device__ __forceinline__ void simple_apply(
    const gko::batch::matrix::csr::batch_item<const ValueType, IndexType,
                                              MatValues>& mat,
    const BValues b, const XValues x)
{
    using arithmetic_type = arithmetic_type_t<decltype(x)>;
    const auto num_rows = mat.num_rows;
    const auto val = mat.values;
    const auto col = mat.col_idxs;
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        auto temp = zero<arithmetic_type>();
        for (auto nnz = mat.row_ptrs[row]; nnz < mat.row_ptrs[row + 1]; nnz++) {
            const auto col_idx = col[nnz];
            temp += val[nnz] * b[col_idx];
        }
        x[row] = temp;
    }
}

template <typename ValueType, typename IndexType, typename MatValues,
          typename BValues, typename XValues>
__global__ __launch_bounds__(default_block_size) void simple_apply_kernel(
    const gko::batch::matrix::csr::uniform_batch<const ValueType, IndexType,
                                                 MatValues>
        mat,
    const gko::batch::multi_vector::uniform_batch<const ValueType, BValues> b,
    const gko::batch::multi_vector::uniform_batch<ValueType, XValues> x)
{
    for (size_type batch_id = blockIdx.x; batch_id < mat.num_batch_items;
         batch_id += gridDim.x) {
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const auto b_b = gko::batch::extract_batch_item(b, batch_id);
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        simple_apply(mat_b, b_b.values, x_b.values);
    }
}


template <typename ArithmeticType, typename ValueType, typename IndexType,
          typename MatValues, typename BValues, typename XValues>
__device__ __forceinline__ void advanced_apply(
    const ArithmeticType alpha,
    const gko::batch::matrix::csr::batch_item<const ValueType, IndexType,
                                              MatValues>& mat,
    const BValues b, const ArithmeticType beta, const XValues x)
{
    const auto num_rows = mat.num_rows;
    const auto val = mat.values;
    const auto col = mat.col_idxs;
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        auto temp = zero<ArithmeticType>();
        for (auto nnz = mat.row_ptrs[row]; nnz < mat.row_ptrs[row + 1]; nnz++) {
            const auto col_idx = col[nnz];
            temp += alpha * val[nnz] * b[col_idx];
        }
        x[row] = temp + beta * x[row];
    }
}

template <typename ValueType, typename IndexType, typename AlphaValues,
          typename MatValues, typename BValues, typename BetaValues,
          typename XValues>
__global__ __launch_bounds__(default_block_size) void advanced_apply_kernel(
    const gko::batch::multi_vector::uniform_batch<const ValueType, AlphaValues>
        alpha,
    const gko::batch::matrix::csr::uniform_batch<const ValueType, IndexType,
                                                 MatValues>
        mat,
    const gko::batch::multi_vector::uniform_batch<const ValueType, BValues> b,
    const gko::batch::multi_vector::uniform_batch<const ValueType, BetaValues>
        beta,
    const gko::batch::multi_vector::uniform_batch<ValueType, XValues> x)
{
    for (size_type batch_id = blockIdx.x; batch_id < mat.num_batch_items;
         batch_id += gridDim.x) {
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const auto b_b = gko::batch::extract_batch_item(b, batch_id);
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        const auto alpha_b = gko::batch::extract_batch_item(alpha, batch_id);
        const auto beta_b = gko::batch::extract_batch_item(beta, batch_id);
        advanced_apply(alpha_b.values[0], mat_b, b_b.values, beta_b.values[0],
                       x_b.values);
    }
}


template <typename ScaleValues, typename IndexType, typename Values>
__device__ __forceinline__ void scale(
    const int num_rows, const ScaleValues col_scale,
    const ScaleValues row_scale, const IndexType* const __restrict__ col_idxs,
    const IndexType* const __restrict__ row_ptrs, const Values values)
{
    constexpr auto warp_size = config::warp_size;
    const auto tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const int tile_rank = threadIdx.x / warp_size;
    const int num_tiles = ceildiv(blockDim.x, warp_size);

    for (int row = tile_rank; row < num_rows; row += num_tiles) {
        const auto row_scalar = row_scale[row];
        for (int col = row_ptrs[row] + tile.thread_rank();
             col < row_ptrs[row + 1]; col += warp_size) {
            values[col] *= row_scalar * col_scale[col_idxs[col]];
        }
    }
}


template <typename ScaleValues, typename ValueType, typename IndexType,
          typename Values>
__global__ void scale_kernel(
    const ScaleValues col_scale_vals, const ScaleValues row_scale_vals,
    const gko::batch::matrix::csr::uniform_batch<ValueType, IndexType, Values>
        mat)
{
    auto num_rows = mat.num_rows;
    auto num_cols = mat.num_cols;
    for (size_type batch_id = blockIdx.x; batch_id < mat.num_batch_items;
         batch_id += gridDim.x) {
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const auto col_scale_b = gko::batch::multi_vector::batch_item_ptr(
            col_scale_vals, 1, num_cols, batch_id);
        const auto row_scale_b = gko::batch::multi_vector::batch_item_ptr(
            row_scale_vals, 1, num_rows, batch_id);
        scale(mat.num_rows, col_scale_b, row_scale_b, mat_b.col_idxs,
              mat_b.row_ptrs, mat_b.values);
    }
}


template <typename ArithmeticType, typename ValueType, typename IndexType,
          typename Values>
__device__ __forceinline__ void add_scaled_identity(
    const ArithmeticType alpha, const ArithmeticType beta,
    const gko::batch::matrix::csr::batch_item<ValueType, IndexType, Values>&
        mat)
{
    constexpr auto warp_size = config::warp_size;
    const auto tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const int tile_rank = threadIdx.x / warp_size;
    const int num_tiles = ceildiv(blockDim.x, warp_size);

    for (int row = tile_rank; row < mat.num_rows; row += num_tiles) {
        for (int nnz = mat.row_ptrs[row] + tile.thread_rank();
             nnz < mat.row_ptrs[row + 1]; nnz += warp_size) {
            mat.values[nnz] *= beta;
            if (row == mat.col_idxs[nnz]) {
                mat.values[nnz] += alpha;
            }
        }
    }
}


template <typename ValueType, typename IndexType, typename AlphaValues,
          typename BetaValues, typename Values>
__global__ void add_scaled_identity_kernel(
    const gko::batch::multi_vector::uniform_batch<const ValueType, AlphaValues>
        alpha,
    const gko::batch::multi_vector::uniform_batch<const ValueType, BetaValues>
        beta,
    const gko::batch::matrix::csr::uniform_batch<ValueType, IndexType, Values>
        mat)
{
    const size_type num_batch_items = mat.num_batch_items;
    for (size_type batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        const auto alpha_b = gko::batch::extract_batch_item(alpha, batch_id);
        const auto beta_b = gko::batch::extract_batch_item(beta, batch_id);
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        add_scaled_identity(alpha_b.values[0], beta_b.values[0], mat_b);
    }
}


}  // namespace batch_single_kernels
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif
