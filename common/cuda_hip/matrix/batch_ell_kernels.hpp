// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_MATRIX_BATCH_ELL_KERNELS_HPP_
#define GKO_COMMON_CUDA_HIP_MATRIX_BATCH_ELL_KERNELS_HPP_


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>

#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
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
    const gko::batch::matrix::ell::batch_item<const ValueType, IndexType,
                                              MatValues>& mat,
    const BValues b, const XValues x)
{
    using arithmetic_type = arithmetic_type_t<decltype(x)>;
    const auto num_rows = mat.num_rows;
    const auto num_stored_elements_per_row = mat.num_stored_elems_per_row;
    const auto stride = mat.stride;
    const auto val = mat.values;
    const auto col = mat.col_idxs;
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<arithmetic_type>();
        for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
            const auto ind = tidx + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                temp += val[ind] * b[col_idx];
            }
        }
        x[tidx] = temp;
    }
}

template <typename ValueType, typename IndexType, typename MatValues,
          typename BValues, typename XValues>
__global__ __launch_bounds__(default_block_size) void simple_apply_kernel(
    const gko::batch::matrix::ell::uniform_batch<const ValueType, IndexType,
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
    const gko::batch::matrix::ell::batch_item<const ValueType, IndexType,
                                              MatValues>& mat,
    const BValues b, const ArithmeticType beta, const XValues x)
{
    const auto num_rows = mat.num_rows;
    const auto num_stored_elements_per_row = mat.num_stored_elems_per_row;
    const auto stride = mat.stride;
    const auto val = mat.values;
    const auto col = mat.col_idxs;
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ArithmeticType>();
        for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
            const auto ind = tidx + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                temp += alpha * val[ind] * b[col_idx];
            }
        }
        x[tidx] = temp + beta * x[tidx];
    }
}

template <typename ValueType, typename IndexType, typename AlphaValues,
          typename MatValues, typename BValues, typename BetaValues,
          typename XValues>
__global__ __launch_bounds__(default_block_size) void advanced_apply_kernel(
    const gko::batch::multi_vector::uniform_batch<const ValueType, AlphaValues>
        alpha,
    const gko::batch::matrix::ell::uniform_batch<const ValueType, IndexType,
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


template <typename ScaleValues, typename ValueType, typename IndexType,
          typename Values>
__device__ __forceinline__ void scale(
    const ScaleValues col_scale, const ScaleValues row_scale,
    const gko::batch::matrix::ell::batch_item<ValueType, IndexType, Values>&
        mat)
{
    for (int tidx = threadIdx.x; tidx < mat.num_rows; tidx += blockDim.x) {
        const auto r_scale = row_scale[tidx];
        for (size_type idx = 0; idx < mat.num_stored_elems_per_row; idx++) {
            const auto ind = tidx + idx * mat.stride;
            const auto col_idx = mat.col_idxs[ind];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                mat.values[ind] *= r_scale * col_scale[col_idx];
            }
        }
    }
}


template <typename ScaleValues, typename ValueType, typename IndexType,
          typename Values>
__global__ void scale_kernel(
    const ScaleValues col_scale_vals, const ScaleValues row_scale_vals,
    const gko::batch::matrix::ell::uniform_batch<ValueType, IndexType, Values>
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
        scale(col_scale_b, row_scale_b, mat_b);
    }
}


template <typename ArithmeticType, typename ValueType, typename IndexType,
          typename Values>
__device__ __forceinline__ void add_scaled_identity(
    const ArithmeticType alpha, const ArithmeticType beta,
    const gko::batch::matrix::ell::batch_item<ValueType, IndexType, Values>&
        mat)
{
    for (int tidx = threadIdx.x; tidx < mat.num_rows; tidx += blockDim.x) {
        for (size_type idx = 0; idx < mat.num_stored_elems_per_row; idx++) {
            const auto ind = tidx + idx * mat.stride;
            mat.values[ind] *= beta;
            const auto col_idx = mat.col_idxs[ind];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                if (tidx == col_idx) {
                    mat.values[ind] += alpha;
                }
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
    const gko::batch::matrix::ell::uniform_batch<ValueType, IndexType, Values>
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
