// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/csr_kernels.hpp"

#include <algorithm>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>

#include "accessor/cuda_hip_helper.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/pointer_mode_guard.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/atomic.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/format_conversion.hpp"
#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/memory.hpp"
#include "common/cuda_hip/components/merging.hpp"
#include "common/cuda_hip/components/prefix_sum.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/segment_scan.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "core/base/array_access.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_accessor_helper.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup csr
 */
namespace csr {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;
constexpr int classical_oversubscription = 32;


/**
 * A compile-time list of the number items per threads for which spmv kernel
 * should be compiled.
 */
using compiled_kernels = syn::value_list<int, 3, 4, 6, 7, 8, 12, 14>;

using classical_kernels =
    syn::value_list<int, config::warp_size, 32, 16, 8, 4, 2, 1>;

using spgeam_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/cuda_hip/matrix/csr_common.hpp.inc"


namespace kernel {


template <typename T>
__host__ __device__ __forceinline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}


template <typename ValueType, typename IndexType>
__device__ __forceinline__ bool block_segment_scan_reverse(
    const IndexType* __restrict__ ind, ValueType* __restrict__ val)
{
    bool last = true;
    const auto reg_ind = ind[threadIdx.x];
#pragma unroll
    for (int i = 1; i < spmv_block_size; i <<= 1) {
        if (i == 1 && threadIdx.x < spmv_block_size - 1 &&
            reg_ind == ind[threadIdx.x + 1]) {
            last = false;
        }
        auto temp = zero<ValueType>();
        if (threadIdx.x >= i && reg_ind == ind[threadIdx.x - i]) {
            temp = val[threadIdx.x - i];
        }
        group::this_thread_block().sync();
        val[threadIdx.x] += temp;
        group::this_thread_block().sync();
    }

    return last;
}


template <bool overflow, typename IndexType>
__device__ __forceinline__ void find_next_row(
    const IndexType num_rows, const IndexType data_size, const IndexType ind,
    IndexType& row, IndexType& row_end, const IndexType row_predict,
    const IndexType row_predict_end, const IndexType* __restrict__ row_ptr)
{
    if (!overflow || ind < data_size) {
        if (ind >= row_end) {
            row = row_predict;
            row_end = row_predict_end;
            while (ind >= row_end) {
                row_end = row_ptr[++row + 1];
            }
        }

    } else {
        row = num_rows - 1;
        row_end = data_size;
    }
}


template <unsigned subwarp_size, typename ValueType, typename IndexType,
          typename output_accessor, typename Closure>
__device__ __forceinline__ void warp_atomic_add(
    const group::thread_block_tile<subwarp_size>& group, bool force_write,
    ValueType& val, const IndexType row, acc::range<output_accessor>& c,
    const IndexType column_id, Closure scale)
{
    // do a local scan to avoid atomic collisions
    const bool need_write = segment_scan(
        group, row, val, [](ValueType a, ValueType b) { return a + b; });
    if (need_write && force_write) {
        atomic_add(c->get_storage_address(row, column_id), scale(val));
    }
    if (!need_write || force_write) {
        val = zero<ValueType>();
    }
}


template <bool last, unsigned subwarp_size, typename arithmetic_type,
          typename matrix_accessor, typename IndexType, typename input_accessor,
          typename output_accessor, typename Closure>
__device__ __forceinline__ void process_window(
    const group::thread_block_tile<subwarp_size>& group,
    const IndexType num_rows, const IndexType data_size, const IndexType ind,
    IndexType& row, IndexType& row_end, IndexType& nrow, IndexType& nrow_end,
    arithmetic_type& temp_val, acc::range<matrix_accessor> val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, acc::range<input_accessor> b,
    acc::range<output_accessor> c, const IndexType column_id, Closure scale)
{
    const auto curr_row = row;
    find_next_row<last>(num_rows, data_size, ind, row, row_end, nrow, nrow_end,
                        row_ptrs);
    // segmented scan
    if (group.any(curr_row != row)) {
        warp_atomic_add(group, curr_row != row, temp_val, curr_row, c,
                        column_id, scale);
        nrow = group.shfl(row, subwarp_size - 1);
        nrow_end = group.shfl(row_end, subwarp_size - 1);
    }

    if (!last || ind < data_size) {
        const auto col = col_idxs[ind];
        temp_val += val(ind) * b(col, column_id);
    }
}


template <typename IndexType>
__device__ __forceinline__ IndexType get_warp_start_idx(
    const IndexType nwarps, const IndexType nnz, const IndexType warp_idx)
{
    const long long cache_lines = ceildivT<IndexType>(nnz, config::warp_size);
    return (warp_idx * cache_lines / nwarps) * config::warp_size;
}


template <typename matrix_accessor, typename input_accessor,
          typename output_accessor, typename IndexType, typename Closure>
__device__ __forceinline__ void spmv_kernel(
    const IndexType nwarps, const IndexType num_rows,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c, Closure scale)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    const IndexType warp_idx = blockIdx.x * warps_in_block + threadIdx.y;
    const IndexType column_id = blockIdx.y;
    if (warp_idx >= nwarps) {
        return;
    }
    const IndexType data_size = row_ptrs[num_rows];
    const IndexType start = get_warp_start_idx(nwarps, data_size, warp_idx);
    constexpr IndexType wsize = config::warp_size;
    const IndexType end =
        min(get_warp_start_idx(nwarps, data_size, warp_idx + 1),
            ceildivT<IndexType>(data_size, wsize) * wsize);
    auto row = srow[warp_idx];
    auto row_end = row_ptrs[row + 1];
    auto nrow = row;
    auto nrow_end = row_end;
    auto temp_val = zero<arithmetic_type>();
    IndexType ind = start + threadIdx.x;
    find_next_row<true>(num_rows, data_size, ind, row, row_end, nrow, nrow_end,
                        row_ptrs);
    const IndexType ind_end = end - wsize;
    const auto tile_block =
        group::tiled_partition<wsize>(group::this_thread_block());
    for (; ind < ind_end; ind += wsize) {
        process_window<false>(tile_block, num_rows, data_size, ind, row,
                              row_end, nrow, nrow_end, temp_val, val, col_idxs,
                              row_ptrs, b, c, column_id, scale);
    }
    process_window<true>(tile_block, num_rows, data_size, ind, row, row_end,
                         nrow, nrow_end, temp_val, val, col_idxs, row_ptrs, b,
                         c, column_id, scale);
    warp_atomic_add(tile_block, true, temp_val, row, c, column_id, scale);
}


template <typename matrix_accessor, typename input_accessor,
          typename output_accessor, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    using output_type = typename output_accessor::storage_type;
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, c,
                [](const arithmetic_type& x) {
                    // using atomic add to accumluate data, so it needs to be
                    // the output storage type
                    // TODO: Does it make sense to use atomicCAS when the
                    // arithmetic_type and output_type are different? It may
                    // allow the non floating point storage or more precise
                    // result.
                    return static_cast<output_type>(x);
                });
}


template <typename matrix_accessor, typename input_accessor,
          typename output_accessor, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const typename matrix_accessor::storage_type* __restrict__ alpha,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    using output_type = typename output_accessor::storage_type;
    const auto scale_factor = static_cast<arithmetic_type>(alpha[0]);
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, c,
                [&scale_factor](const arithmetic_type& x) {
                    return static_cast<output_type>(scale_factor * x);
                });
}


template <typename IndexType>
__forceinline__ __device__ void merge_path_search(
    const IndexType diagonal, const IndexType a_len, const IndexType b_len,
    const IndexType* __restrict__ a, const IndexType offset_b,
    IndexType* __restrict__ x, IndexType* __restrict__ y)
{
    auto x_min = max(diagonal - b_len, zero<IndexType>());
    auto x_max = min(diagonal, a_len);
    while (x_min < x_max) {
        auto pivot = x_min + (x_max - x_min) / 2;
        if (a[pivot] <= offset_b + diagonal - pivot - 1) {
            x_min = pivot + 1;
        } else {
            x_max = pivot;
        }
    }

    *x = min(x_min, a_len);
    *y = diagonal - x_min;
}


template <typename arithmetic_type, typename IndexType,
          typename output_accessor, typename Alpha_op>
__device__ void merge_path_reduce(const IndexType nwarps,
                                  const arithmetic_type* __restrict__ last_val,
                                  const IndexType* __restrict__ last_row,
                                  acc::range<output_accessor> c,
                                  Alpha_op alpha_op)
{
    const IndexType cache_lines = ceildivT<IndexType>(nwarps, spmv_block_size);
    const IndexType tid = threadIdx.x;
    const IndexType start = min(tid * cache_lines, nwarps);
    const IndexType end = min((tid + 1) * cache_lines, nwarps);
    auto value = zero<arithmetic_type>();
    IndexType row = last_row[nwarps - 1];
    if (start < nwarps) {
        value = last_val[start];
        row = last_row[start];
        for (IndexType i = start + 1; i < end; i++) {
            if (last_row[i] != row) {
                c(row, 0) += alpha_op(value);
                row = last_row[i];
                value = last_val[i];
            } else {
                value += last_val[i];
            }
        }
    }
    __shared__ IndexType tmp_ind[spmv_block_size];
    __shared__ uninitialized_array<arithmetic_type, spmv_block_size> tmp_val;
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = row;
    group::this_thread_block().sync();
    bool last =
        block_segment_scan_reverse(static_cast<IndexType*>(tmp_ind),
                                   static_cast<arithmetic_type*>(tmp_val));
    group::this_thread_block().sync();
    if (last) {
        c(row, 0) += alpha_op(tmp_val[threadIdx.x]);
    }
}


template <int items_per_thread, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType,
          typename Alpha_op, typename Beta_op>
__device__ void merge_path_spmv(
    const IndexType num_rows, acc::range<matrix_accessor> val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c,
    IndexType* __restrict__ row_out,
    typename output_accessor::arithmetic_type* __restrict__ val_out,
    Alpha_op alpha_op, Beta_op beta_op)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    const auto* row_end_ptrs = row_ptrs + 1;
    const auto nnz = row_ptrs[num_rows];
    const IndexType num_merge_items = num_rows + nnz;
    const auto block_items = spmv_block_size * items_per_thread;
    __shared__ IndexType shared_row_ptrs[block_items];
    const IndexType diagonal =
        min(IndexType(block_items * blockIdx.x), num_merge_items);
    const IndexType diagonal_end = min(diagonal + block_items, num_merge_items);
    IndexType block_start_x;
    IndexType block_start_y;
    IndexType end_x;
    IndexType end_y;
    merge_path_search(diagonal, num_rows, nnz, row_end_ptrs, zero<IndexType>(),
                      &block_start_x, &block_start_y);
    merge_path_search(diagonal_end, num_rows, nnz, row_end_ptrs,
                      zero<IndexType>(), &end_x, &end_y);
    const IndexType block_num_rows = end_x - block_start_x;
    const IndexType block_num_nonzeros = end_y - block_start_y;
    for (int i = threadIdx.x;
         i < block_num_rows && block_start_x + i < num_rows;
         i += spmv_block_size) {
        shared_row_ptrs[i] = row_end_ptrs[block_start_x + i];
    }
    group::this_thread_block().sync();

    IndexType start_x;
    IndexType start_y;
    merge_path_search(IndexType(items_per_thread * threadIdx.x), block_num_rows,
                      block_num_nonzeros, shared_row_ptrs, block_start_y,
                      &start_x, &start_y);


    IndexType ind = block_start_y + start_y;
    IndexType row_i = block_start_x + start_x;
    auto value = zero<arithmetic_type>();
#pragma unroll
    for (IndexType i = 0; i < items_per_thread; i++) {
        if (row_i < num_rows) {
            if (start_x == block_num_rows || ind < shared_row_ptrs[start_x]) {
                value += val(ind) * b(col_idxs[ind], 0);
                ind++;
            } else {
                c(row_i, 0) = alpha_op(value) + beta_op(c(row_i, 0));
                start_x++;
                row_i++;
                value = zero<arithmetic_type>();
            }
        }
    }
    group::this_thread_block().sync();
    IndexType* tmp_ind = shared_row_ptrs;
    arithmetic_type* tmp_val =
        reinterpret_cast<arithmetic_type*>(shared_row_ptrs + spmv_block_size);
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = row_i;
    group::this_thread_block().sync();
    bool last = block_segment_scan_reverse(tmp_ind, tmp_val);
    if (threadIdx.x == spmv_block_size - 1) {
        row_out[blockIdx.x] = min(end_x, num_rows - 1);
        val_out[blockIdx.x] = tmp_val[threadIdx.x];
    } else if (last) {
        c(row_i, 0) += alpha_op(tmp_val[threadIdx.x]);
    }
}

template <int items_per_thread, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_merge_path_spmv(
    const IndexType num_rows, acc::range<matrix_accessor> val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c,
    IndexType* __restrict__ row_out,
    typename output_accessor::arithmetic_type* __restrict__ val_out)
{
    using type = typename output_accessor::arithmetic_type;
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, c, row_out, val_out,
        [](const type& x) { return x; },
        [](const type& x) { return zero<type>(); });
}


template <int items_per_thread, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_merge_path_spmv(
    const IndexType num_rows,
    const typename matrix_accessor::storage_type* __restrict__ alpha,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b,
    const typename output_accessor::storage_type* __restrict__ beta,
    acc::range<output_accessor> c, IndexType* __restrict__ row_out,
    typename output_accessor::arithmetic_type* __restrict__ val_out)
{
    using type = typename output_accessor::arithmetic_type;
    const type alpha_val = alpha[0];
    const type beta_val = beta[0];
    if (is_zero(beta_val)) {
        merge_path_spmv<items_per_thread>(
            num_rows, val, col_idxs, row_ptrs, srow, b, c, row_out, val_out,
            [&alpha_val](const type& x) { return alpha_val * x; },
            [](const type& x) { return zero<type>(); });
    } else {
        merge_path_spmv<items_per_thread>(
            num_rows, val, col_idxs, row_ptrs, srow, b, c, row_out, val_out,
            [&alpha_val](const type& x) { return alpha_val * x; },
            [&beta_val](const type& x) { return beta_val * x; });
    }
}


template <typename arithmetic_type, typename IndexType,
          typename output_accessor>
__global__ __launch_bounds__(spmv_block_size) void abstract_reduce(
    const IndexType nwarps, const arithmetic_type* __restrict__ last_val,
    const IndexType* __restrict__ last_row, acc::range<output_accessor> c)
{
    merge_path_reduce(nwarps, last_val, last_row, c,
                      [](const arithmetic_type& x) { return x; });
}


template <typename arithmetic_type, typename MatrixValueType,
          typename IndexType, typename output_accessor>
__global__ __launch_bounds__(spmv_block_size) void abstract_reduce(
    const IndexType nwarps, const arithmetic_type* __restrict__ last_val,
    const IndexType* __restrict__ last_row,
    const MatrixValueType* __restrict__ alpha, acc::range<output_accessor> c)
{
    const auto alpha_val = static_cast<arithmetic_type>(alpha[0]);
    merge_path_reduce(
        nwarps, last_val, last_row, c,
        [&alpha_val](const arithmetic_type& x) { return alpha_val * x; });
}


template <size_type subwarp_size, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType,
          typename Closure>
__device__ void device_classical_spmv(const size_type num_rows,
                                      acc::range<matrix_accessor> val,
                                      const IndexType* __restrict__ col_idxs,
                                      const IndexType* __restrict__ row_ptrs,
                                      acc::range<input_accessor> b,
                                      acc::range<output_accessor> c,
                                      Closure scale)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    auto subwarp_tile =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto subrow = thread::get_subwarp_num_flat<subwarp_size>();
    const auto subid = subwarp_tile.thread_rank();
    // can not use auto for hip because the type is
    // __HIP_Coordinates<__HIP_BlockIdx>::__Y which is not allowed in accessor
    // operator()
    const int column_id = blockIdx.y;
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    for (; row < num_rows; row += subrow) {
        const auto ind_end = row_ptrs[row + 1];
        auto temp_val = zero<arithmetic_type>();
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subwarp_size) {
            temp_val += val(ind) * b(col_idxs[ind], column_id);
        }
        auto subwarp_result =
            reduce(subwarp_tile, temp_val,
                   [](const arithmetic_type& a, const arithmetic_type& b) {
                       return a + b;
                   });
        if (subid == 0) {
            c(row, column_id) = scale(subwarp_result, c(row, column_id));
        }
    }
}


template <size_type subwarp_size, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_classical_spmv(
    const size_type num_rows, acc::range<matrix_accessor> val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, acc::range<input_accessor> b,
    acc::range<output_accessor> c)
{
    using type = typename output_accessor::arithmetic_type;
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, c,
        [](const type& x, const type& y) { return x; });
}


template <size_type subwarp_size, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_classical_spmv(
    const size_type num_rows,
    const typename matrix_accessor::storage_type* __restrict__ alpha,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, acc::range<input_accessor> b,
    const typename output_accessor::storage_type* __restrict__ beta,
    acc::range<output_accessor> c)
{
    using type = typename output_accessor::arithmetic_type;
    const type alpha_val = alpha[0];
    const type beta_val = beta[0];
    if (is_zero(beta_val)) {
        device_classical_spmv<subwarp_size>(
            num_rows, val, col_idxs, row_ptrs, b, c,
            [&alpha_val](const type& x, const type& y) {
                return alpha_val * x;
            });
    } else {
        device_classical_spmv<subwarp_size>(
            num_rows, val, col_idxs, row_ptrs, b, c,
            [&alpha_val, &beta_val](const type& x, const type& y) {
                return alpha_val * x + beta_val * y;
            });
    }
}


template <int subwarp_size, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spgeam_nnz(
    const IndexType* __restrict__ a_row_ptrs,
    const IndexType* __restrict__ a_col_idxs,
    const IndexType* __restrict__ b_row_ptrs,
    const IndexType* __restrict__ b_col_idxs, IndexType num_rows,
    IndexType* __restrict__ nnz)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (row >= num_rows) {
        return;
    }

    const auto a_begin = a_row_ptrs[row];
    const auto b_begin = b_row_ptrs[row];
    const auto a_size = a_row_ptrs[row + 1] - a_begin;
    const auto b_size = b_row_ptrs[row + 1] - b_begin;
    IndexType count{};
    group_merge<subwarp_size>(
        a_col_idxs + a_begin, a_size, b_col_idxs + b_begin, b_size, subwarp,
        [&](IndexType, IndexType a_col, IndexType, IndexType b_col, IndexType,
            bool valid) {
            count += popcnt(subwarp.ballot(a_col != b_col && valid));
            return true;
        });

    if (subwarp.thread_rank() == 0) {
        nnz[row] = count;
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spgeam(
    const ValueType* __restrict__ palpha,
    const IndexType* __restrict__ a_row_ptrs,
    const IndexType* __restrict__ a_col_idxs,
    const ValueType* __restrict__ a_vals, const ValueType* __restrict__ pbeta,
    const IndexType* __restrict__ b_row_ptrs,
    const IndexType* __restrict__ b_col_idxs,
    const ValueType* __restrict__ b_vals, IndexType num_rows,
    const IndexType* __restrict__ c_row_ptrs,
    IndexType* __restrict__ c_col_idxs, ValueType* __restrict__ c_vals)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (row >= num_rows) {
        return;
    }

    const auto alpha = palpha[0];
    const auto beta = pbeta[0];
    const auto lane = static_cast<IndexType>(subwarp.thread_rank());
    constexpr auto lanemask_full =
        ~config::lane_mask_type{} >> (config::warp_size - subwarp_size);
    const auto lanemask_eq = config::lane_mask_type{1} << lane;
    const auto lanemask_lt = lanemask_eq - 1;

    const auto a_begin = a_row_ptrs[row];
    const auto b_begin = b_row_ptrs[row];
    const auto a_size = a_row_ptrs[row + 1] - a_begin;
    const auto b_size = b_row_ptrs[row + 1] - b_begin;
    auto c_begin = c_row_ptrs[row];
    bool skip_first{};
    group_merge<subwarp_size>(
        a_col_idxs + a_begin, a_size, b_col_idxs + b_begin, b_size, subwarp,
        [&](IndexType a_nz, IndexType a_col, IndexType b_nz, IndexType b_col,
            IndexType, bool valid) {
            auto c_col = min(a_col, b_col);
            auto equal_mask = subwarp.ballot(a_col == b_col && valid);
            // check if the elements in the previous merge step are
            // equal
            auto prev_equal_mask = equal_mask << 1 | skip_first;
            // store the highest bit for the next group_merge_step
            skip_first = bool(equal_mask >> (subwarp_size - 1));
            auto prev_equal = bool(prev_equal_mask & lanemask_eq);
            // only output an entry if the previous cols weren't equal.
            // if they were equal, they were both handled in the
            // previous step
            if (valid && !prev_equal) {
                auto c_ofs = popcnt(~prev_equal_mask & lanemask_lt);
                c_col_idxs[c_begin + c_ofs] = c_col;
                auto a_val =
                    a_col <= b_col ? a_vals[a_nz + a_begin] : zero<ValueType>();
                auto b_val =
                    b_col <= a_col ? b_vals[b_nz + b_begin] : zero<ValueType>();
                c_vals[c_begin + c_ofs] = alpha * a_val + beta * b_val;
            }
            // advance by the number of merged elements
            // in theory, we would need to mask by `valid`, but this
            // would only be false somewhere in the last iteration, where
            // we don't need the value of c_begin afterwards, anyways.
            c_begin += popcnt(~prev_equal_mask & lanemask_full);
            return true;
        });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type num_rows, const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values, size_type stride,
    ValueType* __restrict__ result)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_rows) {
        for (auto i = row_ptrs[tidx]; i < row_ptrs[tidx + 1]; i++) {
            result[stride * tidx + col_idxs[i]] = values[i];
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void extract_diagonal(
    size_type diag_size, size_type nnz,
    const ValueType* __restrict__ orig_values,
    const IndexType* __restrict__ orig_row_ptrs,
    const IndexType* __restrict__ orig_col_idxs, ValueType* __restrict__ diag)
{
    constexpr auto warp_size = config::warp_size;
    const auto row = thread::get_subwarp_id_flat<warp_size>();
    const auto local_tidx = threadIdx.x % warp_size;

    if (row < diag_size) {
        for (size_type i = local_tidx;
             i < orig_row_ptrs[row + 1] - orig_row_ptrs[row]; i += warp_size) {
            const auto orig_idx = i + orig_row_ptrs[row];
            if (orig_idx < nnz) {
                if (orig_col_idxs[orig_idx] == row) {
                    diag[row] = orig_values[orig_idx];
                    return;
                }
            }
        }
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void row_ptr_permute(
    size_type num_rows, const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs, IndexType* __restrict__ out_nnz)
{
    auto tid = thread::get_thread_id_flat();
    if (tid >= num_rows) {
        return;
    }
    const auto in_row = permutation[tid];
    const auto out_row = tid;
    out_nnz[out_row] = in_row_ptrs[in_row + 1] - in_row_ptrs[in_row];
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void inv_row_ptr_permute(
    size_type num_rows, const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs, IndexType* __restrict__ out_nnz)
{
    auto tid = thread::get_thread_id_flat();
    if (tid >= num_rows) {
        return;
    }
    const auto in_row = tid;
    const auto out_row = permutation[tid];
    out_nnz[out_row] = in_row_ptrs[in_row + 1] - in_row_ptrs[in_row];
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void row_permute(
    size_type num_rows, const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    const auto lane = threadIdx.x % subwarp_size;
    const auto in_row = permutation[tid];
    const auto out_row = tid;
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void inv_row_permute(
    size_type num_rows, const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    const auto lane = threadIdx.x % subwarp_size;
    const auto in_row = tid;
    const auto out_row = permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void inv_symm_permute(
    size_type num_rows, const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    const auto lane = threadIdx.x % subwarp_size;
    const auto in_row = tid;
    const auto out_row = permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = permutation[in_cols[in_begin + i]];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void inv_nonsymm_permute(
    size_type num_rows, const IndexType* __restrict__ row_permutation,
    const IndexType* __restrict__ col_permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    const auto lane = threadIdx.x % subwarp_size;
    const auto in_row = tid;
    const auto out_row = row_permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = col_permutation[in_cols[in_begin + i]];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void row_scale_permute(
    size_type num_rows, const ValueType* __restrict__ scale,
    const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    const auto lane = threadIdx.x % subwarp_size;
    const auto in_row = permutation[tid];
    const auto out_row = tid;
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i] * scale[in_row];
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void inv_row_scale_permute(
    size_type num_rows, const ValueType* __restrict__ scale,
    const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    const auto lane = threadIdx.x % subwarp_size;
    const auto in_row = tid;
    const auto out_row = permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i] / scale[out_row];
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void inv_symm_scale_permute(
    size_type num_rows, const ValueType* __restrict__ scale,
    const IndexType* __restrict__ permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    const auto lane = threadIdx.x % subwarp_size;
    const auto in_row = tid;
    const auto out_row = permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        const auto out_col = permutation[in_cols[in_begin + i]];
        out_cols[out_begin + i] = out_col;
        out_vals[out_begin + i] =
            in_vals[in_begin + i] / (scale[out_row] * scale[out_col]);
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void inv_nonsymm_scale_permute(
    size_type num_rows, const ValueType* __restrict__ row_scale,
    const IndexType* __restrict__ row_permutation,
    const ValueType* __restrict__ col_scale,
    const IndexType* __restrict__ col_permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>();
    if (tid >= num_rows) {
        return;
    }
    const auto lane = threadIdx.x % subwarp_size;
    const auto in_row = tid;
    const auto out_row = row_permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        const auto out_col = col_permutation[in_cols[in_begin + i]];
        out_cols[out_begin + i] = out_col;
        out_vals[out_begin + i] =
            in_vals[in_begin + i] / (row_scale[out_row] * col_scale[out_col]);
    }
}


template <typename ValueType, typename IndexType>
__global__
__launch_bounds__(default_block_size) void compute_submatrix_idxs_and_vals(
    const size_type num_rows, const size_type num_cols,
    const size_type row_offset, const size_type col_offset,
    const IndexType* __restrict__ source_row_ptrs,
    const IndexType* __restrict__ source_col_idxs,
    const ValueType* __restrict__ source_values,
    const IndexType* __restrict__ result_row_ptrs,
    IndexType* __restrict__ result_col_idxs,
    ValueType* __restrict__ result_values)
{
    const auto res_row = thread::get_thread_id_flat();
    if (res_row < num_rows) {
        const auto src_row = res_row + row_offset;
        auto res_nnz = result_row_ptrs[res_row];
        for (auto nnz = source_row_ptrs[src_row];
             nnz < source_row_ptrs[src_row + 1]; ++nnz) {
            const auto res_col =
                source_col_idxs[nnz] - static_cast<IndexType>(col_offset);
            if (res_col < num_cols && res_col >= 0) {
                result_col_idxs[res_nnz] = res_col;
                result_values[res_nnz] = source_values[nnz];
                res_nnz++;
            }
        }
    }
}


template <typename IndexType>
__global__
__launch_bounds__(default_block_size) void calculate_nnz_per_row_in_span(
    const span row_span, const span col_span,
    const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs, IndexType* __restrict__ nnz_per_row)
{
    const auto src_row = thread::get_thread_id_flat() + row_span.begin;
    if (src_row < row_span.end) {
        IndexType nnz{};
        for (auto i = row_ptrs[src_row]; i < row_ptrs[src_row + 1]; ++i) {
            if (col_idxs[i] >= col_span.begin && col_idxs[i] < col_span.end) {
                nnz++;
            }
        }
        nnz_per_row[src_row - row_span.begin] = nnz;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void add_scaled_identity(
    const ValueType* const __restrict__ alpha,
    const ValueType* const __restrict__ beta, const IndexType num_rows,
    const IndexType* const __restrict__ row_ptrs,
    const IndexType* const __restrict__ col_idxs,
    ValueType* const __restrict__ values)
{
    constexpr int warp_size = config::warp_size;
    auto tile_grp =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const auto warpid = thread::get_subwarp_id_flat<warp_size, IndexType>();
    if (warpid < num_rows) {
        const auto tid_in_warp = tile_grp.thread_rank();
        const IndexType row_start = row_ptrs[warpid];
        const IndexType num_nz = row_ptrs[warpid + 1] - row_start;
        const auto beta_val = beta[0];
        const auto alpha_val = alpha[0];
        for (IndexType iz = tid_in_warp; iz < num_nz; iz += warp_size) {
            if (beta_val != one<ValueType>()) {
                values[iz + row_start] *= beta_val;
            }
            if (col_idxs[iz + row_start] == warpid &&
                alpha_val != zero<ValueType>()) {
                values[iz + row_start] += alpha_val;
            }
        }
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* source, int bs,
                      array<IndexType>& block_row_ptr_array,
                      array<IndexType>& block_col_idx_array,
                      array<ValueType>& block_value_array)
{
    using tuple_type = thrust::tuple<IndexType, IndexType>;
    const auto nnz = source->get_num_stored_elements();
    array<IndexType> in_row_idxs{exec, nnz};
    array<IndexType> in_col_idxs{exec, nnz};
    array<ValueType> in_values{exec, nnz};
    exec->copy(nnz, source->get_const_col_idxs(), in_col_idxs.get_data());
    exec->copy(nnz, source->get_const_values(), in_values.get_data());
    components::convert_ptrs_to_idxs(exec, source->get_const_row_ptrs(),
                                     source->get_size()[0],
                                     in_row_idxs.get_data());
    auto block_row_ptrs = block_row_ptr_array.get_data();
    auto num_block_rows = block_row_ptr_array.get_size() - 1;
    if (nnz == 0) {
        components::fill_array(exec, block_row_ptrs, num_block_rows + 1,
                               IndexType{});
        block_col_idx_array.resize_and_reset(0);
        block_value_array.resize_and_reset(0);
        return;
    }
    auto in_rows = in_row_idxs.get_data();
    auto in_cols = in_col_idxs.get_data();
    auto in_vals = as_device_type(in_values.get_data());
    auto in_loc_it =
        thrust::make_zip_iterator(thrust::make_tuple(in_rows, in_cols));
    thrust::sort_by_key(thrust_policy(exec), in_loc_it, in_loc_it + nnz,
                        in_vals, [bs] __device__(tuple_type a, tuple_type b) {
                            return thrust::make_pair(thrust::get<0>(a) / bs,
                                                     thrust::get<1>(a) / bs) <
                                   thrust::make_pair(thrust::get<0>(b) / bs,
                                                     thrust::get<1>(b) / bs);
                        });
    // build block pattern
    auto adj_predicate = [bs, in_rows, in_cols, nnz] __device__(size_type i) {
        const auto a_block_row = i > 0 ? in_rows[i - 1] / bs : -1;
        const auto a_block_col = i > 0 ? in_cols[i - 1] / bs : -1;
        const auto b_block_row = in_rows[i] / bs;
        const auto b_block_col = in_cols[i] / bs;
        return (a_block_row != b_block_row) || (a_block_col != b_block_col);
    };
    auto iota = thrust::make_counting_iterator(size_type{});
    // count how many blocks we have by counting how often the block changes
    auto num_blocks = static_cast<size_type>(
        thrust::count_if(thrust_policy(exec), iota, iota + nnz, adj_predicate));
    // allocate storage
    array<IndexType> block_row_idx_array{exec, num_blocks};
    array<size_type> block_ptr_array{exec, num_blocks};
    block_col_idx_array.resize_and_reset(num_blocks);
    block_value_array.resize_and_reset(num_blocks * bs * bs);
    auto row_idxs = block_row_idx_array.get_data();
    auto col_idxs = block_col_idx_array.get_data();
    auto values = as_device_type(block_value_array.get_data());
    auto block_ptrs = block_ptr_array.get_data();
    // write (block_row, block_col, block_start_idx) tuples for each block
    thrust::copy_if(thrust_policy(exec), iota, iota + nnz, block_ptrs,
                    adj_predicate);
    auto block_output_it =
        thrust::make_zip_iterator(thrust::make_tuple(row_idxs, col_idxs));
    thrust::transform(
        thrust_policy(exec), block_ptrs, block_ptrs + num_blocks,
        block_output_it, [bs, in_rows, in_cols] __device__(size_type i) {
            return thrust::make_tuple(in_rows[i] / bs, in_cols[i] / bs);
        });
    // build row pointers from row indices
    components::convert_idxs_to_ptrs(exec, block_row_idx_array.get_const_data(),
                                     block_row_idx_array.get_size(),
                                     num_block_rows, block_row_ptrs);
    // fill in values
    components::fill_array(exec, block_value_array.get_data(),
                           num_blocks * bs * bs, zero<ValueType>());
    thrust::for_each_n(thrust_policy(exec), iota, num_blocks,
                       [block_ptrs, nnz, num_blocks, bs, in_rows, in_cols,
                        in_vals, values] __device__(size_type i) {
                           const auto block_begin = block_ptrs[i];
                           const auto block_end =
                               i < num_blocks - 1 ? block_ptrs[i + 1] : nnz;
                           for (auto nz = block_begin; nz < block_end; nz++) {
                               values[i * bs * bs + (in_cols[nz] % bs) * bs +
                                      (in_rows[nz] % bs)] = in_vals[nz];
                           }
                       });
}


namespace kernel {


template <int subwarp_size, typename IndexType>
__global__ __launch_bounds__(default_block_size) void build_csr_lookup(
    const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs, size_type num_rows,
    matrix::csr::sparsity_type allowed, const IndexType* storage_offsets,
    int64* __restrict__ row_desc, int32* __restrict__ storage)
{
    using matrix::csr::sparsity_type;
    constexpr int bitmap_block_size = matrix::csr::sparsity_bitmap_block_size;
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    if (row >= num_rows) {
        return;
    }
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto row_begin = row_ptrs[row];
    const auto row_len = row_ptrs[row + 1] - row_begin;
    const auto storage_begin = storage_offsets[row];
    const auto available_storage = storage_offsets[row + 1] - storage_begin;
    const auto local_storage = storage + storage_begin;
    const auto local_cols = col_idxs + row_begin;
    const auto lane = subwarp.thread_rank();
    const auto min_col = row_len > 0 ? local_cols[0] : 0;
    const auto col_range =
        row_len > 0 ? local_cols[row_len - 1] - min_col + 1 : 0;

    // full column range
    if (col_range == row_len &&
        csr_lookup_allowed(allowed, sparsity_type::full)) {
        if (lane == 0) {
            row_desc[row] = static_cast<int64>(sparsity_type::full);
        }
        return;
    }
    // dense bitmap storage
    const auto num_blocks =
        static_cast<int32>(ceildiv(col_range, bitmap_block_size));
    if (num_blocks * 2 <= available_storage &&
        csr_lookup_allowed(allowed, sparsity_type::bitmap)) {
        if (lane == 0) {
            row_desc[row] = (static_cast<int64>(num_blocks) << 32) |
                            static_cast<int64>(sparsity_type::bitmap);
        }
        const auto block_ranks = local_storage;
        const auto block_bitmaps =
            reinterpret_cast<uint32*>(block_ranks + num_blocks);
        // fill bitmaps with zeros
        for (int32 i = lane; i < num_blocks; i += subwarp_size) {
            block_bitmaps[i] = 0;
        }
        // fill bitmaps with sparsity pattern
        for (IndexType base_i = 0; base_i < row_len; base_i += subwarp_size) {
            const auto i = base_i + lane;
            const auto col = i < row_len
                                 ? local_cols[i]
                                 : device_numeric_limits<IndexType>::max();
            const auto rel_col = static_cast<int32>(col - min_col);
            const auto block = rel_col / bitmap_block_size;
            const auto col_in_block = rel_col % bitmap_block_size;
            auto local_bitmap = uint32{i < row_len ? 1u : 0u} << col_in_block;
            bool is_first =
                segment_scan(subwarp, block, local_bitmap,
                             [](config::lane_mask_type a,
                                config::lane_mask_type b) { return a | b; });
            // memory barrier - just to be sure
            subwarp.sync();
            if (is_first && i < row_len) {
                block_bitmaps[block] |= local_bitmap;
            }
        }
        // compute bitmap ranks
        int32 block_partial_sum{};
        for (int32 base_i = 0; base_i < num_blocks; base_i += subwarp_size) {
            const auto i = base_i + lane;
            const auto bitmap = i < num_blocks ? block_bitmaps[i] : 0;
            int32 local_partial_sum{};
            int32 local_total_sum{};
            subwarp_prefix_sum<false>(popcnt(bitmap), local_partial_sum,
                                      local_total_sum, subwarp);
            if (i < num_blocks) {
                block_ranks[i] = local_partial_sum + block_partial_sum;
            }
            block_partial_sum += local_total_sum;
        }
        return;
    }
    // if hash lookup is not allowed, we are done here
    if (!csr_lookup_allowed(allowed, sparsity_type::hash)) {
        if (lane == 0) {
            row_desc[row] = static_cast<int64>(sparsity_type::none);
        }
        return;
    }
    // sparse hashmap storage
    // we need at least one unfilled entry to avoid infinite loops on search
    GKO_ASSERT(row_len < available_storage);
    constexpr double inv_golden_ratio = 0.61803398875;
    // use golden ratio as approximation for hash parameter that spreads
    // consecutive values as far apart as possible. Ensure lowest bit is set
    // otherwise we skip odd hashtable entries
    const auto hash_parameter =
        1u | static_cast<uint32>(available_storage * inv_golden_ratio);
    if (lane == 0) {
        row_desc[row] = (static_cast<int64>(hash_parameter) << 32) |
                        static_cast<int64>(sparsity_type::hash);
    }
    // fill hashmap with sentinel
    constexpr int32 empty = invalid_index<int32>();
    for (int32 i = lane; i < available_storage; i += subwarp_size) {
        local_storage[i] = empty;
    }
    // memory barrier
    subwarp.sync();
    // fill with actual entries
    for (IndexType base_i = 0; base_i < row_len; base_i += subwarp_size) {
        const auto i = base_i + lane;
        const auto col = i < row_len ? local_cols[i] : 0;
        // make sure that each idle thread gets a unique out-of-bounds value
        auto hash = i < row_len ? (static_cast<uint32>(col) * hash_parameter) %
                                      static_cast<uint32>(available_storage)
                                : static_cast<uint32>(available_storage + lane);
        // collision resolution
#if !defined(__HIP_DEVICE_COMPILE__) && defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ >= 700)
        const auto this_lane_mask = config::lane_mask_type{1} << lane;
        const auto lane_prefix_mask = this_lane_mask - 1;
        // memory barrier to previous loop iteration
        subwarp.sync();
        int32 entry = i < row_len ? local_storage[hash] : empty;
        // find all threads in the subwarp with the same hash key
        auto colliding = subwarp.match_any(hash);
        // if there are any collisions with previously filled buckets
        while (subwarp.any(entry != empty || colliding != this_lane_mask)) {
            if (entry != empty || colliding != this_lane_mask) {
                // assign consecutive indices to matching threads
                hash += (entry == empty ? 0 : 1) +
                        popcnt(colliding & lane_prefix_mask);
                // cheap modulo replacement
                if (hash >= available_storage) {
                    hash -= available_storage;
                }
                // this could only fail for available_storage < warp_size, as
                // popcnt(colliding) is at most warp_size. At the same time, we
                // only increase hash by row_length at most, so this is still
                // safe.
                GKO_ASSERT(hash < available_storage);
                entry = local_storage[hash];
            }
            colliding = subwarp.match_any(hash);
        }
        if (i < row_len) {
            local_storage[hash] = i;
        }
#else
        if (i < row_len) {
            while (atomic_cas_relaxed_local(local_storage + hash, empty,
                                            static_cast<int32>(i)) != empty) {
                hash++;
                if (hash >= available_storage) {
                    hash = 0;
                }
            }
        }
#endif
    }
}


}  // namespace kernel


template <typename IndexType>
void build_lookup(std::shared_ptr<const DefaultExecutor> exec,
                  const IndexType* row_ptrs, const IndexType* col_idxs,
                  size_type num_rows, matrix::csr::sparsity_type allowed,
                  const IndexType* storage_offsets, int64* row_desc,
                  int32* storage)
{
    const auto num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    kernel::build_csr_lookup<config::warp_size>
        <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
            row_ptrs, col_idxs, num_rows, allowed, storage_offsets, row_desc,
            storage);
}


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void spgeam(syn::value_list<int, subwarp_size>,
            std::shared_ptr<const DefaultExecutor> exec, const ValueType* alpha,
            const IndexType* a_row_ptrs, const IndexType* a_col_idxs,
            const ValueType* a_vals, const ValueType* beta,
            const IndexType* b_row_ptrs, const IndexType* b_col_idxs,
            const ValueType* b_vals, matrix::Csr<ValueType, IndexType>* c)
{
    auto m = static_cast<IndexType>(c->get_size()[0]);
    auto c_row_ptrs = c->get_row_ptrs();
    // count nnz for alpha * A + beta * B
    auto subwarps_per_block = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(m, subwarps_per_block);
    if (num_blocks > 0) {
        kernel::spgeam_nnz<subwarp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                a_row_ptrs, a_col_idxs, b_row_ptrs, b_col_idxs, m, c_row_ptrs);
    }

    // build row pointers
    components::prefix_sum_nonnegative(exec, c_row_ptrs, m + 1);

    // accumulate non-zeros for alpha * A + beta * B
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto c_nnz = exec->copy_val_to_host(c_row_ptrs + m);
    c_builder.get_col_idx_array().resize_and_reset(c_nnz);
    c_builder.get_value_array().resize_and_reset(c_nnz);
    auto c_col_idxs = c->get_col_idxs();
    auto c_vals = c->get_values();
    if (num_blocks > 0) {
        kernel::spgeam<subwarp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                as_device_type(alpha), a_row_ptrs, a_col_idxs,
                as_device_type(a_vals), as_device_type(beta), b_row_ptrs,
                b_col_idxs, as_device_type(b_vals), m, c_row_ptrs, c_col_idxs,
                as_device_type(c_vals));
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_spgeam, spgeam);


}  // namespace


template <typename ValueType, typename IndexType>
void spgeam(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType>* alpha,
            const matrix::Csr<ValueType, IndexType>* a,
            const matrix::Dense<ValueType>* beta,
            const matrix::Csr<ValueType, IndexType>* b,
            matrix::Csr<ValueType, IndexType>* c)
{
    auto total_nnz =
        a->get_num_stored_elements() + b->get_num_stored_elements();
    auto nnz_per_row = a->get_size()[0] ? total_nnz / a->get_size()[0] : 0;
    select_spgeam(
        spgeam_kernels(),
        [&](int compiled_subwarp_size) {
            return compiled_subwarp_size >= nnz_per_row ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec,
        alpha->get_const_values(), a->get_const_row_ptrs(),
        a->get_const_col_idxs(), a->get_const_values(),
        beta->get_const_values(), b->get_const_row_ptrs(),
        b->get_const_col_idxs(), b->get_const_values(), c);
}


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Csr<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();
    const auto row_ptrs = source->get_const_row_ptrs();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();

    auto grid_dim = ceildiv(num_rows, default_block_size);
    if (grid_dim > 0) {
        kernel::fill_in_dense<<<grid_dim, default_block_size, 0,
                                exec->get_stream()>>>(
            num_rows, as_device_type(row_ptrs), as_device_type(col_idxs),
            as_device_type(vals), stride, as_device_type(result->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void inv_symm_permute(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType* perm,
                      const matrix::Csr<ValueType, IndexType>* orig,
                      matrix::Csr<ValueType, IndexType>* permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    if (count_num_blocks > 0) {
        kernel::inv_row_ptr_permute<<<count_num_blocks, default_block_size, 0,
                                      exec->get_stream()>>>(
            num_rows, perm, orig->get_const_row_ptrs(),
            permuted->get_row_ptrs());
    }
    components::prefix_sum_nonnegative(exec, permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (copy_num_blocks > 0) {
        kernel::inv_symm_permute<config::warp_size>
            <<<copy_num_blocks, default_block_size, 0, exec->get_stream()>>>(
                num_rows, perm, orig->get_const_row_ptrs(),
                orig->get_const_col_idxs(),
                as_device_type(orig->get_const_values()),
                permuted->get_row_ptrs(), permuted->get_col_idxs(),
                as_device_type(permuted->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void inv_nonsymm_permute(std::shared_ptr<const DefaultExecutor> exec,
                         const IndexType* row_perm, const IndexType* col_perm,
                         const matrix::Csr<ValueType, IndexType>* orig,
                         matrix::Csr<ValueType, IndexType>* permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    if (count_num_blocks > 0) {
        kernel::inv_row_ptr_permute<<<count_num_blocks, default_block_size, 0,
                                      exec->get_stream()>>>(
            num_rows, row_perm, orig->get_const_row_ptrs(),
            permuted->get_row_ptrs());
    }
    components::prefix_sum_nonnegative(exec, permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (copy_num_blocks > 0) {
        kernel::inv_nonsymm_permute<config::warp_size>
            <<<copy_num_blocks, default_block_size, 0, exec->get_stream()>>>(
                num_rows, row_perm, col_perm, orig->get_const_row_ptrs(),
                orig->get_const_col_idxs(),
                as_device_type(orig->get_const_values()),
                permuted->get_row_ptrs(), permuted->get_col_idxs(),
                as_device_type(permuted->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const DefaultExecutor> exec,
                 const IndexType* perm,
                 const matrix::Csr<ValueType, IndexType>* orig,
                 matrix::Csr<ValueType, IndexType>* row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    if (count_num_blocks > 0) {
        kernel::row_ptr_permute<<<count_num_blocks, default_block_size, 0,
                                  exec->get_stream()>>>(
            num_rows, perm, orig->get_const_row_ptrs(),
            row_permuted->get_row_ptrs());
    }
    components::prefix_sum_nonnegative(exec, row_permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (copy_num_blocks > 0) {
        kernel::row_permute<config::warp_size>
            <<<copy_num_blocks, default_block_size, 0, exec->get_stream()>>>(
                num_rows, perm, orig->get_const_row_ptrs(),
                orig->get_const_col_idxs(),
                as_device_type(orig->get_const_values()),
                row_permuted->get_row_ptrs(), row_permuted->get_col_idxs(),
                as_device_type(row_permuted->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void inv_row_permute(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType* perm,
                     const matrix::Csr<ValueType, IndexType>* orig,
                     matrix::Csr<ValueType, IndexType>* row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    if (count_num_blocks > 0) {
        kernel::inv_row_ptr_permute<<<count_num_blocks, default_block_size, 0,
                                      exec->get_stream()>>>(
            num_rows, perm, orig->get_const_row_ptrs(),
            row_permuted->get_row_ptrs());
    }
    components::prefix_sum_nonnegative(exec, row_permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (copy_num_blocks > 0) {
        kernel::inv_row_permute<config::warp_size>
            <<<copy_num_blocks, default_block_size, 0, exec->get_stream()>>>(
                num_rows, perm, orig->get_const_row_ptrs(),
                orig->get_const_col_idxs(),
                as_device_type(orig->get_const_values()),
                row_permuted->get_row_ptrs(), row_permuted->get_col_idxs(),
                as_device_type(row_permuted->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void inv_symm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                            const ValueType* scale, const IndexType* perm,
                            const matrix::Csr<ValueType, IndexType>* orig,
                            matrix::Csr<ValueType, IndexType>* permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    if (count_num_blocks > 0) {
        kernel::inv_row_ptr_permute<<<count_num_blocks, default_block_size, 0,
                                      exec->get_stream()>>>(
            num_rows, perm, orig->get_const_row_ptrs(),
            permuted->get_row_ptrs());
    }
    components::prefix_sum_nonnegative(exec, permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (copy_num_blocks > 0) {
        kernel::inv_symm_scale_permute<config::warp_size>
            <<<copy_num_blocks, default_block_size, 0, exec->get_stream()>>>(
                num_rows, as_device_type(scale), perm,
                orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
                as_device_type(orig->get_const_values()),
                permuted->get_row_ptrs(), permuted->get_col_idxs(),
                as_device_type(permuted->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void inv_nonsymm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                               const ValueType* row_scale,
                               const IndexType* row_perm,
                               const ValueType* col_scale,
                               const IndexType* col_perm,
                               const matrix::Csr<ValueType, IndexType>* orig,
                               matrix::Csr<ValueType, IndexType>* permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    if (count_num_blocks > 0) {
        kernel::inv_row_ptr_permute<<<count_num_blocks, default_block_size, 0,
                                      exec->get_stream()>>>(
            num_rows, row_perm, orig->get_const_row_ptrs(),
            permuted->get_row_ptrs());
    }
    components::prefix_sum_nonnegative(exec, permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (copy_num_blocks > 0) {
        kernel::inv_nonsymm_scale_permute<config::warp_size>
            <<<copy_num_blocks, default_block_size, 0, exec->get_stream()>>>(
                num_rows, as_device_type(row_scale), row_perm,
                as_device_type(col_scale), col_perm, orig->get_const_row_ptrs(),
                orig->get_const_col_idxs(),
                as_device_type(orig->get_const_values()),
                permuted->get_row_ptrs(), permuted->get_col_idxs(),
                as_device_type(permuted->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void row_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                       const ValueType* scale, const IndexType* perm,
                       const matrix::Csr<ValueType, IndexType>* orig,
                       matrix::Csr<ValueType, IndexType>* row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    if (count_num_blocks > 0) {
        kernel::row_ptr_permute<<<count_num_blocks, default_block_size, 0,
                                  exec->get_stream()>>>(
            num_rows, perm, orig->get_const_row_ptrs(),
            row_permuted->get_row_ptrs());
    }
    components::prefix_sum_nonnegative(exec, row_permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (copy_num_blocks > 0) {
        kernel::row_scale_permute<config::warp_size>
            <<<copy_num_blocks, default_block_size, 0, exec->get_stream()>>>(
                num_rows, as_device_type(scale), perm,
                orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
                as_device_type(orig->get_const_values()),
                row_permuted->get_row_ptrs(), row_permuted->get_col_idxs(),
                as_device_type(row_permuted->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void inv_row_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                           const ValueType* scale, const IndexType* perm,
                           const matrix::Csr<ValueType, IndexType>* orig,
                           matrix::Csr<ValueType, IndexType>* row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    if (count_num_blocks > 0) {
        kernel::inv_row_ptr_permute<<<count_num_blocks, default_block_size, 0,
                                      exec->get_stream()>>>(
            num_rows, perm, orig->get_const_row_ptrs(),
            row_permuted->get_row_ptrs());
    }
    components::prefix_sum_nonnegative(exec, row_permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    if (copy_num_blocks > 0) {
        kernel::inv_row_scale_permute<config::warp_size>
            <<<copy_num_blocks, default_block_size, 0, exec->get_stream()>>>(
                num_rows, as_device_type(scale), perm,
                orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
                as_device_type(orig->get_const_values()),
                row_permuted->get_row_ptrs(), row_permuted->get_col_idxs(),
                as_device_type(row_permuted->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row_in_span(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source, const span& row_span,
    const span& col_span, array<IndexType>* row_nnz)
{
    const auto num_rows = source->get_size()[0];
    auto row_ptrs = source->get_const_row_ptrs();
    auto col_idxs = source->get_const_col_idxs();
    auto grid_dim = ceildiv(row_span.length(), default_block_size);
    if (grid_dim > 0) {
        kernel::calculate_nnz_per_row_in_span<<<grid_dim, default_block_size, 0,
                                                exec->get_stream()>>>(
            row_span, col_span, as_device_type(row_ptrs),
            as_device_type(col_idxs), as_device_type(row_nnz->get_data()));
    }
}


template <typename ValueType, typename IndexType>
void compute_submatrix(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Csr<ValueType, IndexType>* source,
                       gko::span row_span, gko::span col_span,
                       matrix::Csr<ValueType, IndexType>* result)
{
    auto row_offset = row_span.begin;
    auto col_offset = col_span.begin;
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto row_ptrs = source->get_const_row_ptrs();
    auto grid_dim = ceildiv(num_rows, default_block_size);
    if (grid_dim > 0) {
        kernel::compute_submatrix_idxs_and_vals<<<grid_dim, default_block_size,
                                                  0, exec->get_stream()>>>(
            num_rows, num_cols, row_offset, col_offset,
            as_device_type(source->get_const_row_ptrs()),
            as_device_type(source->get_const_col_idxs()),
            as_device_type(source->get_const_values()),
            as_device_type(result->get_const_row_ptrs()),
            as_device_type(result->get_col_idxs()),
            as_device_type(result->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row_in_index_set(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source,
    const gko::index_set<IndexType>& row_index_set,
    const gko::index_set<IndexType>& col_index_set,
    IndexType* row_nnz) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void compute_submatrix_from_index_set(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source,
    const gko::index_set<IndexType>& row_index_set,
    const gko::index_set<IndexType>& col_index_set,
    matrix::Csr<ValueType, IndexType>* result) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void fallback_transpose(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* input,
                        matrix::Csr<ValueType, IndexType>* output)
{
    const auto in_num_rows = input->get_size()[0];
    const auto out_num_rows = output->get_size()[0];
    const auto nnz = output->get_num_stored_elements();
    const auto in_row_ptrs = input->get_const_row_ptrs();
    const auto in_col_idxs = input->get_const_col_idxs();
    const auto in_vals = as_device_type(input->get_const_values());
    const auto out_row_ptrs = output->get_row_ptrs();
    const auto out_col_idxs = output->get_col_idxs();
    const auto out_vals = as_device_type(output->get_values());
    array<IndexType> out_row_idxs{exec, nnz};
    components::convert_ptrs_to_idxs(exec, in_row_ptrs, in_num_rows,
                                     out_col_idxs);
    exec->copy(nnz, in_vals, out_vals);
    exec->copy(nnz, in_col_idxs, out_row_idxs.get_data());
    auto loc_it = thrust::make_zip_iterator(
        thrust::make_tuple(out_row_idxs.get_data(), out_col_idxs));
    thrust::sort_by_key(thrust_policy(exec), loc_it, loc_it + nnz, out_vals);
    components::convert_idxs_to_ptrs(exec, out_row_idxs.get_data(), nnz,
                                     out_num_rows, out_row_ptrs);
}


template <typename ValueType, typename IndexType>
void fallback_sort(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::Csr<ValueType, IndexType>* to_sort)
{
    const auto row_ptrs = to_sort->get_const_row_ptrs();
    const auto col_idxs = to_sort->get_col_idxs();
    const auto vals = as_device_type(to_sort->get_values());
    const auto nnz = to_sort->get_num_stored_elements();
    const auto num_rows = to_sort->get_size()[0];
    array<IndexType> row_idx_array(exec, nnz);
    const auto row_idxs = row_idx_array.get_data();
    components::convert_ptrs_to_idxs(exec, row_ptrs, num_rows, row_idxs);
    const auto row_val_it =
        thrust::make_zip_iterator(thrust::make_tuple(row_idxs, vals));
    const auto col_val_it =
        thrust::make_zip_iterator(thrust::make_tuple(col_idxs, vals));
    // two sorts by integer keys hopefully enable Thrust to use cub's RadixSort
    thrust::sort_by_key(thrust_policy(exec), col_idxs, col_idxs + nnz,
                        row_val_it);
    thrust::stable_sort_by_key(thrust_policy(exec), row_idxs, row_idxs + nnz,
                               col_val_it);
}


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* to_check, bool* is_sorted)
{
    *is_sorted = true;
    auto cpu_array = make_array_view(exec->get_master(), 1, is_sorted);
    auto gpu_array = array<bool>{exec, cpu_array};
    auto block_size = default_block_size;
    auto num_rows = static_cast<IndexType>(to_check->get_size()[0]);
    auto num_blocks = ceildiv(num_rows, block_size);
    if (num_blocks > 0) {
        kernel::
            check_unsorted<<<num_blocks, block_size, 0, exec->get_stream()>>>(
                to_check->get_const_row_ptrs(), to_check->get_const_col_idxs(),
                num_rows, gpu_array.get_data());
    }
    cpu_array = gpu_array;
}


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    const auto nnz = orig->get_num_stored_elements();
    const auto diag_size = diag->get_size()[0];
    const auto num_blocks =
        ceildiv(config::warp_size * diag_size, default_block_size);

    const auto orig_values = orig->get_const_values();
    const auto orig_row_ptrs = orig->get_const_row_ptrs();
    const auto orig_col_idxs = orig->get_const_col_idxs();
    auto diag_values = diag->get_values();
    if (num_blocks > 0) {
        kernel::extract_diagonal<<<num_blocks, default_block_size, 0,
                                   exec->get_stream()>>>(
            diag_size, nnz, as_device_type(orig_values),
            as_device_type(orig_row_ptrs), as_device_type(orig_col_idxs),
            as_device_type(diag_values));
    }
}


template <typename ValueType, typename IndexType>
void check_diagonal_entries_exist(std::shared_ptr<const DefaultExecutor> exec,
                                  const matrix::Csr<ValueType, IndexType>* mtx,
                                  bool& has_all_diags)
{
    const auto num_diag = static_cast<IndexType>(
        std::min(mtx->get_size()[0], mtx->get_size()[1]));
    if (num_diag > 0) {
        const IndexType num_blocks =
            ceildiv(num_diag, default_block_size / config::warp_size);
        array<bool> has_diags(exec, {true});
        kernel::check_diagonal_entries<<<num_blocks, default_block_size, 0,
                                         exec->get_stream()>>>(
            num_diag, mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            has_diags.get_data());
        has_all_diags = get_element(has_diags, 0);
    } else {
        has_all_diags = true;
    }
}


template <typename ValueType, typename IndexType>
void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Dense<ValueType>* alpha,
                         const matrix::Dense<ValueType>* beta,
                         matrix::Csr<ValueType, IndexType>* mtx)
{
    const auto nrows = mtx->get_size()[0];
    if (nrows == 0) {
        return;
    }
    const auto nthreads = nrows * config::warp_size;
    const auto nblocks = ceildiv(nthreads, default_block_size);
    kernel::add_scaled_identity<<<nblocks, default_block_size, 0,
                                  exec->get_stream()>>>(
        as_device_type(alpha->get_const_values()),
        as_device_type(beta->get_const_values()), static_cast<IndexType>(nrows),
        mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
        as_device_type(mtx->get_values()));
}


namespace host_kernel {
namespace {


template <int items_per_thread, typename MatrixValueType,
          typename InputValueType, typename OutputValueType, typename IndexType>
void merge_path_spmv(syn::value_list<int, items_per_thread>,
                     std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Csr<MatrixValueType, IndexType>* a,
                     const matrix::Dense<InputValueType>* b,
                     matrix::Dense<OutputValueType>* c,
                     const matrix::Dense<MatrixValueType>* alpha = nullptr,
                     const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    const IndexType total = a->get_size()[0] + a->get_num_stored_elements();
    const IndexType grid_num =
        ceildiv(total, spmv_block_size * items_per_thread);
    const auto grid = grid_num;
    const auto block = spmv_block_size;
    // TODO: workspace?
    array<IndexType> row_out(exec, grid_num);
    // TODO: should we store the value in arithmetic_type or output_type?
    array<arithmetic_type> val_out(exec, grid_num);

    const auto a_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(a);

    for (IndexType column_id = 0; column_id < b->get_size()[1]; column_id++) {
        const auto column_span =
            acc::index_span(static_cast<acc::size_type>(column_id),
                            static_cast<acc::size_type>(column_id + 1));
        const auto b_vals =
            acc::helper::build_const_rrm_accessor<arithmetic_type>(b,
                                                                   column_span);
        auto c_vals =
            acc::helper::build_rrm_accessor<arithmetic_type>(c, column_span);
        if (alpha == nullptr && beta == nullptr) {
            if (grid_num > 0) {
                kernel::abstract_merge_path_spmv<items_per_thread>
                    <<<grid, block, 0, exec->get_stream()>>>(
                        static_cast<IndexType>(a->get_size()[0]),
                        acc::as_device_range(a_vals), a->get_const_col_idxs(),
                        as_device_type(a->get_const_row_ptrs()),
                        as_device_type(a->get_const_srow()),
                        acc::as_device_range(b_vals),
                        acc::as_device_range(c_vals),
                        as_device_type(row_out.get_data()),
                        as_device_type(val_out.get_data()));
            }
            kernel::
                abstract_reduce<<<1, spmv_block_size, 0, exec->get_stream()>>>(
                    grid_num, as_device_type(val_out.get_data()),
                    as_device_type(row_out.get_data()),
                    acc::as_device_range(c_vals));

        } else if (alpha != nullptr && beta != nullptr) {
            if (grid_num > 0) {
                kernel::abstract_merge_path_spmv<items_per_thread>
                    <<<grid, block, 0, exec->get_stream()>>>(
                        static_cast<IndexType>(a->get_size()[0]),
                        as_device_type(alpha->get_const_values()),
                        acc::as_device_range(a_vals), a->get_const_col_idxs(),
                        as_device_type(a->get_const_row_ptrs()),
                        as_device_type(a->get_const_srow()),
                        acc::as_device_range(b_vals),
                        as_device_type(beta->get_const_values()),
                        acc::as_device_range(c_vals),
                        as_device_type(row_out.get_data()),
                        as_device_type(val_out.get_data()));
            }
            kernel::
                abstract_reduce<<<1, spmv_block_size, 0, exec->get_stream()>>>(
                    grid_num, as_device_type(val_out.get_data()),
                    as_device_type(row_out.get_data()),
                    as_device_type(alpha->get_const_values()),
                    acc::as_device_range(c_vals));
        } else {
            GKO_KERNEL_NOT_FOUND;
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_merge_path_spmv, merge_path_spmv);


template <typename ValueType, typename IndexType>
int compute_items_per_thread(std::shared_ptr<const DefaultExecutor> exec)
{
#if defined(GKO_COMPILING_CUDA) || GINKGO_HIP_PLATFORM_NVCC


    const int version =
        (exec->get_major_version() << 4) + exec->get_minor_version();
    // The num_item is decided to make the occupancy 100%
    // TODO: Extend this list when new GPU is released
    //       Tune this parameter
    // 128 threads/block the number of items per threads
    // 3.0 3.5: 6
    // 3.7: 14
    // 5.0, 5.3, 6.0, 6.2: 8
    // 5.2, 6.1, 7.0: 12
    int num_item = 6;
    switch (version) {
    case 0x50:
    case 0x53:
    case 0x60:
    case 0x62:
        num_item = 8;
        break;
    case 0x52:
    case 0x61:
    case 0x70:
        num_item = 12;
        break;
    case 0x37:
        num_item = 14;
    }


#else


    // HIP uses the minimal num_item to make the code work correctly.
    // TODO: this parameter should be tuned.
    int num_item = 6;


#endif  // GINKGO_HIP_PLATFORM_NVCC


    // Ensure that the following is satisfied:
    // sizeof(IndexType) + sizeof(ValueType)
    // <= items_per_thread * sizeof(IndexType)
    constexpr int minimal_num =
        ceildiv(sizeof(IndexType) + sizeof(ValueType), sizeof(IndexType));
    int items_per_thread = num_item * 4 / sizeof(IndexType);
    return std::max(minimal_num, items_per_thread);
}


template <int subwarp_size, typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void classical_spmv(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<MatrixValueType, IndexType>* a,
                    const matrix::Dense<InputValueType>* b,
                    matrix::Dense<OutputValueType>* c,
                    const matrix::Dense<MatrixValueType>* alpha = nullptr,
                    const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;

    const auto nwarps = exec->get_num_warps_per_sm() *
                        exec->get_num_multiprocessor() *
                        classical_oversubscription;
    const auto gridx =
        std::min(ceildiv(a->get_size()[0], spmv_block_size / subwarp_size),
                 int64(nwarps / warps_in_block));
    const dim3 grid(gridx, b->get_size()[1]);
    const auto block = spmv_block_size;

    const auto a_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(a);
    const auto b_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(b);
    auto c_vals = acc::helper::build_rrm_accessor<arithmetic_type>(c);
    if (alpha == nullptr && beta == nullptr) {
        if (grid.x > 0 && grid.y > 0) {
            kernel::abstract_classical_spmv<subwarp_size>
                <<<grid, block, 0, exec->get_stream()>>>(
                    a->get_size()[0], acc::as_device_range(a_vals),
                    a->get_const_col_idxs(),
                    as_device_type(a->get_const_row_ptrs()),
                    acc::as_device_range(b_vals), acc::as_device_range(c_vals));
        }
    } else if (alpha != nullptr && beta != nullptr) {
        if (grid.x > 0 && grid.y > 0) {
            kernel::abstract_classical_spmv<subwarp_size>
                <<<grid, block, 0, exec->get_stream()>>>(
                    a->get_size()[0], as_device_type(alpha->get_const_values()),
                    acc::as_device_range(a_vals), a->get_const_col_idxs(),
                    as_device_type(a->get_const_row_ptrs()),
                    acc::as_device_range(b_vals),
                    as_device_type(beta->get_const_values()),
                    acc::as_device_range(c_vals));
        }
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_classical_spmv, classical_spmv);


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
bool load_balance_spmv(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Csr<MatrixValueType, IndexType>* a,
                       const matrix::Dense<InputValueType>* b,
                       matrix::Dense<OutputValueType>* c,
                       const matrix::Dense<MatrixValueType>* alpha = nullptr,
                       const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;

    // not support 16 bit atomic
#if !(defined(CUDA_VERSION) && (__CUDA_ARCH__ >= 700))
    if constexpr (sizeof(remove_complex<OutputValueType>) == sizeof(int16)) {
        return false;
    } else
#endif
    {
        if (beta) {
            dense::scale(exec, beta, c);
        } else {
            dense::fill(exec, c, zero<OutputValueType>());
        }
        const IndexType nwarps = a->get_num_srow_elements();
        if (nwarps > 0) {
            const dim3 csr_block(config::warp_size, warps_in_block, 1);
            const dim3 csr_grid(ceildiv(nwarps, warps_in_block),
                                b->get_size()[1]);
            const auto a_vals =
                acc::helper::build_const_rrm_accessor<arithmetic_type>(a);
            const auto b_vals =
                acc::helper::build_const_rrm_accessor<arithmetic_type>(b);
            auto c_vals = acc::helper::build_rrm_accessor<arithmetic_type>(c);
            if (alpha) {
                if (csr_grid.x > 0 && csr_grid.y > 0) {
                    kernel::abstract_spmv<<<csr_grid, csr_block, 0,
                                            exec->get_stream()>>>(
                        nwarps, static_cast<IndexType>(a->get_size()[0]),
                        as_device_type(alpha->get_const_values()),
                        acc::as_device_range(a_vals), a->get_const_col_idxs(),
                        as_device_type(a->get_const_row_ptrs()),
                        as_device_type(a->get_const_srow()),
                        acc::as_device_range(b_vals),
                        acc::as_device_range(c_vals));
                }
            } else {
                if (csr_grid.x > 0 && csr_grid.y > 0) {
                    kernel::abstract_spmv<<<csr_grid, csr_block, 0,
                                            exec->get_stream()>>>(
                        nwarps, static_cast<IndexType>(a->get_size()[0]),
                        acc::as_device_range(a_vals), a->get_const_col_idxs(),
                        as_device_type(a->get_const_row_ptrs()),
                        as_device_type(a->get_const_srow()),
                        acc::as_device_range(b_vals),
                        acc::as_device_range(c_vals));
                }
            }
        }
        return true;
    }
}


template <typename ValueType, typename IndexType>
bool try_general_sparselib_spmv(std::shared_ptr<const DefaultExecutor> exec,
                                const ValueType* alpha,
                                const matrix::Csr<ValueType, IndexType>* a,
                                const matrix::Dense<ValueType>* b,
                                const ValueType* beta,
                                matrix::Dense<ValueType>* c)
{
#ifdef GKO_COMPILING_HIP
    bool try_sparselib = sparselib::is_supported<ValueType, IndexType>::value;
    try_sparselib =
        try_sparselib && b->get_stride() == 1 && c->get_stride() == 1;
    // rocSPARSE has issues with zero matrices
    try_sparselib = try_sparselib && a->get_num_stored_elements() > 0;
    if (try_sparselib) {
        auto descr = sparselib::create_mat_descr();

        auto row_ptrs = a->get_const_row_ptrs();
        auto col_idxs = a->get_const_col_idxs();

        sparselib::spmv(exec->get_sparselib_handle(),
                        SPARSELIB_OPERATION_NON_TRANSPOSE, a->get_size()[0],
                        a->get_size()[1], a->get_num_stored_elements(), alpha,
                        descr, a->get_const_values(), row_ptrs, col_idxs,
                        b->get_const_values(), beta, c->get_values());

        sparselib::destroy(descr);
    }
    return try_sparselib;
#else  // GKO_COMPILING_CUDA
    auto handle = exec->get_sparselib_handle();
    // workaround for a division by zero in cuSPARSE 11.?
    if (a->get_size()[1] == 0) {
        return false;
    }
    cusparseOperation_t trans = SPARSELIB_OPERATION_NON_TRANSPOSE;
    auto row_ptrs = const_cast<IndexType*>(a->get_const_row_ptrs());
    auto col_idxs = const_cast<IndexType*>(a->get_const_col_idxs());
    auto values = const_cast<ValueType*>(a->get_const_values());
    auto mat = sparselib::create_csr(a->get_size()[0], a->get_size()[1],
                                     a->get_num_stored_elements(), row_ptrs,
                                     col_idxs, values);
    auto b_val = const_cast<ValueType*>(b->get_const_values());
    auto c_val = c->get_values();
    if (b->get_stride() == 1 && c->get_stride() == 1) {
        auto vecb = sparselib::create_dnvec(b->get_size()[0], b_val);
        auto vecc = sparselib::create_dnvec(c->get_size()[0], c_val);
#if CUDA_VERSION >= 11021
        constexpr auto alg = CUSPARSE_SPMV_CSR_ALG1;
#else
        constexpr auto alg = CUSPARSE_CSRMV_ALG1;
#endif
        size_type buffer_size = 0;
        sparselib::spmv_buffersize<ValueType>(handle, trans, alpha, mat, vecb,
                                              beta, vecc, alg, &buffer_size);

        array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();
        sparselib::spmv<ValueType>(handle, trans, alpha, mat, vecb, beta, vecc,
                                   alg, buffer);
        sparselib::destroy(vecb);
        sparselib::destroy(vecc);
    } else {
#if CUDA_VERSION >= 11060
        if (b->get_size()[1] == 1) {
            // cusparseSpMM seems to take the single strided vector as column
            // major without considering stride and row major (cuda 11.6)
            return false;
        }
#endif  // CUDA_VERSION >= 11060
        cusparseSpMMAlg_t alg = CUSPARSE_SPMM_CSR_ALG2;
        auto vecb =
            sparselib::create_dnmat(b->get_size(), b->get_stride(), b_val);
        auto vecc =
            sparselib::create_dnmat(c->get_size(), c->get_stride(), c_val);
        size_type buffer_size = 0;
        sparselib::spmm_buffersize<ValueType>(handle, trans, trans, alpha, mat,
                                              vecb, beta, vecc, alg,
                                              &buffer_size);

        array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();
        sparselib::spmm<ValueType>(handle, trans, trans, alpha, mat, vecb, beta,
                                   vecc, alg, buffer);
        sparselib::destroy(vecb);
        sparselib::destroy(vecc);
    }
    sparselib::destroy(mat);
    return true;
#endif  // GKO_COMPILING_CUDA
}


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType,
          typename = std::enable_if_t<
              !std::is_same<MatrixValueType, InputValueType>::value ||
              !std::is_same<MatrixValueType, OutputValueType>::value>>
bool try_sparselib_spmv(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<MatrixValueType, IndexType>* a,
                        const matrix::Dense<InputValueType>* b,
                        matrix::Dense<OutputValueType>* c,
                        const matrix::Dense<MatrixValueType>* alpha = nullptr,
                        const matrix::Dense<OutputValueType>* beta = nullptr)
{
    // TODO: support sparselib mixed
    return false;
}

template <typename ValueType, typename IndexType>
bool try_sparselib_spmv(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* a,
                        const matrix::Dense<ValueType>* b,
                        matrix::Dense<ValueType>* c,
                        const matrix::Dense<ValueType>* alpha = nullptr,
                        const matrix::Dense<ValueType>* beta = nullptr)
{
    if (alpha) {
        return try_general_sparselib_spmv(exec, alpha->get_const_values(), a, b,
                                          beta->get_const_values(), c);
    } else {
        auto handle = exec->get_sparselib_handle();
        sparselib::pointer_mode_guard pm_guard(handle);
        const auto valpha = one<ValueType>();
        const auto vbeta = zero<ValueType>();
        return try_general_sparselib_spmv(exec, &valpha, a, b, &vbeta, c);
    }
}


}  // anonymous namespace
}  // namespace host_kernel


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Csr<MatrixValueType, IndexType>* a,
          const matrix::Dense<InputValueType>* b,
          matrix::Dense<OutputValueType>* c)
{
    if (c->get_size()[0] == 0 || c->get_size()[1] == 0) {
        // empty output: nothing to do
    } else if (a->get_strategy()->get_name() == "merge_path") {
        using arithmetic_type =
            highest_precision<InputValueType, OutputValueType, MatrixValueType>;
        int items_per_thread =
            host_kernel::compute_items_per_thread<arithmetic_type, IndexType>(
                exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
    } else {
        bool use_classical = true;
        if (a->get_strategy()->get_name() == "load_balance") {
            use_classical = !host_kernel::load_balance_spmv(exec, a, b, c);
        } else if (a->get_strategy()->get_name() == "sparselib" ||
                   a->get_strategy()->get_name() == "cusparse") {
            use_classical = !host_kernel::try_sparselib_spmv(exec, a, b, c);
        }
        if (use_classical) {
            IndexType max_length_per_row = 0;
            using Tcsr = matrix::Csr<MatrixValueType, IndexType>;
            if (auto strategy =
                    std::dynamic_pointer_cast<const typename Tcsr::classical>(
                        a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else if (auto strategy = std::dynamic_pointer_cast<
                           const typename Tcsr::automatical>(
                           a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else {
                // as a fall-back: use average row length, at least 1
                max_length_per_row = a->get_num_stored_elements() /
                                     std::max<size_type>(a->get_size()[0], 1);
            }
            max_length_per_row = std::max<size_type>(max_length_per_row, 1);
            host_kernel::select_classical_spmv(
                classical_kernels(),
                [&max_length_per_row](int compiled_info) {
                    return max_length_per_row >= compiled_info;
                },
                syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
        }
    }
}


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::Csr<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    if (c->get_size()[0] == 0 || c->get_size()[1] == 0) {
        // empty output: nothing to do
    } else if (a->get_strategy()->get_name() == "merge_path") {
        using arithmetic_type =
            highest_precision<InputValueType, OutputValueType, MatrixValueType>;
        int items_per_thread =
            host_kernel::compute_items_per_thread<arithmetic_type, IndexType>(
                exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha,
            beta);
    } else {
        bool use_classical = true;
        if (a->get_strategy()->get_name() == "load_balance") {
            use_classical =
                !host_kernel::load_balance_spmv(exec, a, b, c, alpha, beta);
        } else if (a->get_strategy()->get_name() == "sparselib" ||
                   a->get_strategy()->get_name() == "cusparse") {
            use_classical =
                !host_kernel::try_sparselib_spmv(exec, a, b, c, alpha, beta);
        }
        if (use_classical) {
            IndexType max_length_per_row = 0;
            using Tcsr = matrix::Csr<MatrixValueType, IndexType>;
            if (auto strategy =
                    std::dynamic_pointer_cast<const typename Tcsr::classical>(
                        a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else if (auto strategy = std::dynamic_pointer_cast<
                           const typename Tcsr::automatical>(
                           a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else {
                // as a fall-back: use average row length, at least 1
                max_length_per_row = a->get_num_stored_elements() /
                                     std::max<size_type>(a->get_size()[0], 1);
            }
            max_length_per_row = std::max<size_type>(max_length_per_row, 1);
            host_kernel::select_classical_spmv(
                classical_kernels(),
                [&max_length_per_row](int compiled_info) {
                    return max_length_per_row >= compiled_info;
                },
                syn::value_list<int>(), syn::type_list<>(), exec, a, b, c,
                alpha, beta);
        }
    }
}


template <typename ValueType, typename IndexType>
void spgemm(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Csr<ValueType, IndexType>* a,
            const matrix::Csr<ValueType, IndexType>* b,
            matrix::Csr<ValueType, IndexType>* c)
{
#ifdef GKO_COMPILING_HIP
    if (sparselib::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_sparselib_handle();
        sparselib::pointer_mode_guard pm_guard(handle);
        auto a_descr = sparselib::create_mat_descr();
        auto b_descr = sparselib::create_mat_descr();
        auto c_descr = sparselib::create_mat_descr();
        auto d_descr = sparselib::create_mat_descr();
        auto info = sparselib::create_spgemm_info();

        auto alpha = one<ValueType>();
        auto a_nnz = static_cast<IndexType>(a->get_num_stored_elements());
        auto a_vals = a->get_const_values();
        auto a_row_ptrs = a->get_const_row_ptrs();
        auto a_col_idxs = a->get_const_col_idxs();
        auto b_nnz = static_cast<IndexType>(b->get_num_stored_elements());
        auto b_vals = b->get_const_values();
        auto b_row_ptrs = b->get_const_row_ptrs();
        auto b_col_idxs = b->get_const_col_idxs();
        auto null_value = static_cast<ValueType*>(nullptr);
        auto null_index = static_cast<IndexType*>(nullptr);
        auto zero_nnz = IndexType{};
        auto m = static_cast<IndexType>(a->get_size()[0]);
        auto n = static_cast<IndexType>(b->get_size()[1]);
        auto k = static_cast<IndexType>(a->get_size()[1]);
        auto c_row_ptrs = c->get_row_ptrs();
        matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
        auto& c_col_idxs_array = c_builder.get_col_idx_array();
        auto& c_vals_array = c_builder.get_value_array();

        // allocate buffer
        size_type buffer_size{};
        sparselib::spgemm_buffer_size(
            handle, m, n, k, &alpha, a_descr, a_nnz, a_row_ptrs, a_col_idxs,
            b_descr, b_nnz, b_row_ptrs, b_col_idxs, null_value, d_descr,
            zero_nnz, null_index, null_index, info, buffer_size);
        array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();

        // count nnz
        IndexType c_nnz{};
        sparselib::spgemm_nnz(
            handle, m, n, k, a_descr, a_nnz, a_row_ptrs, a_col_idxs, b_descr,
            b_nnz, b_row_ptrs, b_col_idxs, d_descr, zero_nnz, null_index,
            null_index, c_descr, c_row_ptrs, &c_nnz, info, buffer);

        // accumulate non-zeros
        c_col_idxs_array.resize_and_reset(c_nnz);
        c_vals_array.resize_and_reset(c_nnz);
        auto c_col_idxs = c_col_idxs_array.get_data();
        auto c_vals = c_vals_array.get_data();
        sparselib::spgemm(handle, m, n, k, &alpha, a_descr, a_nnz, a_vals,
                          a_row_ptrs, a_col_idxs, b_descr, b_nnz, b_vals,
                          b_row_ptrs, b_col_idxs, null_value, d_descr, zero_nnz,
                          null_value, null_index, null_index, c_descr, c_vals,
                          c_row_ptrs, c_col_idxs, info, buffer);

        sparselib::destroy_spgemm_info(info);
        sparselib::destroy(d_descr);
        sparselib::destroy(c_descr);
        sparselib::destroy(b_descr);
        sparselib::destroy(a_descr);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
#else   // GKO_COMPILING_CUDA
    auto a_vals = a->get_const_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto b_vals = b->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    auto c_row_ptrs = c->get_row_ptrs();

    auto handle = exec->get_sparselib_handle();
    sparselib::pointer_mode_guard pm_guard(handle);

    auto alpha = one<ValueType>();
    auto a_nnz = static_cast<IndexType>(a->get_num_stored_elements());
    auto b_nnz = static_cast<IndexType>(b->get_num_stored_elements());
    auto null_value = static_cast<ValueType*>(nullptr);
    auto null_index = static_cast<IndexType*>(nullptr);
    auto zero_nnz = IndexType{};
    auto m = IndexType(a->get_size()[0]);
    auto n = IndexType(b->get_size()[1]);
    auto k = IndexType(a->get_size()[1]);
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();

    const auto beta = zero<ValueType>();
    auto spgemm_descr = sparselib::create_spgemm_descr();
    auto a_descr = sparselib::create_csr(
        m, k, a_nnz, const_cast<IndexType*>(a_row_ptrs),
        const_cast<IndexType*>(a_col_idxs), const_cast<ValueType*>(a_vals));
    auto b_descr = sparselib::create_csr(
        k, n, b_nnz, const_cast<IndexType*>(b_row_ptrs),
        const_cast<IndexType*>(b_col_idxs), const_cast<ValueType*>(b_vals));
    auto c_descr = sparselib::create_csr(m, n, zero_nnz, null_index, null_index,
                                         null_value);

    // estimate work
    size_type buffer1_size{};
    sparselib::spgemm_work_estimation(handle, &alpha, a_descr, b_descr, &beta,
                                      c_descr, spgemm_descr, buffer1_size,
                                      nullptr);
    array<char> buffer1{exec, buffer1_size};
    sparselib::spgemm_work_estimation(handle, &alpha, a_descr, b_descr, &beta,
                                      c_descr, spgemm_descr, buffer1_size,
                                      buffer1.get_data());

    // compute spgemm
    size_type buffer2_size{};
    sparselib::spgemm_compute(handle, &alpha, a_descr, b_descr, &beta, c_descr,
                              spgemm_descr, buffer1.get_data(), buffer2_size,
                              nullptr);
    array<char> buffer2{exec, buffer2_size};
    sparselib::spgemm_compute(handle, &alpha, a_descr, b_descr, &beta, c_descr,
                              spgemm_descr, buffer1.get_data(), buffer2_size,
                              buffer2.get_data());

    // copy data to result
    auto c_nnz = sparselib::sparse_matrix_nnz(c_descr);
    c_col_idxs_array.resize_and_reset(c_nnz);
    c_vals_array.resize_and_reset(c_nnz);
    sparselib::csr_set_pointers(c_descr, c_row_ptrs,
                                c_col_idxs_array.get_data(),
                                c_vals_array.get_data());

    sparselib::spgemm_copy(handle, &alpha, a_descr, b_descr, &beta, c_descr,
                           spgemm_descr);

    sparselib::destroy(c_descr);
    sparselib::destroy(b_descr);
    sparselib::destroy(a_descr);
    sparselib::destroy(spgemm_descr);
#endif  // GKO_COMPILING_CUDA
}


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Dense<ValueType>* alpha,
                     const matrix::Csr<ValueType, IndexType>* a,
                     const matrix::Csr<ValueType, IndexType>* b,
                     const matrix::Dense<ValueType>* beta,
                     const matrix::Csr<ValueType, IndexType>* d,
                     matrix::Csr<ValueType, IndexType>* c)
{
#ifdef GKO_COMPILING_HIP
    if (sparselib::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_sparselib_handle();
        sparselib::pointer_mode_guard pm_guard(handle);
        auto a_descr = sparselib::create_mat_descr();
        auto b_descr = sparselib::create_mat_descr();
        auto c_descr = sparselib::create_mat_descr();
        auto d_descr = sparselib::create_mat_descr();
        auto info = sparselib::create_spgemm_info();

        auto a_nnz = static_cast<IndexType>(a->get_num_stored_elements());
        auto a_vals = a->get_const_values();
        auto a_row_ptrs = a->get_const_row_ptrs();
        auto a_col_idxs = a->get_const_col_idxs();
        auto b_nnz = static_cast<IndexType>(b->get_num_stored_elements());
        auto b_vals = b->get_const_values();
        auto b_row_ptrs = b->get_const_row_ptrs();
        auto b_col_idxs = b->get_const_col_idxs();
        auto d_vals = d->get_const_values();
        auto d_row_ptrs = d->get_const_row_ptrs();
        auto d_col_idxs = d->get_const_col_idxs();
        auto null_value = static_cast<ValueType*>(nullptr);
        auto null_index = static_cast<IndexType*>(nullptr);
        auto one_value = one<ValueType>();
        auto m = static_cast<IndexType>(a->get_size()[0]);
        auto n = static_cast<IndexType>(b->get_size()[1]);
        auto k = static_cast<IndexType>(a->get_size()[1]);

        // allocate buffer
        size_type buffer_size{};
        sparselib::spgemm_buffer_size(
            handle, m, n, k, &one_value, a_descr, a_nnz, a_row_ptrs, a_col_idxs,
            b_descr, b_nnz, b_row_ptrs, b_col_idxs, null_value, d_descr,
            IndexType{}, null_index, null_index, info, buffer_size);
        array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();

        // count nnz
        array<IndexType> c_tmp_row_ptrs_array(exec, m + 1);
        auto c_tmp_row_ptrs = c_tmp_row_ptrs_array.get_data();
        IndexType c_nnz{};
        sparselib::spgemm_nnz(
            handle, m, n, k, a_descr, a_nnz, a_row_ptrs, a_col_idxs, b_descr,
            b_nnz, b_row_ptrs, b_col_idxs, d_descr, IndexType{}, null_index,
            null_index, c_descr, c_tmp_row_ptrs, &c_nnz, info, buffer);

        // accumulate non-zeros for A * B
        array<IndexType> c_tmp_col_idxs_array(exec, c_nnz);
        array<ValueType> c_tmp_vals_array(exec, c_nnz);
        auto c_tmp_col_idxs = c_tmp_col_idxs_array.get_data();
        auto c_tmp_vals = c_tmp_vals_array.get_data();
        sparselib::spgemm(handle, m, n, k, &one_value, a_descr, a_nnz, a_vals,
                          a_row_ptrs, a_col_idxs, b_descr, b_nnz, b_vals,
                          b_row_ptrs, b_col_idxs, null_value, d_descr,
                          IndexType{}, null_value, null_index, null_index,
                          c_descr, c_tmp_vals, c_tmp_row_ptrs, c_tmp_col_idxs,
                          info, buffer);

        // destroy hipsparse context
        sparselib::destroy_spgemm_info(info);
        sparselib::destroy(d_descr);
        sparselib::destroy(c_descr);
        sparselib::destroy(b_descr);
        sparselib::destroy(a_descr);

        auto total_nnz = c_nnz + d->get_num_stored_elements();
        auto nnz_per_row = total_nnz / m;
        select_spgeam(
            spgeam_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size >= nnz_per_row ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), exec,
            alpha->get_const_values(), c_tmp_row_ptrs, c_tmp_col_idxs,
            c_tmp_vals, beta->get_const_values(), d_row_ptrs, d_col_idxs,
            d_vals, c);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
#else   // GKO_COMPILING_CUDA
    auto handle = exec->get_sparselib_handle();
    sparselib::pointer_mode_guard pm_guard(handle);

    auto valpha = exec->copy_val_to_host(alpha->get_const_values());
    auto a_nnz = IndexType(a->get_num_stored_elements());
    auto a_vals = a->get_const_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto b_nnz = IndexType(b->get_num_stored_elements());
    auto b_vals = b->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    auto vbeta = exec->copy_val_to_host(beta->get_const_values());
    auto d_nnz = IndexType(d->get_num_stored_elements());
    auto d_vals = d->get_const_values();
    auto d_row_ptrs = d->get_const_row_ptrs();
    auto d_col_idxs = d->get_const_col_idxs();
    auto m = IndexType(a->get_size()[0]);
    auto n = IndexType(b->get_size()[1]);
    auto k = IndexType(a->get_size()[1]);
    auto c_row_ptrs = c->get_row_ptrs();

    auto null_value = static_cast<ValueType*>(nullptr);
    auto null_index = static_cast<IndexType*>(nullptr);
    auto one_val = one<ValueType>();
    auto zero_val = zero<ValueType>();
    auto zero_nnz = IndexType{};
    auto spgemm_descr = sparselib::create_spgemm_descr();
    auto a_descr = sparselib::create_csr(
        m, k, a_nnz, const_cast<IndexType*>(a_row_ptrs),
        const_cast<IndexType*>(a_col_idxs), const_cast<ValueType*>(a_vals));
    auto b_descr = sparselib::create_csr(
        k, n, b_nnz, const_cast<IndexType*>(b_row_ptrs),
        const_cast<IndexType*>(b_col_idxs), const_cast<ValueType*>(b_vals));
    auto c_descr = sparselib::create_csr(m, n, zero_nnz, null_index, null_index,
                                         null_value);

    // estimate work
    size_type buffer1_size{};
    sparselib::spgemm_work_estimation(handle, &one_val, a_descr, b_descr,
                                      &zero_val, c_descr, spgemm_descr,
                                      buffer1_size, nullptr);
    array<char> buffer1{exec, buffer1_size};
    sparselib::spgemm_work_estimation(handle, &one_val, a_descr, b_descr,
                                      &zero_val, c_descr, spgemm_descr,
                                      buffer1_size, buffer1.get_data());

    // compute spgemm
    size_type buffer2_size{};
    sparselib::spgemm_compute(handle, &one_val, a_descr, b_descr, &zero_val,
                              c_descr, spgemm_descr, buffer1.get_data(),
                              buffer2_size, nullptr);
    array<char> buffer2{exec, buffer2_size};
    sparselib::spgemm_compute(handle, &one_val, a_descr, b_descr, &zero_val,
                              c_descr, spgemm_descr, buffer1.get_data(),
                              buffer2_size, buffer2.get_data());

    // write result to temporary storage
    auto c_tmp_nnz = sparselib::sparse_matrix_nnz(c_descr);
    array<IndexType> c_tmp_row_ptrs_array(exec, m + 1);
    array<IndexType> c_tmp_col_idxs_array(exec, c_tmp_nnz);
    array<ValueType> c_tmp_vals_array(exec, c_tmp_nnz);
    sparselib::csr_set_pointers(c_descr, c_tmp_row_ptrs_array.get_data(),
                                c_tmp_col_idxs_array.get_data(),
                                c_tmp_vals_array.get_data());

    sparselib::spgemm_copy(handle, &one_val, a_descr, b_descr, &zero_val,
                           c_descr, spgemm_descr);

    sparselib::destroy(c_descr);
    sparselib::destroy(b_descr);
    sparselib::destroy(a_descr);
    sparselib::destroy(spgemm_descr);

    auto spgeam_total_nnz = c_tmp_nnz + d->get_num_stored_elements();
    auto nnz_per_row = spgeam_total_nnz / m;
    select_spgeam(
        spgeam_kernels(),
        [&](int compiled_subwarp_size) {
            return compiled_subwarp_size >= nnz_per_row ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec,
        alpha->get_const_values(), c_tmp_row_ptrs_array.get_const_data(),
        c_tmp_col_idxs_array.get_const_data(),
        c_tmp_vals_array.get_const_data(), beta->get_const_values(), d_row_ptrs,
        d_col_idxs, d_vals, c);
#endif  // GKO_COMPILING_CUDA
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* orig,
               matrix::Csr<ValueType, IndexType>* trans)
{
    if (orig->get_size()[0] == 0) {
        return;
    }
    if (sparselib::is_supported<ValueType, IndexType>::value) {
#ifdef GKO_COMPILING_HIP
        hipsparseAction_t copyValues = HIPSPARSE_ACTION_NUMERIC;
        hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;

        sparselib::transpose(
            exec->get_sparselib_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues, idxBase);
#else   // GKO_COMPILING_CUDA
        cudaDataType_t cu_value =
            gko::kernels::cuda::cuda_data_type<ValueType>();
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;
        size_type buffer_size = 0;
        sparselib::transpose_buffersize(
            exec->get_sparselib_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value, copyValues,
            idxBase, alg, &buffer_size);
        array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();
        sparselib::transpose(
            exec->get_sparselib_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value, copyValues,
            idxBase, alg, buffer);
#endif  // GKO_COMPILING_CUDA
    } else {
        fallback_transpose(exec, orig, trans);
    }
}


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* orig,
                    matrix::Csr<ValueType, IndexType>* trans)
{
    if (orig->get_size()[0] == 0) {
        return;
    }
    const auto block_size = default_block_size;
    const auto grid_size =
        ceildiv(trans->get_num_stored_elements(), block_size);
    transpose(exec, orig, trans);
    if (grid_size > 0 && is_complex<ValueType>()) {
        kernel::conjugate<<<grid_size, block_size, 0, exec->get_stream()>>>(
            trans->get_num_stored_elements(),
            as_device_type(trans->get_values()));
    }
}


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const DefaultExecutor> exec,
                          matrix::Csr<ValueType, IndexType>* to_sort)
{
    if (sparselib::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_sparselib_handle();
        auto descr = sparselib::create_mat_descr();
        auto m = IndexType(to_sort->get_size()[0]);
        auto n = IndexType(to_sort->get_size()[1]);
        auto nnz = IndexType(to_sort->get_num_stored_elements());
        auto row_ptrs = to_sort->get_const_row_ptrs();
        auto col_idxs = to_sort->get_col_idxs();
        auto vals = to_sort->get_values();

        // copy values
        array<ValueType> tmp_vals_array(exec, nnz);
        exec->copy(nnz, vals, tmp_vals_array.get_data());
        auto tmp_vals = tmp_vals_array.get_const_data();

        // init identity permutation
        array<IndexType> permutation_array(exec, nnz);
        auto permutation = permutation_array.get_data();
        components::fill_seq_array(exec, permutation, nnz);

        // allocate buffer
        size_type buffer_size{};
        sparselib::csrsort_buffer_size(handle, m, n, nnz, row_ptrs, col_idxs,
                                       buffer_size);
        array<char> buffer_array{exec, buffer_size};
        auto buffer = buffer_array.get_data();

        // sort column indices
        sparselib::csrsort(handle, m, n, nnz, descr, row_ptrs, col_idxs,
                           permutation, buffer);

        // sort values
#ifdef GKO_COMPILING_HIP
        sparselib::gather(handle, nnz, tmp_vals, vals, permutation);
#else  // GKO_COMPILING_CUDA
        auto val_vec = sparselib::create_spvec(nnz, nnz, permutation, vals);
        auto tmp_vec =
            sparselib::create_dnvec(nnz, const_cast<ValueType*>(tmp_vals));
        sparselib::gather(handle, tmp_vec, val_vec);
#endif

        sparselib::destroy(descr);
    } else {
        fallback_sort(exec, to_sort);
    }
}


}  // namespace csr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
