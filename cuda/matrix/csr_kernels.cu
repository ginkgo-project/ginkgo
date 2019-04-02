/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/components/zero_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace csr {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * cuda_config::warp_size;
constexpr int classical_block_size = 64;
constexpr int wsize = cuda_config::warp_size;


/**
 * A compile-time list of the number items per threads for which spmv kernel
 * should be compiled.
 */
using compiled_kernels = syn::value_list<int, 3, 4, 6, 7, 8, 12, 14>;


namespace kernel {


template <typename T>
__host__ __device__ __forceinline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}


template <typename ValueType, typename IndexType>
__device__ __forceinline__ bool segment_scan(
    const group::thread_block_tile<wsize> &group, const IndexType ind,
    ValueType *__restrict__ val)
{
    bool head = true;
#pragma unroll
    for (int i = 1; i < wsize; i <<= 1) {
        const IndexType add_ind = group.shfl_up(ind, i);
        ValueType add_val = zero<ValueType>();
        if (add_ind == ind && threadIdx.x >= i) {
            add_val = *val;
            if (i == 1) {
                head = false;
            }
        }
        add_val = group.shfl_down(add_val, i);
        if (threadIdx.x < wsize - i) {
            *val += add_val;
        }
    }
    return head;
}


template <typename ValueType, typename IndexType>
__device__ __forceinline__ bool block_segment_scan_reverse(
    const IndexType *__restrict__ ind, ValueType *__restrict__ val)
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
    IndexType *__restrict__ row, IndexType *__restrict__ row_end,
    const IndexType row_predict, const IndexType row_predict_end,
    const IndexType *__restrict__ row_ptr)
{
    if (!overflow || ind < data_size) {
        if (ind >= *row_end) {
            *row = row_predict;
            *row_end = row_predict_end;
            for (; ind >= *row_end; *row_end = row_ptr[++*row + 1])
                ;
        }

    } else {
        *row = num_rows - 1;
        *row_end = data_size;
    }
}


template <typename ValueType, typename IndexType, typename Closure>
__device__ __forceinline__ void warp_atomic_add(
    const group::thread_block_tile<wsize> &group, bool force_write,
    ValueType *__restrict__ val, const IndexType row, ValueType *__restrict__ c,
    const size_type c_stride, const IndexType column_id, Closure scale)
{
    // do a local scan to avoid atomic collisions
    const bool need_write = segment_scan(group, row, val);
    if (need_write && force_write) {
        atomic_add(&(c[row * c_stride + column_id]), scale(*val));
    }
    if (!need_write || force_write) {
        *val = zero<ValueType>();
    }
}


template <bool last, typename ValueType, typename IndexType, typename Closure>
__device__ __forceinline__ void process_window(
    const group::thread_block_tile<wsize> &group, const IndexType num_rows,
    const IndexType data_size, const IndexType ind, IndexType *__restrict__ row,
    IndexType *__restrict__ row_end, IndexType *__restrict__ nrow,
    IndexType *__restrict__ nrow_end, ValueType *__restrict__ temp_val,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride, const IndexType column_id, Closure scale)
{
    const IndexType curr_row = *row;
    find_next_row<last>(num_rows, data_size, ind, row, row_end, *nrow,
                        *nrow_end, row_ptrs);
    // segmented scan
    if (group.any(curr_row != *row)) {
        warp_atomic_add(group, curr_row != *row, temp_val, curr_row, c,
                        c_stride, column_id, scale);
        *nrow = group.shfl(*row, wsize - 1);
        *nrow_end = group.shfl(*row_end, wsize - 1);
    }

    if (!last || ind < data_size) {
        const auto col = col_idxs[ind];
        *temp_val += val[ind] * b[col * b_stride + column_id];
    }
}


template <typename IndexType>
__device__ __forceinline__ IndexType get_warp_start_idx(
    const IndexType nwarps, const IndexType nnz, const IndexType warp_idx)
{
    const long long cache_lines = ceildivT<IndexType>(nnz, wsize);
    return (warp_idx * cache_lines / nwarps) * wsize;
}


template <typename ValueType, typename IndexType, typename Closure>
__device__ __forceinline__ void spmv_kernel(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride, Closure scale)
{
    const IndexType warp_idx = blockIdx.x * warps_in_block + threadIdx.y;
    const IndexType column_id = blockIdx.y;
    if (warp_idx >= nwarps) {
        return;
    }
    const IndexType data_size = row_ptrs[num_rows];
    const IndexType start = get_warp_start_idx(nwarps, data_size, warp_idx);
    const IndexType end =
        min(get_warp_start_idx(nwarps, data_size, warp_idx + 1),
            ceildivT<IndexType>(data_size, wsize) * wsize);
    auto row = srow[warp_idx];
    auto row_end = row_ptrs[row + 1];
    auto nrow = row;
    auto nrow_end = row_end;
    ValueType temp_val = zero<ValueType>();
    IndexType ind = start + threadIdx.x;
    find_next_row<true>(num_rows, data_size, ind, &row, &row_end, nrow,
                        nrow_end, row_ptrs);
    const IndexType ind_end = end - wsize;
    const auto tile_block =
        group::tiled_partition<wsize>(group::this_thread_block());
    for (; ind < ind_end; ind += wsize) {
        process_window<false>(tile_block, num_rows, data_size, ind, &row,
                              &row_end, &nrow, &nrow_end, &temp_val, val,
                              col_idxs, row_ptrs, b, b_stride, c, c_stride,
                              column_id, scale);
    }
    process_window<true>(tile_block, num_rows, data_size, ind, &row, &row_end,
                         &nrow, &nrow_end, &temp_val, val, col_idxs, row_ptrs,
                         b, b_stride, c, c_stride, column_id, scale);
    warp_atomic_add(tile_block, true, &temp_val, row, c, c_stride, column_id,
                    scale);
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c,
                c_stride, [](const ValueType &x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c,
                c_stride, [&scale_factor](const ValueType &x) {
                    return scale_factor * x;
                });
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void set_zero(
    const size_type nnz, ValueType *__restrict__ val)
{
    const auto ind =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (ind < nnz) {
        val[ind] = zero<ValueType>();
    }
}


template <typename IndexType>
__forceinline__ __device__ void merge_path_search(
    const IndexType diagonal, const IndexType a_len, const IndexType b_len,
    const IndexType *__restrict__ a, const IndexType offset_b,
    IndexType *__restrict__ x, IndexType *__restrict__ y)
{
    auto x_min = max(diagonal - b_len, zero<IndexType>());
    auto x_max = min(diagonal, a_len);
    while (x_min < x_max) {
        auto pivot = (x_min + x_max) >> 1;
        if (a[pivot] <= offset_b + diagonal - pivot - 1) {
            x_min = pivot + 1;
        } else {
            x_max = pivot;
        }
    }

    *x = min(x_min, a_len);
    *y = diagonal - x_min;
}


template <typename ValueType, typename IndexType, typename Alpha_op>
__device__ void reduce(const IndexType nwarps,
                       const ValueType *__restrict__ last_val,
                       const IndexType *__restrict__ last_row,
                       ValueType *__restrict__ c, const size_type c_stride,
                       Alpha_op alpha_op)
{
    const IndexType cache_lines = ceildivT<IndexType>(nwarps, spmv_block_size);
    const IndexType tid = threadIdx.x;
    const IndexType start = min(tid * cache_lines, nwarps);
    const IndexType end = min((tid + 1) * cache_lines, nwarps);
    ValueType value = zero<ValueType>();
    IndexType row = last_row[nwarps - 1];
    if (start < nwarps) {
        value = last_val[start];
        row = last_row[start];
        for (IndexType i = start + 1; i < end; i++) {
            if (last_row[i] != row) {
                c[row * c_stride] += alpha_op(value);
                row = last_row[i];
                value = last_val[i];
            } else {
                value += last_val[i];
            }
        }
    }
    __shared__ UninitializedArray<IndexType, spmv_block_size> tmp_ind;
    __shared__ UninitializedArray<ValueType, spmv_block_size> tmp_val;
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = row;
    group::this_thread_block().sync();
    bool last = block_segment_scan_reverse(static_cast<IndexType *>(tmp_ind),
                                           static_cast<ValueType *>(tmp_val));
    group::this_thread_block().sync();
    if (last) {
        c[row * c_stride] += alpha_op(tmp_val[threadIdx.x]);
    }
}


template <int items_per_thread, typename ValueType, typename IndexType,
          typename Alpha_op, typename Beta_op>
__device__ void merge_path_spmv(
    const IndexType num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride,
    IndexType *__restrict__ row_out, ValueType *__restrict__ val_out,
    Alpha_op alpha_op, Beta_op beta_op)
{
    const auto *row_end_ptrs = row_ptrs + 1;
    const auto nnz = row_ptrs[num_rows];
    const IndexType num_merge_items = num_rows + nnz;
    const auto block_items = spmv_block_size * items_per_thread;
    __shared__ IndexType shared_row_ptrs[block_items];
    const IndexType diagonal =
        min(static_cast<IndexType>(block_items * blockIdx.x), num_merge_items);
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
    merge_path_search(static_cast<IndexType>(items_per_thread * threadIdx.x),
                      block_num_rows, block_num_nonzeros, shared_row_ptrs,
                      block_start_y, &start_x, &start_y);


    IndexType ind = block_start_y + start_y;
    IndexType row_i = block_start_x + start_x;
    ValueType value = zero<ValueType>();
#pragma unroll
    for (IndexType i = 0; i < items_per_thread; i++) {
        if (row_i < num_rows) {
            if (start_x == block_num_rows || ind < shared_row_ptrs[start_x]) {
                value += val[ind] * b[col_idxs[ind] * b_stride];
                ind++;
            } else {
                c[row_i * c_stride] =
                    alpha_op(value) + beta_op(c[row_i * c_stride]);
                start_x++;
                row_i++;
                value = zero<ValueType>();
            }
        }
    }
    group::this_thread_block().sync();
    IndexType *tmp_ind = shared_row_ptrs;
    ValueType *tmp_val =
        reinterpret_cast<ValueType *>(shared_row_ptrs + spmv_block_size);
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = row_i;
    group::this_thread_block().sync();
    bool last = block_segment_scan_reverse(static_cast<IndexType *>(tmp_ind),
                                           static_cast<ValueType *>(tmp_val));
    if (threadIdx.x == spmv_block_size - 1) {
        row_out[blockIdx.x] = min(end_x, num_rows - 1);
        val_out[blockIdx.x] = tmp_val[threadIdx.x];
    } else if (last) {
        c[row_i * c_stride] += alpha_op(tmp_val[threadIdx.x]);
    }
}

template <int items_per_thread, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_merge_path_spmv(
    const IndexType num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride,
    IndexType *__restrict__ row_out, ValueType *__restrict__ val_out)
{
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c, c_stride,
        row_out, val_out, [](ValueType &x) { return x; },
        [](ValueType &x) { return zero<ValueType>(); });
}


template <int items_per_thread, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_merge_path_spmv(
    const IndexType num_rows, const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    const ValueType *__restrict__ beta, ValueType *__restrict__ c,
    const size_type c_stride, IndexType *__restrict__ row_out,
    ValueType *__restrict__ val_out)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c, c_stride,
        row_out, val_out, [&alpha_val](ValueType &x) { return alpha_val * x; },
        [&beta_val](ValueType &x) { return beta_val * x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_reduce(
    const IndexType nwarps, const ValueType *__restrict__ last_val,
    const IndexType *__restrict__ last_row, ValueType *__restrict__ c,
    const size_type c_stride)
{
    reduce(nwarps, last_val, last_row, c, c_stride,
           [](ValueType &x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_reduce(
    const IndexType nwarps, const ValueType *__restrict__ last_val,
    const IndexType *__restrict__ last_row, const ValueType *__restrict__ alpha,
    ValueType *__restrict__ c, const size_type c_stride)
{
    const auto alpha_val = alpha[0];
    reduce(nwarps, last_val, last_row, c, c_stride,
           [&alpha_val](ValueType &x) { return alpha_val * x; });
}


template <typename ValueType, typename IndexType, typename Closure>
__device__ void classical_spmv(const size_type num_rows,
                               const ValueType *__restrict__ val,
                               const IndexType *__restrict__ col_idxs,
                               const IndexType *__restrict__ row_ptrs,
                               const ValueType *__restrict__ b,
                               const size_type b_stride,
                               ValueType *__restrict__ c,
                               const size_type c_stride, Closure scale)
{
    const auto tid =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tid >= num_rows) {
        return;
    }
    const auto column_id = blockIdx.y;
    const auto ind_end = row_ptrs[tid + 1];
    ValueType temp_value = zero<ValueType>();
    for (auto ind = row_ptrs[tid]; ind < ind_end; ind++) {
        temp_value += val[ind] * b[col_idxs[ind] * b_stride + column_id];
    }
    c[tid * c_stride + column_id] =
        scale(temp_value, c[tid * c_stride + column_id]);
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(classical_block_size) void abstract_classical_spmv(
    const size_type num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride)
{
    classical_spmv(num_rows, val, col_idxs, row_ptrs, b, b_stride, c, c_stride,
                   [](const ValueType &x, const ValueType &y) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(classical_block_size) void abstract_classical_spmv(
    const size_type num_rows, const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const ValueType *__restrict__ b,
    const size_type b_stride, const ValueType *__restrict__ beta,
    ValueType *__restrict__ c, const size_type c_stride)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    classical_spmv(
        num_rows, val, col_idxs, row_ptrs, b, b_stride, c, c_stride,
        [&alpha_val, &beta_val](const ValueType &x, const ValueType &y) {
            return alpha_val * x + beta_val * y;
        });
}


}  // namespace kernel


namespace host_kernel {


template <int items_per_thread, typename ValueType, typename IndexType>
void merge_path_spmv(syn::value_list<int, items_per_thread>,
                     std::shared_ptr<const CudaExecutor> exec,
                     const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Dense<ValueType> *b,
                     matrix::Dense<ValueType> *c,
                     const matrix::Dense<ValueType> *alpha = nullptr,
                     const matrix::Dense<ValueType> *beta = nullptr)
{
    const IndexType total = a->get_size()[0] + a->get_num_stored_elements();
    const IndexType grid_num =
        ceildiv(total, spmv_block_size * items_per_thread);
    const dim3 grid(grid_num);
    const dim3 block(spmv_block_size);
    Array<IndexType> row_out(exec, grid_num);
    Array<ValueType> val_out(exec, grid_num);

    for (IndexType column_id = 0; column_id < b->get_size()[1]; column_id++) {
        if (alpha == nullptr && beta == nullptr) {
            const auto b_vals = b->get_const_values() + column_id;
            auto c_vals = c->get_values() + column_id;
            kernel::abstract_merge_path_spmv<items_per_thread>
                <<<grid, block, 0, 0>>>(
                    static_cast<IndexType>(a->get_size()[0]),
                    as_cuda_type(a->get_const_values()),
                    a->get_const_col_idxs(),
                    as_cuda_type(a->get_const_row_ptrs()),
                    as_cuda_type(a->get_const_srow()), as_cuda_type(b_vals),
                    b->get_stride(), as_cuda_type(c_vals), c->get_stride(),
                    as_cuda_type(row_out.get_data()),
                    as_cuda_type(val_out.get_data()));
            kernel::abstract_reduce<<<1, spmv_block_size>>>(
                grid_num, as_cuda_type(val_out.get_data()),
                as_cuda_type(row_out.get_data()), as_cuda_type(c_vals),
                c->get_stride());

        } else if (alpha != nullptr && beta != nullptr) {
            const auto b_vals = b->get_const_values() + column_id;
            auto c_vals = c->get_values() + column_id;
            kernel::abstract_merge_path_spmv<items_per_thread>
                <<<grid, block, 0, 0>>>(
                    static_cast<IndexType>(a->get_size()[0]),
                    as_cuda_type(alpha->get_const_values()),
                    as_cuda_type(a->get_const_values()),
                    a->get_const_col_idxs(),
                    as_cuda_type(a->get_const_row_ptrs()),
                    as_cuda_type(a->get_const_srow()), as_cuda_type(b_vals),
                    b->get_stride(), as_cuda_type(beta->get_const_values()),
                    as_cuda_type(c_vals), c->get_stride(),
                    as_cuda_type(row_out.get_data()),
                    as_cuda_type(val_out.get_data()));
            kernel::abstract_reduce<<<1, spmv_block_size>>>(
                grid_num, as_cuda_type(val_out.get_data()),
                as_cuda_type(row_out.get_data()),
                as_cuda_type(alpha->get_const_values()), as_cuda_type(c_vals),
                c->get_stride());
        } else {
            GKO_KERNEL_NOT_FOUND;
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_merge_path_spmv, merge_path_spmv);


template <typename ValueType, typename IndexType>
int compute_items_per_thread(std::shared_ptr<const CudaExecutor> exec)
{
    const int version = exec->get_major_version()
                        << 4 + exec->get_minor_version();
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
    // Ensure that satisfy:
    // sizeof(IndexType) + sizeof(ValueType)
    // <= items_per_thread * sizeof(IndexType)
    constexpr int minimal_num =
        ceildiv(sizeof(IndexType) + sizeof(ValueType), sizeof(IndexType));
    int items_per_thread = num_item * 4 / sizeof(IndexType);
    return std::max(minimal_num, items_per_thread);
}


}  // namespace host_kernel


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    if (a->get_strategy()->get_name() == "load_balance") {
        zero_array(c->get_num_stored_elements(), c->get_values());
        const IndexType nwarps = a->get_num_srow_elements();
        if (nwarps > 0) {
            const dim3 csr_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 csr_grid(ceildiv(nwarps, warps_in_block),
                                b->get_size()[1]);
            kernel::abstract_spmv<<<csr_grid, csr_block>>>(
                nwarps, static_cast<IndexType>(a->get_size()[0]),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_ptrs()),
                as_cuda_type(a->get_const_srow()),
                as_cuda_type(b->get_const_values()),
                as_cuda_type(b->get_stride()), as_cuda_type(c->get_values()),
                as_cuda_type(c->get_stride()));
        }
    } else if (a->get_strategy()->get_name() == "merge_path") {
        int items_per_thread =
            host_kernel::compute_items_per_thread<ValueType, IndexType>(exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
    } else if (a->get_strategy()->get_name() == "classical") {
        const dim3 grid(ceildiv(a->get_size()[0], classical_block_size),
                        b->get_size()[1]);
        kernel::abstract_classical_spmv<<<grid, classical_block_size>>>(
            a->get_size()[0], as_cuda_type(a->get_const_values()),
            a->get_const_col_idxs(), as_cuda_type(a->get_const_row_ptrs()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
    } else if (a->get_strategy()->get_name() == "cusparse") {
        if (cusparse::is_supported<ValueType, IndexType>::value) {
            // TODO: add implementation for int64 and multiple RHS
            auto handle = exec->get_cusparse_handle();
            auto descr = cusparse::create_mat_descr();
            GKO_ASSERT_NO_CUSPARSE_ERRORS(
                cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

            auto row_ptrs = a->get_const_row_ptrs();
            auto col_idxs = a->get_const_col_idxs();
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            if (b->get_stride() != 1 || c->get_stride() != 1)
                GKO_NOT_IMPLEMENTED;

            cusparse::spmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           a->get_size()[0], a->get_size()[1],
                           a->get_num_stored_elements(), &alpha, descr,
                           a->get_const_values(), row_ptrs, col_idxs,
                           b->get_const_values(), &beta, c->get_values());

            GKO_ASSERT_NO_CUSPARSE_ERRORS(
                cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));

            cusparse::destroy(descr);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Csr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    if (a->get_strategy()->get_name() == "load_balance") {
        dense::scale(exec, beta, c);

        const IndexType nwarps = a->get_num_srow_elements();

        if (nwarps > 0) {
            const dim3 csr_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 csr_grid(ceildiv(nwarps, warps_in_block),
                                b->get_size()[1]);
            kernel::abstract_spmv<<<csr_grid, csr_block>>>(
                nwarps, static_cast<IndexType>(a->get_size()[0]),
                as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_ptrs()),
                as_cuda_type(a->get_const_srow()),
                as_cuda_type(b->get_const_values()),
                as_cuda_type(b->get_stride()), as_cuda_type(c->get_values()),
                as_cuda_type(c->get_stride()));
        }
    } else if (a->get_strategy()->get_name() == "cusparse") {
        if (cusparse::is_supported<ValueType, IndexType>::value) {
            // TODO: add implementation for int64 and multiple RHS
            auto descr = cusparse::create_mat_descr();

            auto row_ptrs = a->get_const_row_ptrs();
            auto col_idxs = a->get_const_col_idxs();

            if (b->get_stride() != 1 || c->get_stride() != 1)
                GKO_NOT_IMPLEMENTED;

            cusparse::spmv(exec->get_cusparse_handle(),
                           CUSPARSE_OPERATION_NON_TRANSPOSE, a->get_size()[0],
                           a->get_size()[1], a->get_num_stored_elements(),
                           alpha->get_const_values(), descr,
                           a->get_const_values(), row_ptrs, col_idxs,
                           b->get_const_values(), beta->get_const_values(),
                           c->get_values());

            cusparse::destroy(descr);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    } else if (a->get_strategy()->get_name() == "classical") {
        const dim3 grid(ceildiv(a->get_size()[0], classical_block_size),
                        b->get_size()[1]);
        kernel::abstract_classical_spmv<<<grid, classical_block_size>>>(
            a->get_size()[0], as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            as_cuda_type(a->get_const_row_ptrs()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(beta->get_const_values()),
            as_cuda_type(c->get_values()), c->get_stride());
    } else if (a->get_strategy()->get_name() == "merge_path") {
        int items_per_thread =
            host_kernel::compute_items_per_thread<ValueType, IndexType>(exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha,
            beta);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


namespace kernel {


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void convert_row_ptrs_to_idxs(
    size_type num_rows, const IndexType *__restrict__ ptrs,
    IndexType *__restrict__ idxs)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < num_rows) {
        for (auto i = ptrs[tidx]; i < ptrs[tidx + 1]; i++) {
            idxs[i] = tidx;
        }
    }
}


}  // namespace kernel


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const CudaExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs)
{
    const auto grid_dim = ceildiv(num_rows, default_block_size);

    kernel::convert_row_ptrs_to_idxs<<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(ptrs), as_cuda_type(idxs));
}


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Coo<ValueType, IndexType> *result,
                    const matrix::Csr<ValueType, IndexType> *source)
{
    auto num_rows = result->get_size()[0];

    auto row_idxs = result->get_row_idxs();
    const auto source_row_ptrs = source->get_const_row_ptrs();

    convert_row_ptrs_to_idxs(exec, source_row_ptrs, num_rows, row_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_COO_KERNEL);


namespace kernel {


template <typename ValueType>
__global__
    __launch_bounds__(cuda_config::max_block_size) void initialize_zero_dense(
        size_type num_rows, size_type num_cols, size_type stride,
        ValueType *__restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type num_rows, const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, size_type stride,
    ValueType *__restrict__ result)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < num_rows) {
        for (auto i = row_ptrs[tidx]; i < row_ptrs[tidx + 1]; i++) {
            result[stride * tidx + col_idxs[i]] = values[i];
        }
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      matrix::Dense<ValueType> *result,
                      const matrix::Csr<ValueType, IndexType> *source)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();
    const auto row_ptrs = source->get_const_row_ptrs();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();

    const dim3 block_size(cuda_config::warp_size,
                          cuda_config::max_block_size / cuda_config::warp_size,
                          1);
    const dim3 init_grid_dim(ceildiv(stride, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);
    kernel::initialize_zero_dense<<<init_grid_dim, block_size>>>(
        num_rows, num_cols, stride, as_cuda_type(result->get_values()));

    auto grid_dim = ceildiv(num_rows, default_block_size);
    kernel::fill_in_dense<<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(row_ptrs), as_cuda_type(col_idxs),
        as_cuda_type(vals), stride, as_cuda_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_dense(
    std::shared_ptr<const CudaExecutor> exec, matrix::Dense<ValueType> *result,
    matrix::Csr<ValueType, IndexType> *source) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_MOVE_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const CudaExecutor> exec,
               matrix::Csr<ValueType, IndexType> *trans,
               const matrix::Csr<ValueType, IndexType> *orig)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;

        cusparse::transpose(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_col_idxs(), trans->get_row_ptrs(), copyValues, idxBase);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


namespace {


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void conjugate_kernel(
    size_type num_nonzeros, ValueType *__restrict__ val)
{
    const auto tidx =
        static_cast<size_type>(blockIdx.x) * default_block_size + threadIdx.x;

    if (tidx < num_nonzeros) {
        val[tidx] = conj(val[tidx]);
    }
}


}  //  namespace


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *trans,
                    const matrix::Csr<ValueType, IndexType> *orig)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(
            ceildiv(trans->get_num_stored_elements(), block_size.x), 1, 1);

        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;

        cusparse::transpose(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_col_idxs(), trans->get_row_ptrs(), copyValues, idxBase);

        conjugate_kernel<<<grid_size, block_size, 0, 0>>>(
            trans->get_num_stored_elements(),
            as_cuda_type(trans->get_values()));
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


}  // namespace csr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
