/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/csri_kernels.hpp"

#include <cstdio>
#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/shuffle.cuh"
#include "cuda/components/synchronization.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace csri {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * cuda_config::warp_size;
constexpr int classical_block_size = 64;
constexpr int wsize = cuda_config::warp_size;


namespace {


template <typename T>
__host__ __device__ __forceinline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}


template <typename ValueType, typename IndexType>
__device__ __forceinline__ bool segment_scan(const IndexType ind,
                                             ValueType *__restrict__ val)
{
    bool head = true;
#pragma unroll
    for (int i = 1; i < wsize; i <<= 1) {
        const IndexType add_ind = warp::shuffle_up(ind, i);
        ValueType add_val = zero<ValueType>();
        if (add_ind == ind && threadIdx.x >= i) {
            add_val = *val;
            if (i == 1) {
                head = false;
            }
        }
        add_val = warp::shuffle_down(add_val, i);
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
        __syncthreads();
        val[threadIdx.x] += temp;
        __syncthreads();
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


template <bool clean = true, typename ValueType, typename IndexType,
          typename Closure>
__device__ __forceinline__ void warp_atomic_add(bool force_write,
                                                ValueType *__restrict__ val,
                                                IndexType ind,
                                                ValueType *__restrict__ out,
                                                Closure scale)
{
    // do a local scan to avoid atomic collisions
    const bool need_write = segment_scan(ind, val);
    if (need_write && force_write) {
        atomic_add(out + ind, scale(*val));
    }
    if (!need_write || force_write) {
        *val = zero<ValueType>();
    }
}


template <bool last, typename ValueType, typename IndexType, typename Closure>
__device__ __forceinline__ void process_window(
    const IndexType num_rows, const IndexType data_size, const IndexType ind,
    IndexType *__restrict__ row, IndexType *__restrict__ row_end,
    IndexType *__restrict__ nrow, IndexType *__restrict__ nrow_end,
    ValueType *__restrict__ temp_val, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const ValueType *__restrict__ b,
    ValueType *__restrict__ c, Closure scale)
{
    const IndexType curr_row = *row;
    find_next_row<last>(num_rows, data_size, ind, row, row_end, *nrow,
                        *nrow_end, row_ptrs);
    // segmented scan
    if (warp::any(curr_row != *row)) {
        warp_atomic_add(curr_row != *row, temp_val, curr_row, c, scale);
        *nrow = warp::shuffle(*row, wsize - 1);
        *nrow_end = warp::shuffle(*row_end, wsize - 1);
    }

    if (!last || ind < data_size) {
        const auto col = col_idxs[ind];
        *temp_val += val[ind] * b[col];
    }
}


template <typename IndexType>
__device__ __forceinline__ IndexType get_warp_start_idx(
    const IndexType nwarps, const IndexType nnz, const IndexType warp_idx)
{
    const long long cache_lines = ceildivT<IndexType>(nnz, wsize);
    return (warp_idx * cache_lines / nwarps) * wsize;
}


/**
 * The device function of CSRI spmv
 *
 * @param nwarps  the number of warps
 * @param num_rows  the number of rows
 * @param val  the value array of the matrix
 * @param col_idxs  the column index array of the matrix
 * @param row_ptrs  the row pointer array of the matrix
 * @param srow  the starting row index array of the matrix
 * @param b  the input dense vector
 * @param b_stride  the stride of the input dense vector
 * @param c  the output dense vector
 * @param c_stride  the stride of the output dense vector
 * @param scale  the function on the added value
 */
template <typename ValueType, typename IndexType, typename Closure>
__device__ __forceinline__ void spmv_kernel(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, ValueType *__restrict__ c, Closure scale)
{
    const IndexType warp_idx = blockIdx.x * warps_in_block + threadIdx.y;
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
    for (; ind < ind_end; ind += wsize) {
        process_window<false>(num_rows, data_size, ind, &row, &row_end, &nrow,
                              &nrow_end, &temp_val, val, col_idxs, row_ptrs, b,
                              c, scale);
    }
    process_window<true>(num_rows, data_size, ind, &row, &row_end, &nrow,
                         &nrow_end, &temp_val, val, col_idxs, row_ptrs, b, c,
                         scale);
    warp_atomic_add(true, &temp_val, row, c, scale);
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, ValueType *__restrict__ c)
{
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, c,
                [](const ValueType &x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, ValueType *__restrict__ c)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(
        nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, c,
        [&scale_factor](const ValueType &x) { return scale_factor * x; });
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

template <typename IndexType>
__device__ void merge_path_search(const IndexType diagonal,
                                  const IndexType a_len, const IndexType b_len,
                                  const IndexType *__restrict__ a,
                                  IndexType *__restrict__ x,
                                  IndexType *__restrict__ y)
{
    auto x_min = max(diagonal - b_len, zero<IndexType>());
    auto x_max = min(diagonal, a_len);
    while (x_min < x_max) {
        auto pivot = (x_min + x_max) >> 1;
        if (a[pivot] <= diagonal - pivot - 1) {
            x_min = pivot + 1;
        } else {
            x_max = pivot;
        }
    }

    *x = min(x_min, a_len);
    *y = diagonal - x_min;
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(cuda_config::warp_size) void reduce(
    const size_type nwarps, const ValueType *__restrict__ last_val,
    const IndexType *__restrict__ last_row, ValueType *__restrict__ c,
    const size_type c_stride)
{
    const auto len = nwarps / cuda_config::warp_size;
    const auto res = nwarps % cuda_config::warp_size;
    const IndexType start =
        (threadIdx.x < res) ? (len + 1) * threadIdx.x : len * threadIdx.x + res;
    const IndexType end = (threadIdx.x < res) ? start + len + 1 : start + len;
    auto temp_val = last_val[start];
    auto temp_row = last_row[start];
    for (IndexType i = start + 1; i < end; i++) {
        if (last_row[i] != temp_row) {
            c[temp_row] += temp_val;
            temp_row = last_row[i];
            temp_val = last_val[i];
        } else {
            temp_val += last_val[i];
        }
    }
    bool is_first_in_segment = segment_scan(temp_row, &temp_val);
    if (is_first_in_segment) {
        // printf("temp_row: %d\n", temp_row);
        c[temp_row] += temp_val;
    }
}


template <int items_per_thread, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void merge_path_spmv(
    const IndexType num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride,
    IndexType *__restrict__ row_out, ValueType *__restrict__ val_out)
{
    // const IndexType tid = blockDim.x * blockIdx.x + threadIdx.x;
    // // const IndexType num_threads = blockDim.x * gridDim.x;
    const auto *row_end_ptrs = row_ptrs + 1;
    const auto nnz = row_ptrs[num_rows];
    const IndexType num_merge_items = num_rows + nnz;
    // // const IndexType items_per_thread = ceildiv(num_merge_items,
    // num_threads);
    const auto block_items = spmv_block_size * items_per_thread;
    // __shared__ UninitializedArray<IndexType, block_items + 1>
    // shared_row_ptrs;
    __shared__ IndexType shared_row_ptrs[block_items];
    IndexType diagonal =
        min(static_cast<IndexType>(block_items * blockIdx.x), num_merge_items);
    IndexType diagonal_end = min(diagonal + block_items, num_merge_items);
    IndexType block_start_x;
    IndexType block_start_y;
    IndexType end_x;
    IndexType end_y;
    merge_path_search(diagonal, num_rows, nnz, row_end_ptrs, &block_start_x,
                      &block_start_y);
    merge_path_search(diagonal_end, num_rows, nnz, row_end_ptrs, &end_x,
                      &end_y);
    const IndexType block_num_rows = end_x - block_start_x;
    const IndexType block_num_nonzeros = end_y - block_start_y;
    // Move rows_ptrs to shared_memory;
    for (int i = threadIdx.x;
         i <= block_num_rows && block_start_x + i < num_rows;
         i += spmv_block_size) {
        shared_row_ptrs[i] = row_end_ptrs[block_start_x + i];
    }
    __syncthreads();

    // // diagonal = min(items_per_thread * tid, num_merge_items);
    // // diagonal_end = min(diagonal + items_per_thread, num_merge_items);
    IndexType start_x;
    IndexType start_y;
    merge_path_search(static_cast<IndexType>(items_per_thread * threadIdx.x),
                      block_num_rows, block_num_nonzeros, shared_row_ptrs,
                      block_start_y, &start_x, &start_y);

    ValueType value = zero<ValueType>();
#pragma unroll
    for (IndexType i = 0; i < items_per_thread; i++) {
        const IndexType ind = block_start_y + start_y;
        const IndexType row_i = block_start_x + start_x;
        if (row_i < num_rows) {
            if (ind < shared_row_ptrs[start_x]) {
                value += val[ind] * b[col_idxs[ind]];
                start_y++;
            } else {
                c[row_i] = value;
                value = zero<ValueType>();
                start_x++;
            }
        }
    }
    __shared__ UninitializedArray<IndexType, spmv_block_size> tmp_ind;
    __shared__ UninitializedArray<ValueType, spmv_block_size> tmp_val;
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = block_start_x + start_x;
    __syncthreads();
    block::synchronize();
    bool last = block_segment_scan_reverse(static_cast<IndexType *>(tmp_ind),
                                           static_cast<ValueType *>(tmp_val));
    __syncthreads();
    if (threadIdx.x == spmv_block_size - 1) {
        // printf("end_x: %d, start_x: %d val: %lf\n", end_x,
        //        block_start_x + start_x, tmp_val[threadIdx.x]);
        row_out[blockIdx.x] = min(end_x, num_rows - 1);
        val_out[blockIdx.x] = tmp_val[threadIdx.x];
    } else if (last) {
        // printf("%d, %d\n", blockIdx.x, block_start_x + start_x);
        c[block_start_x + start_x] += tmp_val[threadIdx.x];
        // atomic_add(&c[block_start_x + start_x], tmp_val[threadIdx.x]);
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(64) void classical_spmv(
    const size_type num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride)
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
    c[tid * c_stride + column_id] = temp_value;
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Csri<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    if (a->get_strategy()->get_name() == "load_balance") {
        ASSERT_NO_CUDA_ERRORS(
            cudaMemset(c->get_values(), 0,
                       c->get_num_stored_elements() * sizeof(ValueType)));
        const IndexType nwarps = a->get_num_srow_elements();
        if (nwarps > 0) {
            const dim3 csri_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 csri_grid(ceildiv(nwarps, warps_in_block));
            abstract_spmv<<<csri_grid, csri_block>>>(
                nwarps, static_cast<IndexType>(a->get_size()[0]),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_ptrs()),
                as_cuda_type(a->get_const_srow()),
                as_cuda_type(b->get_const_values()),
                as_cuda_type(c->get_values()));
        }
    } else if (a->get_strategy()->get_name() == "merge_path") {
        const int num_item = 8;
        const IndexType total = a->get_size()[0] + a->get_num_stored_elements();
        const IndexType grid_num = ceildiv(total, spmv_block_size * num_item);
        const dim3 grid(grid_num);
        const dim3 block(spmv_block_size);
        Array<IndexType> row_out(exec, grid_num);
        Array<ValueType> val_out(exec, grid_num);
        merge_path_spmv<num_item><<<grid, block>>>(
            static_cast<IndexType>(a->get_size()[0]),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            as_cuda_type(a->get_const_row_ptrs()),
            as_cuda_type(a->get_const_srow()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride(),
            as_cuda_type(row_out.get_data()), as_cuda_type(val_out.get_data()));
        reduce<<<1, 32>>>(static_cast<size_type>(grid_num),
                          as_cuda_type(val_out.get_data()),
                          as_cuda_type(row_out.get_data()),
                          as_cuda_type(c->get_values()), c->get_stride());
    } else if (a->get_strategy()->get_name() == "classical") {
        classical_spmv<<<ceildiv(a->get_size()[0], classical_block_size),
                         classical_block_size>>>(
            a->get_size()[0], as_cuda_type(a->get_const_values()),
            a->get_const_col_idxs(), as_cuda_type(a->get_const_row_ptrs()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSRI_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Csri<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    if (a->get_strategy()->get_name() == "load_balance") {
        dense::scale(exec, beta, c);

        const IndexType nwarps = a->get_num_srow_elements();

        if (nwarps > 0) {
            const dim3 csri_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 csri_grid(ceildiv(nwarps, warps_in_block));
            abstract_spmv<<<csri_grid, csri_block>>>(
                nwarps, static_cast<IndexType>(a->get_size()[0]),
                as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_ptrs()),
                as_cuda_type(a->get_const_srow()),
                as_cuda_type(b->get_const_values()),
                as_cuda_type(c->get_values()));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSRI_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const CudaExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_CSRI_CONVERT_ROW_PTRS_TO_IDXS_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const CudaExecutor> exec, matrix::Dense<ValueType> *result,
    const matrix::Csri<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSRI_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_dense(std::shared_ptr<const CudaExecutor> exec,
                   matrix::Dense<ValueType> *result,
                   matrix::Csri<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSRI_MOVE_TO_DENSE_KERNEL);


}  // namespace csri
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
