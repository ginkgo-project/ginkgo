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


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/shuffle.cuh"
#include "cuda/components/synchronization.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace csri {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * cuda_config::warp_size;


namespace {


template <typename ValueType, typename IndexType>
__device__ __forceinline__ void segment_scan(IndexType ind, ValueType *val,
                                             bool *head)
{
#pragma unroll
    for (int i = 1; i < cuda_config::warp_size; i <<= 1) {
        const IndexType add_ind = warp::shuffle_up(ind, i);
        ValueType add_val = zero<ValueType>();
        if (threadIdx.x >= i && add_ind == ind) {
            add_val = *val;
            if (i == 1) {
                *head = false;
            }
        }
        add_val = warp::shuffle_down(add_val, i);
        if (threadIdx.x < cuda_config::warp_size - i) {
            *val += add_val;
        }
    }
}


template <bool overflow, typename IndexType>
__device__ __forceinline__ void find_next_row(
    const size_type num_rows, const size_type data_size, const IndexType ind,
    IndexType *__restrict__ row, IndexType *__restrict__ row_end,
    const IndexType row_predict, const IndexType row_predict_end,
    const IndexType *__restrict__ row_ptr)
{
    if (!overflow || ind < data_size) {
        if (ind >= *row_end) {
            *row = row_predict;
            *row_end = row_predict_end;
            while (ind >= *row_end) {
                *row_end = row_ptr[++*row + 1];
            }
        }
    } else {
        *row = num_rows - 1;
        *row_end = data_size;
    }
}


template <bool overflow, bool last, typename ValueType, typename IndexType,
          typename Closure>
__device__ __forceinline__ void process_window(
    const size_type num_rows, const size_type data_size, const IndexType ind,
    const IndexType column_id, IndexType *__restrict__ row,
    IndexType *__restrict__ row_end, IndexType *__restrict__ nrow,
    IndexType *__restrict__ nrow_end, ValueType *__restrict__ temp_val,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride, Closure scale)
{
    if (!overflow || ind < data_size) {
        *temp_val += val[ind] * b[col_idxs[ind] * b_stride + column_id];
    }
    const auto curr_row = *row;
    if (!last) {
        find_next_row<overflow>(
            num_rows, data_size,
            static_cast<IndexType>(ind + cuda_config::warp_size), row, row_end,
            *nrow, *nrow_end, row_ptrs);
    }
    // segmented scan
    if (last || warp::any(curr_row != *row)) {
        bool is_first_in_segment = true;
        segment_scan(curr_row, temp_val, &is_first_in_segment);
        if (is_first_in_segment) {
            atomic_add(&(c[curr_row * c_stride + column_id]), scale(*temp_val));
        }
        if (!last) {
            *nrow = warp::shuffle(*row, cuda_config::warp_size - 1);
            *nrow_end = warp::shuffle(*row_end, cuda_config::warp_size - 1);
            *temp_val = zero<ValueType>();
        }
    }
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
__device__ void spmv_kernel(const size_type nwarps, const size_type num_rows,
                            const ValueType *__restrict__ val,
                            const IndexType *__restrict__ col_idxs,
                            const IndexType *__restrict__ row_ptrs,
                            const IndexType *__restrict__ srow,
                            const ValueType *__restrict__ b,
                            const size_type b_stride, ValueType *__restrict__ c,
                            const size_type c_stride, Closure scale)
{
    const auto warp_idx =
        static_cast<size_type>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (warp_idx >= nwarps) {
        return;
    }
    const auto data_size = row_ptrs[num_rows];
    const auto num_lines = ceildiv(data_size, nwarps * cuda_config::warp_size);
    ValueType temp_val = zero<ValueType>();

    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x *
                           blockDim.y * num_lines +
                       threadIdx.y * blockDim.x * num_lines;
    const IndexType column_id = blockIdx.y;
    auto num = min((data_size > start) *
                       ceildiv(data_size - start, cuda_config::warp_size),
                   num_lines);
    const IndexType ind_start = start + threadIdx.x;
    const IndexType ind_end = ind_start + (num - 2) * cuda_config::warp_size;
    auto row = srow[warp_idx];
    auto row_end = row_ptrs[row + 1];
    auto nrow = row;
    auto nrow_end = row_end;
    IndexType ind = ind_start;
    find_next_row<true>(num_rows, data_size, ind, &row, &row_end, nrow,
                        nrow_end, row_ptrs);
    for (; ind < ind_end; ind += cuda_config::warp_size) {
        process_window<false, false>(num_rows, data_size, ind, column_id, &row,
                                     &row_end, &nrow, &nrow_end, &temp_val, val,
                                     col_idxs, row_ptrs, srow, b, b_stride, c,
                                     c_stride, scale);
    }
    if (num > 1) {
        ind = ind_end;
        process_window<true, false>(num_rows, data_size, ind, column_id, &row,
                                    &row_end, &nrow, &nrow_end, &temp_val, val,
                                    col_idxs, row_ptrs, srow, b, b_stride, c,
                                    c_stride, scale);
    }
    if (num > 0) {
        ind = ind_end + cuda::cuda_config::warp_size;
        process_window<true, true>(num_rows, data_size, ind, column_id, &row,
                                   &row_end, &nrow, &nrow_end, &temp_val, val,
                                   col_idxs, row_ptrs, srow, b, b_stride, c,
                                   c_stride, scale);
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nwarps, const size_type num_rows,
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
    const size_type nwarps, const size_type num_rows,
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
    const auto len = ceildiv(nwarps, cuda_config::warp_size);
    const size_type start = len * threadIdx.x;
    const size_type end = (threadIdx.x == cuda_config::warp_size - 1)
                              ? nwarps
                              : (threadIdx.x + 1) * len;
    auto temp_val = last_val[start];
    auto temp_row = last_row[start];
    for (size_type i = start + 1; i < end; i++) {
        if (last_row[i] != temp_row) {
            c[temp_row] += temp_val;
            temp_row = last_row[i];
            temp_val = last_val[i];
        } else {
            temp_val += last_val[i];
        }
    }
    bool is_first_in_segment = true;
    segment_scan(temp_row, &temp_val, &is_first_in_segment);
    if (is_first_in_segment) {
        c[temp_row] += temp_val;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void merge_path_spmv(
    const IndexType num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride,
    IndexType *__restrict__ row_out, ValueType *__restrict__ val_out)
{
    const IndexType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const IndexType num_threads = blockDim.x * gridDim.x;
    const auto *row_end_ptrs = row_ptrs + 1;
    const auto nnz = row_ptrs[num_rows];
    const IndexType num_merge_items = num_rows + nnz;
    const IndexType items_per_thread = ceildiv(num_merge_items, num_threads);

    IndexType diagonal = min(items_per_thread * tid, num_merge_items);
    IndexType diagonal_end = min(diagonal + items_per_thread, num_merge_items);
    IndexType start_x;
    IndexType start_y;
    IndexType end_x;
    IndexType end_y;
    merge_path_search(diagonal, num_rows, nnz, row_end_ptrs, &start_x,
                      &start_y);
    merge_path_search(diagonal_end, num_rows, nnz, row_end_ptrs, &end_x,
                      &end_y);

    ValueType value = zero<ValueType>();
    for (IndexType i = 0; i < items_per_thread; i++) {
        if (start_y < row_end_ptrs[start_x]) {
            value += val[start_y] * b[col_idxs[start_y]];
            start_y++;
        } else {
            c[start_x] = value;
            value = zero<ValueType>();
            start_x++;
        }
    }

    for (; start_y < end_y; start_y++) {
        value += val[start_y] * b[col_idxs[start_y]];
    }
    row_out[tid] = end_x;
    val_out[tid] = value;
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Csri<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    if (a->get_strategy()->get_name() == "load_balance") {
        auto c_size = c->get_num_stored_elements();
        const dim3 grid(ceildiv(c_size, default_block_size));
        const dim3 block(default_block_size);
        set_zero<<<grid, block>>>(c_size, as_cuda_type(c->get_values()));

        auto a_size = a->get_num_stored_elements();
        auto nwarps = a->get_num_srow_elements();

        if (nwarps > 0) {
            const dim3 csri_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 csri_grid(ceildiv(nwarps, warps_in_block),
                                 b->get_size()[1]);
            abstract_spmv<<<csri_grid, csri_block>>>(
                nwarps, a->get_size()[0], as_cuda_type(a->get_const_values()),
                a->get_const_col_idxs(), as_cuda_type(a->get_const_row_ptrs()),
                as_cuda_type(a->get_const_srow()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        }
    } else if (a->get_strategy()->get_name() == "merge_path") {
        const IndexType num_item = 8;
        const IndexType total = a->get_size()[0] + a->get_num_stored_elements();
        const IndexType grid_num = ceildiv(total, spmv_block_size * num_item);
        const dim3 grid(grid_num);
        const dim3 block(spmv_block_size);
        Array<IndexType> row_out(exec, grid_num * spmv_block_size);
        Array<ValueType> val_out(exec, grid_num * spmv_block_size);
        merge_path_spmv<<<grid, block>>>(
            static_cast<IndexType>(a->get_size()[0]),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            as_cuda_type(a->get_const_row_ptrs()),
            as_cuda_type(a->get_const_srow()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride(),
            as_cuda_type(row_out.get_data()), as_cuda_type(val_out.get_data()));
        reduce<<<1, 32>>>(static_cast<size_type>(grid_num * spmv_block_size),
                          as_cuda_type(val_out.get_data()),
                          as_cuda_type(row_out.get_data()),
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

        auto a_size = a->get_num_stored_elements();
        auto nwarps = a->get_num_srow_elements();

        if (nwarps > 0) {
            int num_lines = ceildiv(a_size, nwarps * cuda_config::warp_size);
            const dim3 csri_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 csri_grid(ceildiv(nwarps, warps_in_block),
                                 b->get_size()[1]);
            abstract_spmv<<<csri_grid, csri_block>>>(
                nwarps, a->get_size()[0],
                as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_ptrs()),
                as_cuda_type(a->get_const_srow()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
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
