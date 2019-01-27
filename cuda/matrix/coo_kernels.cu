/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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

#include "core/matrix/coo_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/synchronization.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace coo {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * cuda_config::warp_size;


namespace {


template <int subwarp_size = cuda_config::warp_size, typename ValueType,
          typename IndexType>
__device__ __forceinline__ void segment_scan(
    const group::thread_block_tile<subwarp_size> &group, IndexType ind, ValueType *val,
    bool *head)
{
#pragma unroll
    for (int i = 1; i < subwarp_size; i <<= 1) {
        const IndexType add_ind = group.shfl_up(ind, i);
        ValueType add_val = zero<ValueType>();
        if (threadIdx.x >= i && add_ind == ind) {
            add_val = *val;
            if (i == 1) {
                *head = false;
            }
        }
        add_val = group.shfl_down(add_val, i);
        if (threadIdx.x < subwarp_size - i) {
            *val += add_val;
        }
    }
}


/**
 * The device function of COO spmv
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_line  the maximum round of each warp
 * @param val  the value array of the matrix
 * @param col  the column index array of the matrix
 * @param row  the row index array of the matrix
 * @param b  the input dense vector
 * @param c  the output dense vector
 * @param scale  the function on the added value
 */
template <int subwarp_size = cuda_config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ void spmv_kernel(const size_type nnz, const size_type num_lines,
                            const ValueType *__restrict__ val,
                            const IndexType *__restrict__ col,
                            const IndexType *__restrict__ row,
                            const ValueType *__restrict__ b,
                            const size_type b_stride, ValueType *__restrict__ c,
                            const size_type c_stride, Closure scale)
{
    ValueType temp_val = zero<ValueType>();
    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x *
                           blockDim.y * num_lines +
                       threadIdx.y * blockDim.x * num_lines;
    const auto column_id = blockIdx.y;
    size_type num = (nnz > start) * ceildiv(nnz - start, subwarp_size);
    num = min(num, num_lines);
    const IndexType ind_start = start + threadIdx.x;
    const IndexType ind_end = ind_start + (num - 1) * subwarp_size;
    IndexType ind = ind_start;
    bool is_first_in_segment = true;
    IndexType curr_row = (ind < nnz) ? row[ind] : 0;
    const auto tile_block =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    for (; ind < ind_end; ind += subwarp_size) {
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] * b_stride + column_id]
                                : zero<ValueType>();
        auto next_row =
            (ind + subwarp_size < nnz) ? row[ind + subwarp_size] : row[nnz - 1];
        // segmented scan
        if (tile_block.any(curr_row != next_row)) {
            is_first_in_segment = true;
            segment_scan<subwarp_size>(tile_block, curr_row, &temp_val,
                                       &is_first_in_segment);
            if (is_first_in_segment) {
                atomic_add(&(c[curr_row * c_stride + column_id]),
                           scale(temp_val));
            }
            temp_val = zero<ValueType>();
        }
        curr_row = next_row;
    }
    if (num > 0) {
        ind = ind_start + (num - 1) * subwarp_size;
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] * b_stride + column_id]
                                : zero<ValueType>();
        // segmented scan
        is_first_in_segment = true;
        segment_scan<subwarp_size>(tile_block, curr_row, &temp_val,
                                   &is_first_in_segment);
        if (is_first_in_segment) {
            atomic_add(&(c[curr_row * c_stride + column_id]), scale(temp_val));
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride)
{
    spmv_kernel(nnz, num_lines, val, col, row, b, b_stride, c, c_stride,
                [](const ValueType &x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col, const IndexType *__restrict__ row,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(
        nnz, num_lines, val, col, row, b, b_stride, c, c_stride,
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


template <typename ValueType>
ValueType calculate_nwarps(const size_type nnz, const ValueType nwarps_in_cuda)
{
    // TODO: multiple is a parameter should be tuned.
    ValueType multiple = 8;
    if (nnz >= 2000000) {
        multiple = 128;
    } else if (nnz >= 200000) {
        multiple = 32;
    }
    return std::min(static_cast<int64>(multiple * nwarps_in_cuda),
                    ceildiv(nnz, cuda_config::warp_size));
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Coo<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto nnz = c->get_num_stored_elements();
    const dim3 grid(ceildiv(nnz, default_block_size));
    const dim3 block(default_block_size);
    set_zero<<<grid, block>>>(nnz, as_cuda_type(c->get_values()));

    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Coo<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Coo<ValueType, IndexType> *a,
           const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto nnz = a->get_num_stored_elements();

    auto warps_per_sm = exec->get_num_cores_per_sm() / cuda_config::warp_size;
    auto nwarps =
        calculate_nwarps(nnz, exec->get_num_multiprocessor() * warps_per_sm);
    if (nwarps > 0) {
        int num_lines = ceildiv(nnz, nwarps * cuda_config::warp_size);
        const dim3 coo_block(cuda_config::warp_size, warps_in_block, 1);
        const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b->get_size()[1]);
        abstract_spmv<<<coo_grid, coo_block>>>(
            nnz, num_lines, as_cuda_type(a->get_const_values()),
            a->get_const_col_idxs(), as_cuda_type(a->get_const_row_idxs()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Dense<ValueType> *alpha,
                    const matrix::Coo<ValueType, IndexType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c)
{
    auto nnz = a->get_num_stored_elements();

    auto warps_per_sm = exec->get_num_cores_per_sm() / cuda_config::warp_size;
    auto nwarps =
        calculate_nwarps(nnz, exec->get_num_multiprocessor() * warps_per_sm);
    if (nwarps > 0) {
        int num_lines = ceildiv(nnz, nwarps * cuda_config::warp_size);
        const dim3 coo_block(cuda_config::warp_size, warps_in_block, 1);
        const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b->get_size()[1]);
        abstract_spmv<<<coo_grid, coo_block>>>(
            nnz, num_lines, as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            as_cuda_type(a->get_const_row_idxs()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const CudaExecutor> exec,
                              const IndexType *idxs, size_type num_nonzeros,
                              IndexType *ptrs,
                              size_type length) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_ROW_IDXS_TO_PTRS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const CudaExecutor> exec,
               matrix::Coo<ValueType, IndexType> *trans,
               const matrix::Coo<ValueType, IndexType> *orig) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Coo<ValueType, IndexType> *trans,
                    const matrix::Coo<ValueType, IndexType> *orig)
    NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONJ_TRANSPOSE_KERNEL);

namespace kernel {

template <typename ValueType>
__global__
    __launch_bounds__(cuda_config::max_block_size) void initialize_zero_dense(
        size_type num_rows, size_type num_cols, size_type stride,
        ValueType *__restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_rows && tidx_y < num_cols) {
        result[tidx_x * stride + tidx_y] = zero<ValueType>();
    }
}

template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type nnz, const IndexType *__restrict__ row_idxs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, size_type stride,
    ValueType *__restrict__ result)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < nnz) {
        result[stride * row_idxs[tidx] + col_idxs[tidx]] = values[tidx];
    }
}

}  // namespace kernel


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      matrix::Dense<ValueType> *result,
                      const matrix::Coo<ValueType, IndexType> *source)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();

    const auto nnz = source->get_num_stored_elements();

    const dim3 block_size(cuda_config::warp_size,
                          cuda_config::max_block_size / cuda_config::warp_size,
                          1);
    const dim3 init_grid_dim(ceildiv(num_rows, block_size.x),
                             ceildiv(num_cols, block_size.y), 1);
    kernel::initialize_zero_dense<<<init_grid_dim, block_size>>>(
        num_rows, num_cols, stride, as_cuda_type(result->get_values()));

    const auto grid_dim = ceildiv(nnz, default_block_size);
    kernel::fill_in_dense<<<grid_dim, default_block_size>>>(
        nnz, as_cuda_type(source->get_const_row_idxs()),
        as_cuda_type(source->get_const_col_idxs()),
        as_cuda_type(source->get_const_values()), stride,
        as_cuda_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_DENSE_KERNEL);


}  // namespace coo
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
