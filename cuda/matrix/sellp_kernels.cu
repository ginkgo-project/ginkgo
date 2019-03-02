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

#include "core/matrix/sellp_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace sellp {


namespace {

constexpr auto default_block_size = 512;

template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(matrix::default_slice_size) void spmv_kernel(
    size_type num_rows, size_type num_right_hand_sides, size_type b_stride,
    size_type c_stride, const size_type *__restrict__ slice_lengths,
    const size_type *__restrict__ slice_sets, const ValueType *__restrict__ a,
    const IndexType *__restrict__ col, const ValueType *__restrict__ b,
    ValueType *__restrict__ c)
{
    const auto slice_id = blockIdx.x;
    const auto slice_size = blockDim.x;
    const auto row_in_slice = threadIdx.x;
    const auto global_row =
        static_cast<size_type>(slice_size) * slice_id + row_in_slice;
    const auto column_id = blockIdx.y;
    ValueType val = 0;
    IndexType ind = 0;
    if (global_row < num_rows && column_id < num_right_hand_sides) {
        for (size_type i = 0; i < slice_lengths[slice_id]; i++) {
            ind = row_in_slice + (slice_sets[slice_id] + i) * slice_size;
            val += a[ind] * b[col[ind] * b_stride + column_id];
        }
        c[global_row * c_stride + column_id] = val;
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Sellp<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    const dim3 blockSize(matrix::default_slice_size);
    const dim3 gridSize(ceildiv(a->get_size()[0], matrix::default_slice_size),
                        b->get_size()[1]);

    spmv_kernel<<<gridSize, blockSize>>>(
        a->get_size()[0], b->get_size()[1], b->get_stride(), c->get_stride(),
        a->get_const_slice_lengths(), a->get_const_slice_sets(),
        as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
        as_cuda_type(b->get_const_values()), as_cuda_type(c->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);


namespace {


template <typename ValueType, typename IndexType>
__global__
    __launch_bounds__(matrix::default_slice_size) void advanced_spmv_kernel(
        size_type num_rows, size_type num_right_hand_sides, size_type b_stride,
        size_type c_stride, const size_type *__restrict__ slice_lengths,
        const size_type *__restrict__ slice_sets,
        const ValueType *__restrict__ alpha, const ValueType *__restrict__ a,
        const IndexType *__restrict__ col, const ValueType *__restrict__ b,
        const ValueType *__restrict__ beta, ValueType *__restrict__ c)
{
    const auto slice_id = blockIdx.x;
    const auto slice_size = blockDim.x;
    const auto row_in_slice = threadIdx.x;
    const auto global_row =
        static_cast<size_type>(slice_size) * slice_id + row_in_slice;
    const auto column_id = blockIdx.y;
    ValueType val = 0;
    IndexType ind = 0;
    if (global_row < num_rows && column_id < num_right_hand_sides) {
        for (size_type i = 0; i < slice_lengths[slice_id]; i++) {
            ind = row_in_slice + (slice_sets[slice_id] + i) * slice_size;
            val += alpha[0] * a[ind] * b[col[ind] * b_stride + column_id];
        }
        c[global_row * c_stride + column_id] =
            beta[0] * c[global_row * c_stride + column_id] + val;
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Sellp<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    const dim3 blockSize(matrix::default_slice_size);
    const dim3 gridSize(ceildiv(a->get_size()[0], matrix::default_slice_size),
                        b->get_size()[1]);

    advanced_spmv_kernel<<<gridSize, blockSize>>>(
        a->get_size()[0], b->get_size()[1], b->get_stride(), c->get_stride(),
        a->get_const_slice_lengths(), a->get_const_slice_sets(),
        as_cuda_type(alpha->get_const_values()),
        as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
        as_cuda_type(b->get_const_values()),
        as_cuda_type(beta->get_const_values()), as_cuda_type(c->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


namespace kernel {


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_zero_dense(
    size_type num_rows, size_type num_cols, size_type stride,
    ValueType *__restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}


template <unsigned int threads_per_row, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type slice_size, const size_type *__restrict__ slice_lengths,
    const size_type *__restrict__ slice_sets,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, ValueType *__restrict__ result)
{
    const auto global_row =
        (blockDim.x * blockIdx.x + threadIdx.x) / threads_per_row;
    const auto row = global_row % slice_size;
    const auto slice = global_row / slice_size;
    const auto start_index = threadIdx.x % threads_per_row;

    if (global_row < num_rows) {
        for (auto i = start_index; i < slice_lengths[slice];
             i += threads_per_row) {
            if (values[(slice_sets[slice] + i) * slice_size + row] !=
                zero<ValueType>()) {
                result[global_row * stride +
                       col_idxs[(slice_sets[slice] + i) * slice_size + row]] =
                    values[(slice_sets[slice] + i) * slice_size + row];
            }
        }
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      matrix::Dense<ValueType> *result,
                      const matrix::Sellp<ValueType, IndexType> *source)
{
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    const auto vals = source->get_const_values();
    const auto col_idxs = source->get_const_col_idxs();
    const auto slice_lengths = source->get_const_slice_lengths();
    const auto slice_sets = source->get_const_slice_sets();
    const auto slice_size = source->get_slice_size();

    const auto slice_num = ceildiv(num_rows, slice_size);

    const dim3 block_size(cuda_config::warp_size,
                          cuda_config::max_block_size / cuda_config::warp_size,
                          1);
    const dim3 init_grid_dim(ceildiv(result->get_stride(), block_size.x),
                             ceildiv(num_rows, block_size.y), 1);

    kernel::initialize_zero_dense<<<init_grid_dim, block_size>>>(
        num_rows, num_cols, result->get_stride(),
        as_cuda_type(result->get_values()));

    constexpr auto threads_per_row = cuda_config::warp_size;
    const auto grid_dim =
        ceildiv(slice_size * slice_num * threads_per_row, default_block_size);

    kernel::fill_in_dense<threads_per_row><<<grid_dim, default_block_size>>>(
        num_rows, num_cols, result->get_stride(), slice_size,
        as_cuda_type(slice_lengths), as_cuda_type(slice_sets),
        as_cuda_type(col_idxs), as_cuda_type(vals),
        as_cuda_type(result->get_values()));
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_CONVERT_TO_DENSE_KERNEL);


}  // namespace sellp
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
