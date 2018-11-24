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

#include "core/matrix/ell_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace ell {


constexpr int default_block_size = 512;


namespace {


template <int subwarp_size, typename ValueType, typename IndexType,
          typename Closure>
__device__ void spmv_kernel(const size_type num_rows,
                            const ValueType *__restrict__ val,
                            const IndexType *__restrict__ col,
                            const size_type stride,
                            const size_type num_stored_elements_per_row,
                            const ValueType *__restrict__ b,
                            const size_type b_stride, ValueType *__restrict__ c,
                            const size_type c_stride, Closure op)
{
    const auto tidx =
        static_cast<IndexType>(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto x = tidx / subwarp_size;
    const auto y = tidx % subwarp_size;

    if (x < num_rows) {
        const auto tile_block =
            group::tiled_partition<subwarp_size>(group::this_thread_block());
        ValueType temp = zero<ValueType>();
        const auto column_id = blockIdx.y;
        for (IndexType idx = y; idx < num_stored_elements_per_row;
             idx += subwarp_size) {
            const auto ind = x + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx < idx) {
                break;
            } else {
                temp += val[ind] * b[col_idx * b_stride + column_id];
            }
        }
        const auto answer = reduce(
            tile_block, temp, [](ValueType x, ValueType y) { return x + y; });
        if (tile_block.thread_rank() == 0) {
            c[x * c_stride + column_id] =
                op(answer, c[x * c_stride + column_id]);
        }
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void abstract_spmv(
    const size_type num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col, const size_type stride,
    const size_type num_stored_elements_per_row,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    spmv_kernel<subwarp_size>(
        num_rows, val, col, stride, num_stored_elements_per_row, b, b_stride, c,
        c_stride, [](const ValueType &x, const ValueType &y) { return x; });
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void abstract_spmv(
    const size_type num_rows, const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    const ValueType *__restrict__ b, const size_type b_stride,
    const ValueType *__restrict__ beta, ValueType *__restrict__ c,
    const size_type c_stride)
{
    const ValueType alpha_val = alpha[0];
    const ValueType beta_val = beta[0];
    spmv_kernel<subwarp_size>(
        num_rows, val, col, stride, num_stored_elements_per_row, b, b_stride, c,
        c_stride,
        [&alpha_val, &beta_val](const ValueType &x, const ValueType &y) {
            return alpha_val * x + beta_val * y;
        });
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Ell<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    int subwarp_size = 1;
    const auto nrows = a->get_size()[0];
    const auto ell_ncols = a->get_num_stored_elements_per_row();

    if (static_cast<double>(ell_ncols) / nrows > 1e-2) {
        while (subwarp_size < 32 && (subwarp_size << 1) <= ell_ncols) {
            subwarp_size <<= 1;
        }
    }
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(nrows * subwarp_size, block_size.x),
                         b->get_size()[1], 1);
    switch (subwarp_size) {
    case 1:
        abstract_spmv<1><<<grid_size, block_size, 0, 0>>>(
            nrows, as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    case 2:
        abstract_spmv<2><<<grid_size, block_size, 0, 0>>>(
            nrows, as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    case 4:
        abstract_spmv<4><<<grid_size, block_size, 0, 0>>>(
            nrows, as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    case 8:
        abstract_spmv<8><<<grid_size, block_size, 0, 0>>>(
            nrows, as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    case 16:
        abstract_spmv<16><<<grid_size, block_size, 0, 0>>>(
            nrows, as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    case 32:
        abstract_spmv<32><<<grid_size, block_size, 0, 0>>>(
            nrows, as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Ell<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    int subwarp_size = 1;
    const auto nrows = a->get_size()[0];
    const auto ell_ncols = a->get_num_stored_elements_per_row();

    if (static_cast<double>(ell_ncols) / nrows > 1e-2) {
        while (subwarp_size < 32 && (subwarp_size << 1) <= ell_ncols) {
            subwarp_size <<= 1;
        }
    }
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(a->get_size()[0], block_size.x),
                         b->get_size()[1], 1);

    switch (subwarp_size) {
    case 1:
        abstract_spmv<1><<<grid_size, block_size, 0, 0>>>(
            a->get_size()[0], as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(beta->get_const_values()),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    case 2:
        abstract_spmv<2><<<grid_size, block_size, 0, 0>>>(
            a->get_size()[0], as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(beta->get_const_values()),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    case 4:
        abstract_spmv<4><<<grid_size, block_size, 0, 0>>>(
            a->get_size()[0], as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(beta->get_const_values()),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    case 8:
        abstract_spmv<8><<<grid_size, block_size, 0, 0>>>(
            a->get_size()[0], as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(beta->get_const_values()),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    case 16:
        abstract_spmv<16><<<grid_size, block_size, 0, 0>>>(
            a->get_size()[0], as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(beta->get_const_values()),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    case 32:
        abstract_spmv<32><<<grid_size, block_size, 0, 0>>>(
            a->get_size()[0], as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(beta->get_const_values()),
            as_cuda_type(c->get_values()), c->get_stride());
        break;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const CudaExecutor> exec, matrix::Dense<ValueType> *result,
    const matrix::Ell<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_DENSE_KERNEL);


}  // namespace ell
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
