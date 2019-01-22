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

#include "core/matrix/ell_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace ell {


constexpr int default_block_size = 512;

// TODO: multiple and ratio are parameters should be tuned
constexpr int multiple = 4;
constexpr double ratio = 1e-2;


namespace {


template <int subwarp_size, bool atomic, typename ValueType, typename IndexType,
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
    const auto nwarps_per_row =
        gridDim.x * blockDim.x / num_rows / subwarp_size;
    const auto x = tidx / subwarp_size / nwarps_per_row;
    const auto warp_id = tidx / subwarp_size % nwarps_per_row;
    const auto y_start = tidx % subwarp_size +
                         num_stored_elements_per_row * warp_id / nwarps_per_row;
    const auto y_end =
        num_stored_elements_per_row * (warp_id + 1) / nwarps_per_row;
    if (x < num_rows) {
        const auto tile_block =
            group::tiled_partition<subwarp_size>(group::this_thread_block());
        ValueType temp = zero<ValueType>();
        const auto column_id = blockIdx.y;
        for (IndexType idx = y_start; idx < y_end; idx += subwarp_size) {
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
            if (atomic) {
                atomic_add(&(c[x * c_stride + column_id]),
                           op(answer, c[x * c_stride + column_id]));
            } else {
                c[x * c_stride + column_id] =
                    op(answer, c[x * c_stride + column_id]);
            }
        }
    }
}


template <int subwarp_size, bool atomic = false, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void abstract_spmv(
    const size_type num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col, const size_type stride,
    const size_type num_stored_elements_per_row,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    spmv_kernel<subwarp_size, atomic>(
        num_rows, val, col, stride, num_stored_elements_per_row, b, b_stride, c,
        c_stride, [](const ValueType &x, const ValueType &y) { return x; });
}


template <int subwarp_size, bool atomic = false, typename ValueType,
          typename IndexType>
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
    if (atomic) {
        spmv_kernel<subwarp_size, atomic>(
            num_rows, val, col, stride, num_stored_elements_per_row, b,
            b_stride, c, c_stride,
            [&alpha_val](const ValueType &x, const ValueType &y) {
                return alpha_val * x;
            });
    } else {
        spmv_kernel<subwarp_size, atomic>(
            num_rows, val, col, stride, num_stored_elements_per_row, b,
            b_stride, c, c_stride,
            [&alpha_val, &beta_val](const ValueType &x, const ValueType &y) {
                return alpha_val * x + beta_val * y;
            });
    }
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
    bool atomic = false;
    int nwarps_per_row = 1;
    const auto nwarps = exec->get_num_cores_per_sm() / cuda_config::warp_size *
                        exec->get_num_multiprocessor() * multiple;

    // Use multithreads to perform the reduction on each row when the matrix is
    // wide.
    // To make every thread have computation, so pick the value which is the
    // power of 2 less than 32 and is less than or equal to ell_ncols. If the
    // subwarp_size is 32 and allow more than one warps to work on the same row,
    // use atomic add to handle the warps write the value into the same
    // position. The #warps is decided according to the number of warps allowed
    // on GPU.
    if (static_cast<double>(ell_ncols) / nrows > ratio) {
        while (subwarp_size < 32 && (subwarp_size << 1) <= ell_ncols) {
            subwarp_size <<= 1;
        }
        if (subwarp_size == 32) {
            nwarps_per_row =
                std::min(ell_ncols / cuda_config::warp_size, nwarps / nrows);
            nwarps_per_row = std::max(nwarps_per_row, 1);
        }
        if (nwarps_per_row > 1) {
            atomic = true;
        }
    }
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(nrows * subwarp_size * nwarps_per_row, block_size.x),
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
        if (atomic) {
            ASSERT_NO_CUDA_ERRORS(
                cudaMemset(c->get_values(), 0,
                           c->get_num_stored_elements() * sizeof(ValueType)));
            abstract_spmv<32, true><<<grid_size, block_size, 0, 0>>>(
                nrows, as_cuda_type(a->get_const_values()),
                a->get_const_col_idxs(), a->get_stride(),
                a->get_num_stored_elements_per_row(),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        } else {
            abstract_spmv<32><<<grid_size, block_size, 0, 0>>>(
                nrows, as_cuda_type(a->get_const_values()),
                a->get_const_col_idxs(), a->get_stride(),
                a->get_num_stored_elements_per_row(),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        }

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
    bool atomic = false;
    int nwarps_per_row = 1;
    const auto nwarps = exec->get_num_cores_per_sm() / cuda_config::warp_size *
                        exec->get_num_multiprocessor() * multiple;
    if (static_cast<double>(ell_ncols) / nrows > ratio) {
        while (subwarp_size < 32 && (subwarp_size << 1) <= ell_ncols) {
            subwarp_size <<= 1;
        }
        if (subwarp_size == 32) {
            nwarps_per_row =
                std::min(ell_ncols / cuda_config::warp_size, nwarps / nrows);
            nwarps_per_row = std::max(nwarps_per_row, 1);
        }
        if (nwarps_per_row > 1) {
            atomic = true;
        }
    }
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(nrows * subwarp_size * nwarps_per_row, block_size.x),
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
        if (atomic) {
            dense::scale(exec, beta, c);
            abstract_spmv<32, true><<<grid_size, block_size, 0, 0>>>(
                a->get_size()[0], as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                a->get_stride(), a->get_num_stored_elements_per_row(),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(beta->get_const_values()),
                as_cuda_type(c->get_values()), c->get_stride());
        } else {
            abstract_spmv<32><<<grid_size, block_size, 0, 0>>>(
                a->get_size()[0], as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                a->get_stride(), a->get_num_stored_elements_per_row(),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(beta->get_const_values()),
                as_cuda_type(c->get_values()), c->get_stride());
        }
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
