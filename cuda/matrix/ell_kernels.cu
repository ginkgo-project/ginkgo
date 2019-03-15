/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, Karlsruhe Institute of Technology
Copyright (c) 2017-2019, Universitat Jaume I
Copyright (c) 2017-2019, University of Tennessee
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

#include "core/matrix/ell_kernels.hpp"


#include <array>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/zero_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {


namespace ell {


constexpr int default_block_size = 512;


// TODO: num_threads_per_core and ratio are parameters should be tuned
/**
 * num_threads_per_core is the oversubscribing parameter. There are
 * `num_threads_per_core` threads assigned to each physical core.
 */
constexpr int num_threads_per_core = 4;
/**
 * ratio is the parameter to decide when to use threads to do reduction on each
 * row. (#cols/#rows > ratio)
 */
constexpr double ratio = 1e-2;


/**
 * A compile-time list of sub-warp sizes for which the spmv kernels should be
 * compiled.
 * 0 is a special case where it uses a sub-warp size of 32 in
 * combination with atomic_adds.
 */
using compiled_kernels = syn::value_list<int, 0, 1, 2, 4, 8, 16, 32>;


namespace kernel {
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
__global__ __launch_bounds__(default_block_size) void spmv(
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
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    const ValueType *__restrict__ b, const size_type b_stride,
    const ValueType *__restrict__ beta, ValueType *__restrict__ c,
    const size_type c_stride)
{
    const ValueType alpha_val = alpha[0];
    const ValueType beta_val = beta[0];
    // Because the atomic operation changes the values of c during computation,
    // it can not do the right alpha * a * b + beta * c operation.
    // Thus, the cuda kernel only computes alpha * a * b when it uses atomic
    // operation.
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
}  // namespace kernel


namespace {


template <int info, typename ValueType, typename IndexType>
void abstract_spmv(syn::value_list<int, info>, int nwarps_per_row,
                   const matrix::Ell<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   matrix::Dense<ValueType> *c,
                   const matrix::Dense<ValueType> *alpha = nullptr,
                   const matrix::Dense<ValueType> *beta = nullptr)
{
    const auto nrows = a->get_size()[0];
    constexpr int subwarp_size = (info == 0) ? 32 : info;
    constexpr bool atomic = (info == 0);
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(nrows * subwarp_size * nwarps_per_row, block_size.x),
        b->get_size()[1], 1);
    if (alpha == nullptr && beta == nullptr) {
        kernel::spmv<subwarp_size, atomic><<<grid_size, block_size, 0, 0>>>(
            nrows, as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());
    } else if (alpha != nullptr && beta != nullptr) {
        kernel::spmv<subwarp_size, atomic><<<grid_size, block_size, 0, 0>>>(
            nrows, as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
            a->get_stride(), a->get_num_stored_elements_per_row(),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(beta->get_const_values()),
            as_cuda_type(c->get_values()), c->get_stride());
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_abstract_spmv, abstract_spmv);


template <typename ValueType, typename IndexType>
std::array<int, 3> compute_subwarp_size_and_atomicity(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Ell<ValueType, IndexType> *a)
{
    int subwarp_size = 1;
    int atomic = 0;
    int nwarps_per_row = 1;

    const auto nrows = a->get_size()[0];
    const auto ell_ncols = a->get_num_stored_elements_per_row();
    const auto nwarps = exec->get_num_cores_per_sm() / cuda_config::warp_size *
                        exec->get_num_multiprocessor() * num_threads_per_core;

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
            atomic = 1;
        }
    }
    return {subwarp_size, atomic, nwarps_per_row};
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Ell<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    const auto data = compute_subwarp_size_and_atomicity(exec, a);
    const int subwarp_size = std::get<0>(data);
    const int atomic = std::get<1>(data);
    const int nwarps_per_row = std::get<2>(data);

    /**
     * info is the parameter for selecting the cuda kernel.
     * for info == 0, it uses the kernel by 32 threads with atomic operation
     * for other value, it uses the kernel without atomic_add
     */
    const int info = (!atomic) * subwarp_size;
    if (atomic) {
        zero_array(c->get_num_stored_elements(), c->get_values());
    }
    select_abstract_spmv(
        compiled_kernels(),
        [&info](int compiled_info) { return info == compiled_info; },
        syn::value_list<int>(), syn::type_list<>(), nwarps_per_row, a, b, c);
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
    const auto data = compute_subwarp_size_and_atomicity(exec, a);
    const int subwarp_size = std::get<0>(data);
    const int atomic = std::get<1>(data);
    const int nwarps_per_row = std::get<2>(data);

    /**
     * info is the parameter for selecting the cuda kernel.
     * for info == 0, it uses the kernel by 32 threads with atomic operation
     * for other value, it uses the kernel without atomic_add
     */
    const int info = (!atomic) * subwarp_size;
    if (atomic) {
        dense::scale(exec, beta, c);
    }
    select_abstract_spmv(
        compiled_kernels(),
        [&info](int compiled_info) { return info == compiled_info; },
        syn::value_list<int>(), syn::type_list<>(), nwarps_per_row, a, b, c,
        alpha, beta);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


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
    size_type num_rows, size_type nnz, size_type source_stride,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, size_type result_stride,
    ValueType *__restrict__ result)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < num_rows) {
        for (auto col = 0; col < nnz; col++) {
            result[tidx * result_stride +
                   col_idxs[tidx + col * source_stride]] +=
                values[tidx + col * source_stride];
        }
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      matrix::Dense<ValueType> *result,
                      const matrix::Ell<ValueType, IndexType> *source)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto result_stride = result->get_stride();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();
    const auto source_stride = source->get_stride();

    const dim3 block_size(cuda_config::warp_size,
                          cuda_config::max_block_size / cuda_config::warp_size,
                          1);
    const dim3 init_grid_dim(ceildiv(result_stride, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);
    kernel::initialize_zero_dense<<<init_grid_dim, block_size>>>(
        num_rows, num_cols, result_stride, as_cuda_type(result->get_values()));

    const auto grid_dim = ceildiv(num_rows, default_block_size);
    kernel::fill_in_dense<<<grid_dim, default_block_size>>>(
        num_rows, source->get_num_stored_elements_per_row(), source_stride,
        as_cuda_type(col_idxs), as_cuda_type(vals), result_stride,
        as_cuda_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_DENSE_KERNEL);


}  // namespace ell
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
