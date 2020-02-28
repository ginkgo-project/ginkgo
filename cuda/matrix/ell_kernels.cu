/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/format_conversion.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/zero_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The ELL matrix format namespace.
 *
 * @ingroup ell
 */
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
 * max_thread_per_worker is the max number of thread per worker. The
 * `compiled_kernels` must be a list <0, 1, 2, ..., max_thread_per_worker>
 */
constexpr int max_thread_per_worker = 32;


/**
 * A compile-time list of sub-warp sizes for which the spmv kernels should be
 * compiled.
 * 0 is a special case where it uses a sub-warp size of warp_size in
 * combination with atomic_adds.
 */
using compiled_kernels = syn::value_list<int, 0, 1, 2, 4, 8, 16, 32>;


#include "common/matrix/ell_kernels.hpp.inc"


namespace {


template <int info, typename ValueType, typename IndexType>
void abstract_spmv(syn::value_list<int, info>, int num_worker_per_row,
                   const matrix::Ell<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   matrix::Dense<ValueType> *c,
                   const matrix::Dense<ValueType> *alpha = nullptr,
                   const matrix::Dense<ValueType> *beta = nullptr)
{
    const auto nrows = a->get_size()[0];
    constexpr int num_thread_per_worker =
        (info == 0) ? max_thread_per_worker : info;
    constexpr bool atomic = (info == 0);
    const dim3 block_size(default_block_size / num_thread_per_worker,
                          num_thread_per_worker, 1);
    const dim3 grid_size(ceildiv(nrows * num_worker_per_row, block_size.x),
                         b->get_size()[1], 1);
    if (alpha == nullptr && beta == nullptr) {
        kernel::spmv<num_thread_per_worker, atomic>
            <<<grid_size, block_size, 0, 0>>>(
                nrows, num_worker_per_row, as_cuda_type(a->get_const_values()),
                a->get_const_col_idxs(), a->get_stride(),
                a->get_num_stored_elements_per_row(),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
    } else if (alpha != nullptr && beta != nullptr) {
        kernel::spmv<num_thread_per_worker, atomic>
            <<<grid_size, block_size, 0, 0>>>(
                nrows, num_worker_per_row,
                as_cuda_type(alpha->get_const_values()),
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
std::array<int, 3> compute_thread_worker_and_atomicity(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Ell<ValueType, IndexType> *a)
{
    int num_thread_per_worker = 1;
    int atomic = 0;
    int num_worker_per_row = 1;

    const auto nrows = a->get_size()[0];
    const auto ell_ncols = a->get_num_stored_elements_per_row();
    // TODO: num_threads_per_core should be tuned for AMD gpu
    const auto nwarps = exec->get_num_warps_per_sm() *
                        exec->get_num_multiprocessor() * num_threads_per_core;

    // Use multithreads to perform the reduction on each row when the matrix is
    // wide.
    // To make every thread have computation, so pick the value which is the
    // power of 2 less than max_thread_per_worker and is less than or equal to
    // ell_ncols. If the num_thread_per_worker is max_thread_per_worker and
    // allow more than one worker to work on the same row, use atomic add to
    // handle the worker write the value into the same position. The #worker is
    // decided according to the number of worker allowed on GPU.
    if (static_cast<double>(ell_ncols) / nrows > ratio) {
        while (num_thread_per_worker < max_thread_per_worker &&
               (num_thread_per_worker << 1) <= ell_ncols) {
            num_thread_per_worker <<= 1;
        }
        if (num_thread_per_worker == max_thread_per_worker) {
            num_worker_per_row =
                std::min(ell_ncols / max_thread_per_worker, nwarps / nrows);
            num_worker_per_row = std::max(num_worker_per_row, 1);
        }
        if (num_worker_per_row > 1) {
            atomic = 1;
        }
    }
    return {num_thread_per_worker, atomic, num_worker_per_row};
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Ell<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    const auto data = compute_thread_worker_and_atomicity(exec, a);
    const int num_thread_per_worker = std::get<0>(data);
    const int atomic = std::get<1>(data);
    const int num_worker_per_row = std::get<2>(data);

    /**
     * info is the parameter for selecting the cuda kernel.
     * for info == 0, it uses the kernel by warp_size threads with atomic
     * operation for other value, it uses the kernel without atomic_add
     */
    const int info = (!atomic) * num_thread_per_worker;
    if (atomic) {
        zero_array(c->get_num_stored_elements(), c->get_values());
    }
    select_abstract_spmv(
        compiled_kernels(),
        [&info](int compiled_info) { return info == compiled_info; },
        syn::value_list<int>(), syn::type_list<>(), num_worker_per_row, a, b,
        c);
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
    const auto data = compute_thread_worker_and_atomicity(exec, a);
    const int num_thread_per_worker = std::get<0>(data);
    const int atomic = std::get<1>(data);
    const int num_worker_per_row = std::get<2>(data);

    /**
     * info is the parameter for selecting the cuda kernel.
     * for info == 0, it uses the kernel by warp_size threads with atomic
     * operation for other value, it uses the kernel without atomic_add
     */
    const int info = (!atomic) * num_thread_per_worker;
    if (atomic) {
        dense::scale(exec, beta, c);
    }
    select_abstract_spmv(
        compiled_kernels(),
        [&info](int compiled_info) { return info == compiled_info; },
        syn::value_list<int>(), syn::type_list<>(), num_worker_per_row, a, b, c,
        alpha, beta);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


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

    const dim3 block_size(config::warp_size,
                          config::max_block_size / config::warp_size, 1);
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


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Ell<ValueType, IndexType> *source)
{
    auto num_rows = result->get_size()[0];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    const auto stride = source->get_stride();
    const auto max_nnz_per_row = source->get_num_stored_elements_per_row();

    constexpr auto rows_per_block =
        ceildiv(default_block_size, config::warp_size);
    const auto grid_dim_nnz = ceildiv(source->get_size()[0], rows_per_block);

    kernel::count_nnz_per_row<<<grid_dim_nnz, default_block_size>>>(
        num_rows, max_nnz_per_row, stride,
        as_cuda_type(source->get_const_values()), as_cuda_type(row_ptrs));

    prefix_sum(exec, row_ptrs, num_rows + 1);

    size_type grid_dim = ceildiv(num_rows, default_block_size);

    kernel::fill_in_csr<<<grid_dim, default_block_size>>>(
        num_rows, max_nnz_per_row, stride,
        as_cuda_type(source->get_const_values()),
        as_cuda_type(source->get_const_col_idxs()), as_cuda_type(row_ptrs),
        as_cuda_type(col_idxs), as_cuda_type(values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Ell<ValueType, IndexType> *source,
                    size_type *result)
{
    const auto num_rows = source->get_size()[0];
    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    *result = reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_COUNT_NONZEROS_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(std::shared_ptr<const CudaExecutor> exec,
                                const matrix::Ell<ValueType, IndexType> *source,
                                Array<size_type> *result)
{
    const auto num_rows = source->get_size()[0];
    const auto max_nnz_per_row = source->get_num_stored_elements_per_row();
    const auto stride = source->get_stride();
    const auto values = source->get_const_values();

    const auto warp_size = config::warp_size;
    const auto grid_dim = ceildiv(num_rows * warp_size, default_block_size);

    kernel::count_nnz_per_row<<<grid_dim, default_block_size>>>(
        num_rows, max_nnz_per_row, stride, as_cuda_type(values),
        as_cuda_type(result->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CALCULATE_NONZEROS_PER_ROW_KERNEL);


}  // namespace ell
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
