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

#include "core/matrix/sellp_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The SELL-P matrix format namespace.
 *
 * @ingroup sellp
 */
namespace sellp {


constexpr auto default_block_size = 512;


#include "common/matrix/sellp_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const HipExecutor> exec,
          const matrix::Sellp<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    const dim3 blockSize(matrix::default_slice_size);
    const dim3 gridSize(ceildiv(a->get_size()[0], matrix::default_slice_size),
                        b->get_size()[1]);

    hipLaunchKernelGGL(
        spmv_kernel, dim3(gridSize), dim3(blockSize), 0, 0, a->get_size()[0],
        b->get_size()[1], b->get_stride(), c->get_stride(),
        a->get_const_slice_lengths(), a->get_const_slice_sets(),
        as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
        as_hip_type(b->get_const_values()), as_hip_type(c->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const HipExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Sellp<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    const dim3 blockSize(matrix::default_slice_size);
    const dim3 gridSize(ceildiv(a->get_size()[0], matrix::default_slice_size),
                        b->get_size()[1]);

    hipLaunchKernelGGL(
        advanced_spmv_kernel, dim3(gridSize), dim3(blockSize), 0, 0,
        a->get_size()[0], b->get_size()[1], b->get_stride(), c->get_stride(),
        a->get_const_slice_lengths(), a->get_const_slice_sets(),
        as_hip_type(alpha->get_const_values()),
        as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
        as_hip_type(b->get_const_values()),
        as_hip_type(beta->get_const_values()), as_hip_type(c->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const HipExecutor> exec,
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

    const dim3 block_size(config::warp_size,
                          config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(result->get_stride(), block_size.x),
                             ceildiv(num_rows, block_size.y), 1);

    hipLaunchKernelGGL(kernel::initialize_zero_dense, dim3(init_grid_dim),
                       dim3(block_size), 0, 0, num_rows, num_cols,
                       result->get_stride(), as_hip_type(result->get_values()));

    constexpr auto threads_per_row = config::warp_size;
    const auto grid_dim =
        ceildiv(slice_size * slice_num * threads_per_row, default_block_size);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::fill_in_dense<threads_per_row>),
                       dim3(grid_dim), dim3(default_block_size), 0, 0, num_rows,
                       num_cols, result->get_stride(), slice_size,
                       as_hip_type(slice_lengths), as_hip_type(slice_sets),
                       as_hip_type(col_idxs), as_hip_type(vals),
                       as_hip_type(result->get_values()));
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const HipExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Sellp<ValueType, IndexType> *source)
{
    const auto num_rows = source->get_size()[0];
    const auto slice_size = source->get_slice_size();
    const auto slice_num = ceildiv(num_rows, slice_size);

    const auto source_values = source->get_const_values();
    const auto source_slice_lengths = source->get_const_slice_lengths();
    const auto source_slice_sets = source->get_const_slice_sets();
    const auto source_col_idxs = source->get_const_col_idxs();

    auto result_values = result->get_values();
    auto result_col_idxs = result->get_col_idxs();
    auto result_row_ptrs = result->get_row_ptrs();

    auto grid_dim = ceildiv(num_rows * config::warp_size, default_block_size);

    hipLaunchKernelGGL(
        kernel::count_nnz_per_row, dim3(grid_dim), dim3(default_block_size), 0,
        0, num_rows, slice_size, as_hip_type(source_slice_sets),
        as_hip_type(source_values), as_hip_type(result_row_ptrs));

    prefix_sum(exec, result_row_ptrs, num_rows + 1);

    grid_dim = ceildiv(num_rows, default_block_size);

    hipLaunchKernelGGL(
        kernel::fill_in_csr, dim3(grid_dim), dim3(default_block_size), 0, 0,
        num_rows, slice_size, as_hip_type(source_slice_sets),
        as_hip_type(source_col_idxs), as_hip_type(source_values),
        as_hip_type(result_row_ptrs), as_hip_type(result_col_idxs),
        as_hip_type(result_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Sellp<ValueType, IndexType> *source,
                    size_type *result)
{
    const auto num_rows = source->get_size()[0];
    const auto slice_size = source->get_slice_size();
    const auto slice_sets = source->get_const_slice_sets();
    const auto values = source->get_const_values();

    auto nnz_per_row = Array<size_type>(exec, num_rows);

    auto grid_dim = ceildiv(num_rows * config::warp_size, default_block_size);

    hipLaunchKernelGGL(kernel::count_nnz_per_row, dim3(grid_dim),
                       dim3(default_block_size), 0, 0, num_rows, slice_size,
                       as_hip_type(slice_sets), as_hip_type(values),
                       as_hip_type(nnz_per_row.get_data()));

    *result = reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_COUNT_NONZEROS_KERNEL);


}  // namespace sellp
}  // namespace hip
}  // namespace kernels
}  // namespace gko
