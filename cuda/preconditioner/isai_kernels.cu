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

#include "core/preconditioner/isai_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/matrix/csr_builder.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/merging.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/components/zero_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Isai preconditioner namespace.
 * @ref Isai
 * @ingroup isai
 */
namespace isai {


constexpr int subwarp_size{config::warp_size};
constexpr int subwarps_per_block{2};
constexpr int default_block_size{subwarps_per_block * subwarp_size};


#include "common/preconditioner/isai_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void generate_tri_inverse(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Csr<ValueType, IndexType> *mtx,
                          matrix::Csr<ValueType, IndexType> *inverse_mtx,
                          IndexType *excess_block_ptrs,
                          IndexType *excess_row_ptrs_full, bool lower)
{
    const auto num_rows = mtx->get_size()[0];
    const auto nnz = inverse_mtx->get_num_stored_elements();

    if (excess_block_ptrs) {
        zero_array(num_rows + 1, excess_block_ptrs);
    }
    if (excess_row_ptrs_full) {
        zero_array(nnz + 1, excess_row_ptrs_full);
    }

    const dim3 block(default_block_size, 1, 1);
    const dim3 grid(ceildiv(num_rows, block.x / config::warp_size), 1, 1);
    if (lower) {
        kernel::generate_l_inverse<subwarp_size, subwarps_per_block>
            <<<grid, block>>>(
                static_cast<IndexType>(num_rows), mtx->get_const_row_ptrs(),
                mtx->get_const_col_idxs(),
                as_cuda_type(mtx->get_const_values()),
                inverse_mtx->get_row_ptrs(), inverse_mtx->get_col_idxs(),
                as_cuda_type(inverse_mtx->get_values()));
    } else {
        kernel::generate_u_inverse<subwarp_size, subwarps_per_block>
            <<<grid, block>>>(
                static_cast<IndexType>(num_rows), mtx->get_const_row_ptrs(),
                mtx->get_const_col_idxs(),
                as_cuda_type(mtx->get_const_values()),
                inverse_mtx->get_row_ptrs(), inverse_mtx->get_col_idxs(),
                as_cuda_type(inverse_mtx->get_values()));
    }
    // Call make_srow()
    matrix::CsrBuilder<ValueType, IndexType> builder(inverse_mtx);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_TRI_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
void generate_excess_system(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *input,
    const matrix::Csr<ValueType, IndexType> *inverse,
    const IndexType *excess_block_ptrs, const IndexType *excess_row_ptrs_full,
    matrix::Csr<ValueType, IndexType> *excess_system,
    matrix::Dense<ValueType> *excess_rhs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_EXCESS_SYSTEM_KERNEL);


template <typename ValueType, typename IndexType>
void scatter_excess_solution(std::shared_ptr<const DefaultExecutor> exec,
                             const IndexType *excess_block_ptrs,
                             const matrix::Dense<ValueType> *excess_solution,
                             matrix::Csr<ValueType, IndexType> *inverse)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_SCATTER_EXCESS_SOLUTION_KERNEL);


}  // namespace isai
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
