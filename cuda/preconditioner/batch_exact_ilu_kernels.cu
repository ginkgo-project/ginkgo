/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/preconditioner/batch_exact_ilu_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_exact_ilu {
namespace {


constexpr size_type default_block_size = 256;


#include "common/cuda_hip/preconditioner/batch_exact_ilu_kernels.hpp.inc"

}  // namespace


template <typename ValueType, typename IndexType>
void compute_factorization(
    std::shared_ptr<const DefaultExecutor> exec,
    const IndexType* const diag_locs,
    matrix::BatchCsr<ValueType, IndexType>* const mat_fact)
{
    const auto num_rows = static_cast<int>(mat_fact->get_size().at(0)[0]);
    const auto nbatch = mat_fact->get_num_batch_entries();
    const auto nnz =
        static_cast<int>(mat_fact->get_num_stored_elements() / nbatch);

    const int dynamic_shared_mem_bytes = 2 * num_rows * sizeof(ValueType);

    generate_exact_ilu0_kernel<<<nbatch, default_block_size,
                                 dynamic_shared_mem_bytes>>>(
        nbatch, num_rows, nnz, diag_locs, mat_fact->get_const_row_ptrs(),
        mat_fact->get_const_col_idxs(), as_cuda_type(mat_fact->get_values()));

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_EXACT_ILU_COMPUTE_FACTORIZATION_KERNEL);


template <typename ValueType, typename IndexType>
void apply_exact_ilu(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const factored_matrix,
    const IndexType* const diag_locs,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_EXACT_ILU_APPLY_KERNEL);

}  // namespace batch_exact_ilu
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
