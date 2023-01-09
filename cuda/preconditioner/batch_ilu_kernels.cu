/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/preconditioner/batch_ilu_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/load_store.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_ilu {
namespace {


constexpr size_type default_block_size = 256;

#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_ilu.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_ilu_kernels.hpp.inc"

}  // namespace


template <typename ValueType, typename IndexType>
void compute_ilu0_factorization(
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
void compute_parilu0_factorization(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    matrix::BatchCsr<ValueType, IndexType>* const mat_fact,
    const int parilu_num_sweeps, const IndexType* const dependencies,
    const IndexType* const nz_ptrs)
{
    const auto num_rows = static_cast<int>(sys_mat->get_size().at(0)[0]);
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto nnz =
        static_cast<int>(sys_mat->get_num_stored_elements() / nbatch);

    const int dynamic_shared_mem_bytes = nnz * sizeof(ValueType);

    generate_parilu0_kernel<<<nbatch, default_block_size,
                              dynamic_shared_mem_bytes>>>(
        nbatch, num_rows, nnz, dependencies, nz_ptrs, parilu_num_sweeps,
        as_cuda_type(sys_mat->get_const_values()),
        as_cuda_type(mat_fact->get_values()));

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PARILU_COMPUTE_FACTORIZATION_KERNEL);


// Only for testing purpose
template <typename ValueType, typename IndexType>
void apply_ilu(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_matrix,
    const matrix::BatchCsr<ValueType, IndexType>* const factored_matrix,
    const IndexType* const diag_locs,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto num_rows =
        static_cast<int>(factored_matrix->get_size().at(0)[0]);
    const auto nbatch = factored_matrix->get_num_batch_entries();
    const auto factored_matrix_batch = get_batch_struct(factored_matrix);
    using d_value_type = cuda_type<ValueType>;
    using prec_type = batch_ilu<d_value_type>;
    prec_type prec(factored_matrix_batch, diag_locs);

    batch_ilu_apply<<<nbatch, default_block_size,
                      prec_type::dynamic_work_size(
                          num_rows,
                          static_cast<int>(
                              sys_matrix->get_num_stored_elements() / nbatch)) *
                          sizeof(ValueType)>>>(
        prec, nbatch, num_rows, as_cuda_type(r->get_const_values()),
        as_cuda_type(z->get_values()));
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void generate_common_pattern_to_fill_l_and_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_mat,
    const IndexType* const l_row_ptrs, const IndexType* const u_row_ptrs,
    IndexType* const l_col_holders, IndexType* const u_col_holders)
{
    const size_type num_rows = first_sys_mat->get_size()[0];
    const size_type num_warps = num_rows;
    const size_type num_blocks =
        ceildiv(num_warps, ceildiv(default_block_size, config::warp_size));

    generate_common_pattern_to_fill_L_and_U<<<num_blocks, default_block_size>>>(
        static_cast<int>(num_rows), first_sys_mat->get_const_row_ptrs(),
        first_sys_mat->get_const_col_idxs(), l_row_ptrs, u_row_ptrs,
        l_col_holders, u_col_holders);

    GKO_CUDA_LAST_IF_ERROR_THROW;
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_GENERATE_COMMON_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_batch_l_and_batch_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    matrix::BatchCsr<ValueType, IndexType>* const l_factor,
    matrix::BatchCsr<ValueType, IndexType>* const u_factor,
    const IndexType* const l_col_holders, const IndexType* const u_col_holders)
{
    const auto num_rows = static_cast<int>(sys_mat->get_size().at(0)[0]);
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto nnz =
        static_cast<int>(sys_mat->get_num_stored_elements() / nbatch);
    const auto l_nnz =
        static_cast<int>(l_factor->get_num_stored_elements() / nbatch);
    const auto u_nnz =
        static_cast<int>(u_factor->get_num_stored_elements() / nbatch);
    const int greater_nnz = l_nnz > u_nnz ? l_nnz : u_nnz;
    const size_type grid_fill_LU =
        ceildiv(greater_nnz * nbatch, default_block_size);

    fill_L_and_U<<<grid_fill_LU, default_block_size>>>(
        nbatch, num_rows, nnz, sys_mat->get_const_col_idxs(),
        as_cuda_type(sys_mat->get_const_values()), l_nnz,
        l_factor->get_col_idxs(), as_cuda_type(l_factor->get_values()),
        l_col_holders, u_nnz, u_factor->get_col_idxs(),
        as_cuda_type(u_factor->get_values()), u_col_holders);

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_INITIALIZE_BATCH_L_AND_BATCH_U);

}  // namespace batch_ilu
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
