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

#include "core/preconditioner/batch_par_ilu_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "hip/base/exception.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/load_store.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"

namespace gko {
namespace kernels {
namespace hip {
namespace batch_par_ilu {
namespace {


constexpr size_type default_block_size = 256;


#include "common/cuda_hip/preconditioner/batch_par_ilu.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_par_ilu_kernels.hpp.inc"


}  // namespace


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

    hipLaunchKernelGGL(generate_common_pattern_to_fill_L_and_U, num_blocks,
                       default_block_size, 0, 0, static_cast<int>(num_rows),
                       first_sys_mat->get_const_row_ptrs(),
                       first_sys_mat->get_const_col_idxs(), l_row_ptrs,
                       u_row_ptrs, l_col_holders, u_col_holders);

    GKO_HIP_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_GENERATE_COMMON_PATTERN_KERNEL);


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

    hipLaunchKernelGGL(fill_L_and_U, grid_fill_LU, default_block_size, 0, 0,
                       nbatch, num_rows, nnz, sys_mat->get_const_col_idxs(),
                       as_hip_type(sys_mat->get_const_values()), l_nnz,
                       l_factor->get_col_idxs(),
                       as_hip_type(l_factor->get_values()), l_col_holders,
                       u_nnz, u_factor->get_col_idxs(),
                       as_hip_type(u_factor->get_values()), u_col_holders);

    GKO_HIP_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_INITIALIZE_BATCH_L_AND_BATCH_U);

template <typename ValueType, typename IndexType>
void compute_par_ilu0(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    matrix::BatchCsr<ValueType, IndexType>* const l_factor,
    matrix::BatchCsr<ValueType, IndexType>* const u_factor,
    const int num_sweeps, const IndexType* const dependencies,
    const IndexType* const nz_ptrs)
{
    const auto num_rows = static_cast<int>(sys_mat->get_size().at(0)[0]);
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto nnz =
        static_cast<int>(sys_mat->get_num_stored_elements() / nbatch);
    const auto l_nnz =
        static_cast<int>(l_factor->get_num_stored_elements() / nbatch);
    const auto u_nnz =
        static_cast<int>(u_factor->get_num_stored_elements() / nbatch);

    const size_type dynamic_shared_mem_bytes =
        (l_nnz + u_nnz) * sizeof(ValueType);

    hipLaunchKernelGGL(compute_parilu0_kernel, nbatch, default_block_size,
                       dynamic_shared_mem_bytes, 0, nbatch, num_rows, nnz,
                       as_hip_type(sys_mat->get_const_values()), l_nnz,
                       as_hip_type(l_factor->get_values()), u_nnz,
                       as_hip_type(u_factor->get_values()), num_sweeps,
                       dependencies, nz_ptrs);

    GKO_HIP_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_COMPUTE_PARILU0_KERNEL);


template <typename ValueType, typename IndexType>
void apply_par_ilu0(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const l_factor,
    const matrix::BatchCsr<ValueType, IndexType>* const u_factor,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto num_rows = static_cast<int>(l_factor->get_size().at(0)[0]);
    const auto nbatch = l_factor->get_num_batch_entries();
    const auto l_batch = get_batch_struct(l_factor);
    const auto u_batch = get_batch_struct(u_factor);
    using d_value_type = cuda_type<ValueType>;
    using prec_type = batch_parilu0<d_value_type>;
    bool is_fallback_required = true;

    prec_type prec(l_batch, u_batch, is_fallback_required);

    hipLaunchKernelGGL(
        batch_parilu_apply, nbatch, default_block_size,
        prec_type::dynamic_work_size(num_rows, 0) * sizeof(ValueType), 0, prec,
        nbatch, num_rows, as_hip_type(r->get_const_values()),
        as_hip_type(z->get_values()));

    GKO_HIP_LAST_IF_ERROR_THROW;
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_APPLY_KERNEL);

}  // namespace batch_par_ilu
}  // namespace hip
}  // namespace kernels
}  // namespace gko
