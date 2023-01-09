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

#include "core/preconditioner/batch_ilu_isai_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "hip/base/exception.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/load_store.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"

namespace gko {
namespace kernels {
namespace hip {
namespace batch_ilu_isai {
namespace {


constexpr size_type default_block_size = 256;

#include "common/cuda_hip/preconditioner/batch_ilu_isai.hpp.inc"

}  // namespace


template <typename ValueType, typename IndexType>
void apply_ilu_isai(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    const matrix::BatchCsr<ValueType, IndexType>* const l,
    const matrix::BatchCsr<ValueType, IndexType>* const u,
    const matrix::BatchCsr<ValueType, IndexType>* const l_inv,
    const matrix::BatchCsr<ValueType, IndexType>* const u_inv,
    const matrix::BatchCsr<ValueType, IndexType>* const mult_invs,
    const matrix::BatchCsr<ValueType, IndexType>* const iter_mat_lower_solve,
    const matrix::BatchCsr<ValueType, IndexType>* const iter_mat_upper_solve,
    const preconditioner::batch_ilu_isai_apply apply_type,
    const int num_relaxation_steps,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto num_rows = static_cast<int>(sys_mat->get_size().at(0)[0]);
    const auto nbatch = sys_mat->get_num_batch_entries();

    const auto l_batch = get_batch_struct(l);
    const auto u_batch = get_batch_struct(u);
    const auto l_inv_batch = get_batch_struct(l_inv);
    const auto u_inv_batch = get_batch_struct(u_inv);
    const auto mult_batch = maybe_null_batch_struct(mult_invs);
    const auto iter_mat_lower_solve_batch =
        maybe_null_batch_struct(iter_mat_lower_solve);
    const auto iter_mat_upper_solve_batch =
        maybe_null_batch_struct(iter_mat_upper_solve);

    using d_value_type = hip_type<ValueType>;
    using prec_type = batch_ilu_isai<d_value_type>;
    prec_type prec(l_batch, u_batch, l_inv_batch, u_inv_batch, mult_batch,
                   iter_mat_lower_solve_batch, iter_mat_upper_solve_batch,
                   apply_type, num_relaxation_steps);

    hipLaunchKernelGGL(
        batch_ilu_isai_apply, nbatch, default_block_size,
        prec_type::dynamic_work_size(
            num_rows,
            static_cast<int>(sys_mat->get_num_stored_elements() / nbatch)) *
            sizeof(ValueType),
        0, prec, nbatch, num_rows, as_hip_type(r->get_const_values()),
        as_hip_type(z->get_values()));

    GKO_HIP_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_ISAI_APPLY_KERNEL);

}  // namespace batch_ilu_isai
}  // namespace hip
}  // namespace kernels
}  // namespace gko
