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

#include "core/preconditioner/batch_ilu_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_ilu.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace batch_ilu {


#include "reference/preconditioner/batch_ilu_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void compute_ilu0_factorization(
    std::shared_ptr<const DefaultExecutor> exec,
    const IndexType* const diag_locs,
    matrix::BatchCsr<ValueType, IndexType>* const mat_fact)
{
    const auto mat_factorized_batch = host::get_batch_struct(mat_fact);

#pragma omp parallel for
    for (size_type batch_id = 0; batch_id < mat_fact->get_num_batch_entries();
         ++batch_id) {
        const auto mat_factorized_entry =
            gko::batch::batch_entry(mat_factorized_batch, batch_id);

        batch_entry_factorize_impl(diag_locs, mat_factorized_entry);
    }
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
    const auto sys_mat_batch = host::get_batch_struct(sys_mat);
    const auto mat_factorized_batch = host::get_batch_struct(mat_fact);

#pragma omp parallel for
    for (size_type batch_id = 0; batch_id < mat_fact->get_num_batch_entries();
         ++batch_id) {
        const auto sys_mat_entry =
            gko::batch::batch_entry(sys_mat_batch, batch_id);
        const auto mat_factorized_entry =
            gko::batch::batch_entry(mat_factorized_batch, batch_id);

        batch_entry_parilu0_factorize_impl(parilu_num_sweeps, dependencies,
                                           nz_ptrs, sys_mat_entry,
                                           mat_factorized_entry);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PARILU_COMPUTE_FACTORIZATION_KERNEL);


template <typename ValueType, typename IndexType>
void apply_ilu(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const factored_matrix,
    const IndexType* const diag_locs,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const batch_csr::UniformBatch<const ValueType> factored_mat_batch =
        gko::kernels::host::get_batch_struct(factored_matrix);
    const auto rub = gko::kernels::host::get_batch_struct(r);
    const auto zub = gko::kernels::host::get_batch_struct(z);

    using prec_type = gko::kernels::host::batch_ilu<ValueType>;
    prec_type prec(factored_mat_batch, diag_locs);

#pragma omp parallel for firstprivate(prec)
    for (size_type batch_id = 0;
         batch_id < factored_matrix->get_num_batch_entries(); batch_id++) {
        const auto work_arr_size = prec_type::dynamic_work_size(
            factored_mat_batch.num_rows, factored_mat_batch.num_nnz);

        std::vector<ValueType> work(work_arr_size);

        const auto r_b = gko::batch::batch_entry(rub, batch_id);
        const auto z_b = gko::batch::batch_entry(zub, batch_id);
        prec.generate(batch_id, gko::batch_csr::BatchEntry<const ValueType>(),
                      work.data());
        prec.apply(r_b, z_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_APPLY_KERNEL);

}  // namespace batch_ilu
}  // namespace omp
}  // namespace kernels
}  // namespace gko
