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


#include <algorithm>
#include <cassert>


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_par_ilu.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace batch_par_ilu {


#include "reference/preconditioner/batch_par_ilu_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void generate_common_pattern_to_fill_l_and_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_mat,
    const IndexType* const l_row_ptrs, const IndexType* const u_row_ptrs,
    IndexType* const l_col_holders, IndexType* const u_col_holders)
{
    const int nrows = static_cast<int>(first_sys_mat->get_size()[0]);
    const IndexType* row_ptrs = first_sys_mat->get_const_row_ptrs();
    const IndexType* col_idxs = first_sys_mat->get_const_col_idxs();

    for (int row_index = 0; row_index < nrows; row_index++) {
        const int L_row_start = l_row_ptrs[row_index];
        const int U_row_start = u_row_ptrs[row_index];
        const int row_start = row_ptrs[row_index];
        const int row_end = row_ptrs[row_index + 1];

        // const int diag_ele_loc = diag_ptrs[row_index];

        const int nnz_per_row_L =
            l_row_ptrs[row_index + 1] - l_row_ptrs[row_index];
        const int diag_ele_loc = row_start + nnz_per_row_L - 1;

        for (int i = row_start; i < row_end; i++) {
            if (i < diag_ele_loc)  // or col_idxs[i] < row_index
            {
                const int corresponding_l_index = L_row_start + (i - row_start);
                l_col_holders[corresponding_l_index] = i;
            } else {
                if (i == diag_ele_loc)  // or col_idxs[i] == row_index
                {
                    const int corresponding_l_index =
                        L_row_start + (i - row_start);
                    l_col_holders[corresponding_l_index] = (-1 * row_index) - 1;
                }

                const int corresponding_u_index =
                    U_row_start + (i - diag_ele_loc);
                u_col_holders[corresponding_u_index] = i;
            }
        }
    }
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
    const size_type nbatch = sys_mat->get_num_batch_entries();
    const int nnz =
        static_cast<int>(sys_mat->get_num_stored_elements() / nbatch);
    const int l_nnz =
        static_cast<int>(l_factor->get_num_stored_elements() / nbatch);
    const int u_nnz =
        static_cast<int>(u_factor->get_num_stored_elements() / nbatch);

    const IndexType* col_idxs = sys_mat->get_const_col_idxs();
    const ValueType* vals = sys_mat->get_const_values();
    IndexType* l_col_idxs = l_factor->get_col_idxs();
    ValueType* l_vals = l_factor->get_values();
    IndexType* u_col_idxs = u_factor->get_col_idxs();
    ValueType* u_vals = u_factor->get_values();

    for (size_type batch_id = 0; batch_id < nbatch; batch_id++) {
        initialize_batch_l_and_batch_u_individual_entry_impl(
            batch_id, nnz, col_idxs, vals, l_nnz, l_col_holders, l_col_idxs,
            l_vals, u_nnz, u_col_holders, u_col_idxs, u_vals);
    }
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
    const int num_rows = static_cast<int>(sys_mat->get_size().at(0)[0]);
    const size_type nbatch = sys_mat->get_num_batch_entries();
    const int nnz =
        static_cast<int>(sys_mat->get_num_stored_elements() / nbatch);
    const int l_nnz =
        static_cast<int>(l_factor->get_num_stored_elements() / nbatch);
    const int u_nnz =
        static_cast<int>(u_factor->get_num_stored_elements() / nbatch);

    for (int batch_id = 0; batch_id < nbatch; batch_id++) {
        compute_parilu0_batch_entry_impl(
            batch_id, num_rows, nnz, sys_mat->get_const_values(), l_nnz,
            l_factor->get_values(), u_nnz, u_factor->get_values(), num_sweeps,
            dependencies, nz_ptrs);
    }
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
    const batch_csr::UniformBatch<const ValueType> l_batch =
        gko::kernels::host::get_batch_struct(l_factor);
    const batch_csr::UniformBatch<const ValueType> u_batch =
        gko::kernels::host::get_batch_struct(u_factor);
    const auto rub = gko::kernels::host::get_batch_struct(r);
    const auto zub = gko::kernels::host::get_batch_struct(z);

    using prec_type = gko::kernels::host::batch_parilu0<ValueType>;
    prec_type prec(l_batch, u_batch);

    const auto work_arr_size =
        prec_type::dynamic_work_size(l_batch.num_rows, -1);
    std::vector<ValueType> work(work_arr_size);

    for (size_type batch_id = 0; batch_id < l_factor->get_num_batch_entries();
         batch_id++) {
        const auto r_b = gko::batch::batch_entry(rub, batch_id);
        const auto z_b = gko::batch::batch_entry(zub, batch_id);
        prec.generate(batch_id, gko::batch_csr::BatchEntry<const ValueType>(),
                      work.data());
        prec.apply(r_b, z_b);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_APPLY_KERNEL);


}  // namespace batch_par_ilu
}  // namespace reference
}  // namespace kernels
}  // namespace gko
