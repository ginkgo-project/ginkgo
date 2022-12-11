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

#include "core/preconditioner/batch_isai_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_isai.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace batch_isai {


#include "reference/preconditioner/batch_isai_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void extract_dense_linear_sys_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const matrix::Csr<ValueType, IndexType>* const first_approx_inv,
    IndexType* const dense_mat_pattern, IndexType* const rhs_one_idxs,
    IndexType* const sizes, IndexType* num_matches_per_row_for_each_csr_sys)
{
    // aiA * A = I on spy(aiA)
    // aiA(i, :) * A = I(i,:) on spy(aiA) for each row i
    // ==> Let J = set of non-zero cols of aiA[i,:], then aiA(i,J) * A(J,J) =
    // I(i,J) for each row i

    const auto num_rows = first_sys_csr->get_size()[0];
    const IndexType* const A_row_ptrs = first_sys_csr->get_const_row_ptrs();
    const IndexType* const A_col_idxs = first_sys_csr->get_const_col_idxs();
    const IndexType* const aiA_row_ptrs =
        first_approx_inv->get_const_row_ptrs();
    const IndexType* const aiA_col_idxs =
        first_approx_inv->get_const_col_idxs();

#pragma omp parallel for
    for (IndexType row_idx = 0; row_idx < num_rows; row_idx++) {
        extract_pattern_for_dense_sys_corr_to_current_row_impl(
            row_idx, static_cast<int>(num_rows), A_row_ptrs, A_col_idxs,
            aiA_row_ptrs, aiA_col_idxs, dense_mat_pattern, rhs_one_idxs, sizes,
            num_matches_per_row_for_each_csr_sys);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_EXTRACT_DENSE_LINEAR_SYSTEM_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_values_dense_mat_and_solve(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const inv,
    const IndexType* const dense_mat_pattern,
    const IndexType* const rhs_one_idxs, const IndexType* const sizes,
    const gko::preconditioner::batch_isai_input_matrix_type&
        input_matrix_type_isai)
{
    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto A_batch = host::get_batch_struct(sys_csr);
    const auto aiA_batch = host::get_batch_struct(inv);

#pragma omp parallel for
    for (size_t batch_idx = 0; batch_idx < nbatch; batch_idx++) {
        const auto A_entry = gko::batch::batch_entry(A_batch, batch_idx);
        const auto aiA_entry = gko::batch::batch_entry(aiA_batch, batch_idx);

        fill_values_dense_mat_and_solve_batch_entry_impl(
            A_entry, aiA_entry, dense_mat_pattern, rhs_one_idxs, sizes,
            input_matrix_type_isai);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_FILL_VALUES_DENSE_MATRIX_AND_SOLVE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_isai(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
                const matrix::BatchCsr<ValueType, IndexType>* const approx_inv,
                const matrix::BatchDense<ValueType>* const r,
                matrix::BatchDense<ValueType>* const z)
{
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto nrows = static_cast<int>(sys_mat->get_size().at(0)[0]);

    const batch_csr::UniformBatch<const ValueType> approx_inv_batch =
        gko::kernels::host::get_batch_struct(approx_inv);
    const auto rub = gko::kernels::host::get_batch_struct(r);
    const auto zub = gko::kernels::host::get_batch_struct(z);

    using prec_type = gko::kernels::host::batch_isai<ValueType>;
    prec_type prec(approx_inv_batch);

#pragma omp parallel for firstprivate(prec)
    for (size_type batch_id = 0; batch_id < nbatch; batch_id++) {
        const auto r_b = gko::batch::batch_entry(rub, batch_id);
        const auto z_b = gko::batch::batch_entry(zub, batch_id);

        const auto work_arr_size = prec_type::dynamic_work_size(
            nrows,
            static_cast<int>(sys_mat->get_num_stored_elements() / nbatch));
        std::vector<ValueType> work(work_arr_size);

        prec.generate(batch_id, gko::batch_csr::BatchEntry<const ValueType>(),
                      work.data());
        prec.apply(r_b, z_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void extract_csr_sys_pattern(
    std::shared_ptr<const DefaultExecutor> exec, const int lin_sys_row,
    const int size,
    const matrix::Csr<ValueType, IndexType>* const first_approx_inv,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    matrix::Csr<gko::remove_complex<ValueType>, IndexType>* const csr_pattern)
{
    const IndexType* const inv_row_ptrs =
        first_approx_inv->get_const_row_ptrs();
    const IndexType* const inv_col_idxs =
        first_approx_inv->get_const_col_idxs();
    const IndexType* const sys_row_ptrs = first_sys_csr->get_const_row_ptrs();
    const IndexType* const sys_col_idxs = first_sys_csr->get_const_col_idxs();
    const IndexType* const csr_pattern_row_ptrs =
        csr_pattern->get_const_row_ptrs();
    IndexType* const csr_pattern_col_idxs = csr_pattern->get_col_idxs();
    gko::remove_complex<ValueType>* const csr_pattern_values =
        csr_pattern->get_values();

    const int inv_row_st = inv_row_ptrs[lin_sys_row];
    const int inv_row_end = inv_row_ptrs[lin_sys_row + 1];

#pragma omp parallel for
    for (int inv_nz = inv_row_st; inv_nz < inv_row_end; inv_nz++) {
        extract_csr_row_impl<ValueType>(
            lin_sys_row, inv_nz, inv_row_ptrs, inv_col_idxs, sys_row_ptrs,
            sys_col_idxs, csr_pattern_row_ptrs, csr_pattern_col_idxs,
            csr_pattern_values);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_EXTRACT_CSR_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_batch_csr_sys_with_values(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<gko::remove_complex<ValueType>, IndexType>* const
        csr_pattern,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const batch_csr_mats)
{
    using real_type = gko::remove_complex<ValueType>;
    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto csr_nnz = csr_pattern->get_num_stored_elements();
    const auto sys_nnz = sys_csr->get_num_stored_elements() / nbatch;
    ValueType* const batch_csr_mats_values = batch_csr_mats->get_values();
    const real_type* const csr_pattern_values = csr_pattern->get_const_values();
    const ValueType* const sys_csr_values = sys_csr->get_const_values();

#pragma omp parallel for
    for (int i = 0; i < nbatch * csr_nnz; i++) {
        const int batch_id = i / csr_nnz;
        const int csr_nnz_id = i % csr_nnz;

        const int sys_idx = static_cast<int>(csr_pattern_values[csr_nnz_id]);
        assert(sys_idx >= 0 && sys_idx < sys_nnz);
        batch_csr_mats_values[batch_id * csr_nnz + csr_nnz_id] =
            sys_csr_values[batch_id * sys_nnz + sys_idx];
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_FILL_BATCH_CSR_SYSTEM_USING_PATTERN);


template <typename ValueType, typename IndexType>
void initialize_b_and_x_vectors(std::shared_ptr<const DefaultExecutor> exec,
                                const IndexType rhs_one_idx,
                                matrix::BatchDense<ValueType>* const b,
                                matrix::BatchDense<ValueType>* const x)
{
    const auto nbatch = b->get_num_batch_entries();
    const auto size = b->get_size().at(0)[0];
    ValueType* const b_vals = b->get_values();
    ValueType* const x_vals = x->get_values();

#pragma omp parallel for
    for (int i = 0; i < nbatch * size; i++) {
        const int batch_id = i / size;
        const int idx = i % size;

        b_vals[idx + batch_id * size] = zero<ValueType>();
        if (idx == rhs_one_idx) {
            b_vals[idx + batch_id * size] = one<ValueType>();
        }
        x_vals[idx + batch_id * size] = zero<ValueType>();
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_INITIALIZE_B_AND_X);

template <typename ValueType, typename IndexType>
void write_large_sys_solution_to_inverse(
    std::shared_ptr<const DefaultExecutor> exec, const int lin_sys_row,
    const matrix::BatchDense<ValueType>* const x,
    matrix::BatchCsr<ValueType, IndexType>* const approx_inv)
{
    const auto nbatch = x->get_num_batch_entries();
    const auto size = x->get_size().at(0)[0];
    const ValueType* const x_vals = x->get_const_values();
    const IndexType* const inv_row_ptrs = approx_inv->get_const_row_ptrs();
    assert(size == inv_row_ptrs[lin_sys_row + 1] - inv_row_ptrs[lin_sys_row]);

    ValueType* const inv_vals = approx_inv->get_values();
    const auto inv_nnz = approx_inv->get_num_stored_elements() / nbatch;

#pragma omp parallel for
    for (int i = 0; i < nbatch * size; i++) {
        const int batch_id = i / size;
        const int idx = i % size;

        inv_vals[inv_row_ptrs[lin_sys_row] + idx + batch_id * inv_nnz] =
            x_vals[idx + batch_id * size];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_WRITE_SOLUTION_TO_INVERSE);


}  // namespace batch_isai
}  // namespace omp
}  // namespace kernels
}  // namespace gko
