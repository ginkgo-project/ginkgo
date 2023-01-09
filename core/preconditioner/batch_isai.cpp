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

#include <ginkgo/core/preconditioner/batch_isai.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/batch_isai_kernels.hpp"
#include "core/preconditioner/isai_kernels.hpp"

namespace gko {
namespace preconditioner {
namespace batch_isai {

namespace {


GKO_REGISTER_OPERATION(extract_dense_linear_sys_pattern,
                       batch_isai::extract_dense_linear_sys_pattern);
GKO_REGISTER_OPERATION(fill_values_dense_mat_and_solve,
                       batch_isai::fill_values_dense_mat_and_solve);

}  // namespace
}  // namespace batch_isai


template <typename ValueType, typename IndexType>
void BatchIsai<ValueType, IndexType>::generate_precond()
{
    using unbatch_type = matrix::Csr<ValueType, IndexType>;
    // generate entire batch of factorizations
    if (!this->system_matrix_->get_size().stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }
    auto exec = this->get_executor();

    std::shared_ptr<matrix_type> sys_csr;

    if (auto temp_csr =
            dynamic_cast<const matrix_type*>(this->system_matrix_.get())) {
        sys_csr = gko::share(gko::clone(exec, temp_csr));
    } else {
        sys_csr = gko::share(matrix_type::create(exec));
        as<ConvertibleTo<matrix_type>>(this->system_matrix_.get())
            ->convert_to(sys_csr.get());
    }

    if (parameters_.skip_sorting != true) {
        sys_csr->sort_by_column_index();
    }


    const auto num_batch = sys_csr->get_num_batch_entries();
    const auto num_rows = sys_csr->get_size().at(0)[0];
    const auto num_nz = sys_csr->get_num_stored_elements() / num_batch;

    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size =
        gko::dim<2>{num_rows, sys_csr->get_size().at(0)[1]};
    auto sys_rows_view = array<IndexType>::const_view(
        exec, num_rows + 1, sys_csr->get_const_row_ptrs());
    auto sys_cols_view = array<IndexType>::const_view(
        exec, num_nz, sys_csr->get_const_col_idxs());
    auto sys_vals_view =
        array<ValueType>::const_view(exec, num_nz, sys_csr->get_const_values());
    auto first_sys_csr = gko::share(unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view)));


    // find approx_inv's sparsity pattern (and allocate memory for it, set its
    // row_ptrs and col_idxs)
    std::shared_ptr<unbatch_type> first_approx_inv;
    first_approx_inv = gko::matrix::detail::extend_sparsity(
        exec, first_sys_csr, parameters_.sparsity_power);

    const auto first_approx_inv_nnz =
        first_approx_inv->get_num_stored_elements();
    this->approx_inv_ = gko::share(matrix_type::create(
        exec, num_batch, first_approx_inv->get_size(), first_approx_inv_nnz));
    exec->copy(num_rows + 1, first_approx_inv->get_const_row_ptrs(),
               approx_inv_->get_row_ptrs());
    exec->copy(first_approx_inv_nnz, first_approx_inv->get_const_col_idxs(),
               approx_inv_->get_col_idxs());


    // arrays to store extracted submatrix pattern
    gko::array<IndexType> dense_mat_pattern(
        exec,
        num_rows * batch_isai::row_size_limit * batch_isai::row_size_limit);
    dense_mat_pattern.fill(static_cast<IndexType>(-1));
    gko::array<IndexType> rhs_one_idxs(exec, num_rows);
    rhs_one_idxs.fill(static_cast<IndexType>(-1));
    gko::array<IndexType> sizes(exec, num_rows);

    exec->run(batch_isai::make_extract_dense_linear_sys_pattern(
        first_sys_csr.get(), first_approx_inv.get(),
        dense_mat_pattern.get_data(), rhs_one_idxs.get_data(),
        sizes.get_data()));

    exec->run(batch_isai::make_fill_values_dense_mat_and_solve(
        sys_csr.get(), this->approx_inv_.get(),
        dense_mat_pattern.get_const_data(), rhs_one_idxs.get_const_data(),
        sizes.get_const_data(), this->parameters_.isai_input_matrix_type));
}


#define GKO_DECLARE_BATCH_ISAI(ValueType) class BatchIsai<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_ISAI);


}  // namespace preconditioner
}  // namespace gko
