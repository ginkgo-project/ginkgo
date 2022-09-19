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

#include <ginkgo/core/preconditioner/batch_exact_ilu.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/batch_exact_ilu_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace batch_exact_ilu {
namespace {


GKO_REGISTER_OPERATION(nonbatch_check_diag_entries_exist,
                       csr::check_diagonal_entries_exist);
GKO_REGISTER_OPERATION(nonbatch_find_diag_locs,
                       csr::find_diagonal_entries_locations);
GKO_REGISTER_OPERATION(compute_factorization,
                       batch_exact_ilu::compute_factorization);


}  // namespace
}  // namespace batch_exact_ilu


template <typename ValueType, typename IndexType>
void BatchExactIlu<ValueType, IndexType>::generate_precond(
    const BatchLinOp* const system_matrix)
{
    using unbatch_type = matrix::Csr<ValueType, IndexType>;
    // generate entire batch of factorizations
    if (!system_matrix->get_size().stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }
    auto exec = this->get_executor();

    if (auto temp_csr = dynamic_cast<const matrix_type*>(system_matrix)) {
        mat_factored_ = gko::share(gko::clone(exec, temp_csr));
    } else {
        mat_factored_ = gko::share(matrix_type::create(exec));
        as<ConvertibleTo<matrix_type>>(system_matrix)
            ->convert_to(mat_factored_.get());
    }


    if (parameters_.skip_sorting != true) {
        mat_factored_->sort_by_column_index();
    }


    const auto num_batch = mat_factored_->get_num_batch_entries();
    const auto num_rows = mat_factored_->get_size().at(0)[0];
    const auto num_nz = mat_factored_->get_num_stored_elements() / num_batch;

    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size =
        gko::dim<2>{num_rows, mat_factored_->get_size().at(0)[1]};
    auto sys_rows_view = array<IndexType>::const_view(
        exec, num_rows + 1, mat_factored_->get_const_row_ptrs());
    auto sys_cols_view = array<IndexType>::const_view(
        exec, num_nz, mat_factored_->get_const_col_idxs());
    auto sys_vals_view = array<ValueType>::const_view(
        exec, num_nz, mat_factored_->get_const_values());
    auto first_csr = unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view));


    bool all_diags{false};
    exec->run(batch_exact_ilu::make_nonbatch_check_diag_entries_exist(
        first_csr.get(), all_diags));
    if (!all_diags) {
        // TODO: Add exception and macro for this.
        throw std::runtime_error("Matrix does not have all diagonal entries!");
        // TODO: Add a diagonal addition kernel for this.
    }

    diag_locations_ = array<IndexType>(exec, num_rows);
    exec->run(batch_exact_ilu::make_nonbatch_find_diag_locs(
        first_csr.get(), diag_locations_.get_data()));

    // Now given that the matrix is in csr form, sorted with all diagonal
    // entries, the following algo. computes exact ILU0 factorization
    exec->run(batch_exact_ilu::make_compute_factorization(
        diag_locations_.get_const_data(), mat_factored_.get()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchExactIlu<ValueType, IndexType>::transpose()
    const GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp>
BatchExactIlu<ValueType, IndexType>::conj_transpose() const GKO_NOT_IMPLEMENTED;


#define GKO_DECLARE_BATCH_EXACT_ILU(ValueType) \
    class BatchExactIlu<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_EXACT_ILU);


}  // namespace preconditioner
}  // namespace gko
