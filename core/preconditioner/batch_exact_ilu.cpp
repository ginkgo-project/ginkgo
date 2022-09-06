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
#include "core/preconditioner/batch_exact_ilu_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace batch_exact_ilu {
namespace {


GKO_REGISTER_OPERATION(check_diag_entries_exist,
                       batch_csr::check_diagonal_entries_exist);
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
    const matrix_type* sys_csr{};
    auto a_matrix = matrix_type::create(exec);
    if (auto temp_csr = dynamic_cast<const matrix_type*>(system_matrix)) {
        sys_csr = temp_csr;
    } else {
        as<ConvertibleTo<matrix_type>>(system_matrix)
            ->convert_to(a_matrix.get());
        sys_csr = a_matrix.get();
    }

    bool all_diags{false};
    exec->run(
        batch_exact_ilu::make_check_diag_entries_exist(sys_csr, all_diags));
    if (!all_diags) {
        // TODO: Add exception and macro for this.
        throw std::runtime_error("Matrix does not have all diagonal entries!");
        // TODO: Add a diagonal addition kernel for this.
    }

    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto nrows = sys_csr->get_size().at(0)[0];
    const auto ncols = sys_csr->get_size().at(0)[1];
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;

    // TODO: If it is easy to check diag locations given that the matrix is
    // sorted, sort it first and then check for diag locations
    if (parameters_.skip_sorting != true) {
        // use sort_by_column_index() of batch csr class but on a non-const
        // batch csr mat object
    }

    // Now assuming the matrix is in csr form, sorted with all diagonal entries,
    // the following algo. computes exact ILU0 factorization

    // Using gko::clone()???
    mat_factored_ = gko::share(
        matrix_type::create(exec, nbatch, dim<2>(nrows, ncols), nnz));
    exec->copy(nnz, sys_csr->get_const_col_idxs(),
               mat_factored_->get_col_idxs());
    exec->copy(nrows + 1, sys_csr->get_const_row_ptrs(),
               mat_factored_->get_row_ptrs());
    exec->copy(nnz * nbatch, sys_csr->get_const_values(),
               mat_factored_->get_values());

    exec->run(batch_exact_ilu::make_compute_factorization(mat_factored_.get()));
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
