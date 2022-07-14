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

#include <ginkgo/core/preconditioner/batch_ilu.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/batch_csr_kernels.hpp"
#include "core/preconditioner/batch_ilu_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace batch_ilu {
namespace {


GKO_REGISTER_OPERATION(unbatch_initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(unbatch_initialize_l_u, factorization::initialize_l_u);
GKO_REGISTER_OPERATION(check_diag_entries_exist,
                       batch_csr::check_diagonal_entries_exist);
GKO_REGISTER_OPERATION(generate_split, batch_ilu::generate_split);


}  // namespace
}  // namespace batch_ilu


template <typename ValueType, typename IndexType>
void BatchIlu<ValueType, IndexType>::generate(
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
    exec->run(batch_ilu::make_check_diag_entries_exist(sys_csr, all_diags));
    if (!all_diags) {
        // TODO: Add exception and macro for this.
        throw std::runtime_error("Matrix does not have all diagonal entries!");
    }
    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto nrows = sys_csr->get_size().at(0)[0];
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;

    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size = gko::dim<2>{nrows, sys_csr->get_size().at(0)[1]};
    auto sys_rows_view = array<IndexType>::const_view(
        exec, nrows + 1, sys_csr->get_const_row_ptrs());
    auto sys_cols_view =
        array<IndexType>::const_view(exec, nnz, sys_csr->get_const_col_idxs());
    auto sys_vals_view =
        array<ValueType>::const_view(exec, nnz, sys_csr->get_const_values());
    auto first_csr = unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view));

    // initialize L and U factors
    array<IndexType> l_row_ptrs(exec, nrows + 1);
    array<IndexType> u_row_ptrs(exec, nrows + 1);
    exec->run(batch_ilu::make_unbatch_initialize_row_ptrs_l_u(
        first_csr.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));
    const auto l_nnz =
        exec->copy_val_to_host(&l_row_ptrs.get_const_data()[nrows]);
    const auto u_nnz =
        exec->copy_val_to_host(&u_row_ptrs.get_const_data()[nrows]);
    auto first_L = unbatch_type::create(exec, unbatch_size, l_nnz);
    auto first_U = unbatch_type::create(exec, unbatch_size, u_nnz);
    exec->copy(nrows + 1, l_row_ptrs.get_const_data(), first_L->get_row_ptrs());
    exec->copy(nrows + 1, u_row_ptrs.get_const_data(), first_U->get_row_ptrs());
    exec->run(batch_ilu::make_unbatch_initialize_l_u(
        first_csr.get(), first_L.get(), first_U.get()));
    l_factor_ =
        gko::share(matrix_type::create(exec, nbatch, unbatch_size, l_nnz));
    u_factor_ =
        gko::share(matrix_type::create(exec, nbatch, unbatch_size, u_nnz));
    exec->copy(nrows + 1, first_L->get_const_row_ptrs(),
               l_factor_->get_row_ptrs());
    exec->copy(nrows + 1, first_U->get_const_row_ptrs(),
               u_factor_->get_row_ptrs());
    exec->copy(l_nnz, first_L->get_const_col_idxs(), l_factor_->get_col_idxs());
    exec->copy(u_nnz, first_U->get_const_col_idxs(), u_factor_->get_col_idxs());

    // compute ILU factorization into the split factor matrices
    exec->run(batch_ilu::make_generate_split(parameters_.factorization_type,
                                             sys_csr, l_factor_.get(),
                                             u_factor_.get()));
}

template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchIlu<ValueType, IndexType>::transpose() const
{
    GKO_NOT_IMPLEMENTED;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchIlu<ValueType, IndexType>::conj_transpose()
    const
{
    GKO_NOT_IMPLEMENTED;
}


#define GKO_DECLARE_BATCH_ILU(ValueType) class BatchIlu<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_ILU);


}  // namespace preconditioner
}  // namespace gko
