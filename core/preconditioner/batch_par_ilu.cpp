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

#include <ginkgo/core/preconditioner/batch_par_ilu.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/batch_par_ilu_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace batch_par_ilu {
namespace {


GKO_REGISTER_OPERATION(nonbatch_check_diag_entries_exist,
                       csr::check_diagonal_entries_exist);
GKO_REGISTER_OPERATION(nonbatch_initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(generate_common_pattern_to_fill_l_and_u,
                       batch_par_ilu::generate_common_pattern_to_fill_l_and_u);
GKO_REGISTER_OPERATION(initialize_batch_l_and_batch_u,
                       batch_par_ilu::initialize_batch_l_and_batch_u);
GKO_REGISTER_OPERATION(compute_par_ilu0, batch_par_ilu::compute_par_ilu0);


template <typename ValueType, typename IndexType>
void create_dependency_graph(
    std::shared_ptr<const Executor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* sys_mat,
    const matrix::BatchCsr<ValueType, IndexType>* l_factor,
    const matrix::BatchCsr<ValueType, IndexType>* u_factor,
    std::vector<IndexType>& dependencies, array<IndexType>& nz_ptrs)
{
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto nrows = sys_mat->get_size().at(0)[0];
    const auto nnz = sys_mat->get_num_stored_elements() / nbatch;
    const auto l_nnz = l_factor->get_num_stored_elements() / nbatch;
    const auto u_nnz = u_factor->get_num_stored_elements() / nbatch;

    array<IndexType> A_row_ptrs(exec->get_master(), nrows + 1);
    exec->get_master()->copy_from(exec.get(), nrows + 1,
                                  sys_mat->get_const_row_ptrs(),
                                  A_row_ptrs.get_data());
    array<IndexType> A_col_idxs(exec->get_master(), nnz);
    exec->get_master()->copy_from(
        exec.get(), nnz, sys_mat->get_const_col_idxs(), A_col_idxs.get_data());
    array<IndexType> L_row_ptrs(exec->get_master(), nrows + 1);
    exec->get_master()->copy_from(exec.get(), nrows + 1,
                                  l_factor->get_const_row_ptrs(),
                                  L_row_ptrs.get_data());
    array<IndexType> L_col_idxs(exec->get_master(), l_nnz);
    exec->get_master()->copy_from(exec.get(), l_nnz,
                                  l_factor->get_const_col_idxs(),
                                  L_col_idxs.get_data());
    array<IndexType> U_row_ptrs(exec->get_master(), nrows + 1);
    exec->get_master()->copy_from(exec.get(), nrows + 1,
                                  u_factor->get_const_row_ptrs(),
                                  U_row_ptrs.get_data());
    array<IndexType> U_col_idxs(exec->get_master(), u_nnz);
    exec->get_master()->copy_from(exec.get(), u_nnz,
                                  u_factor->get_const_col_idxs(),
                                  U_col_idxs.get_data());

    nz_ptrs.get_data()[0] = 0;

    for (int row_index = 0; row_index < nrows; row_index++) {
        const int row_start = A_row_ptrs.get_const_data()[row_index];
        const int row_end = A_row_ptrs.get_const_data()[row_index + 1];

        for (int loc = row_start; loc < row_end; loc++) {
            const int col_index = A_col_idxs.get_const_data()[loc];

            if (row_index > col_index) {
                // find corr. index in L
                const int L_idx =
                    loc - row_start + L_row_ptrs.get_const_data()[row_index];
                dependencies.push_back(L_idx);
            } else {
                // find corr. index in U
                const int U_idx =
                    (U_row_ptrs.get_const_data()[row_index + 1] - 1) -
                    (row_end - 1 - loc);
                dependencies.push_back(U_idx);
            }

            const int k_max = std::min(row_index, col_index) - 1;

            int num_dependencies = 0;

            for (int l_idx = L_row_ptrs.get_const_data()[row_index];
                 l_idx < L_row_ptrs.get_const_data()[row_index + 1]; l_idx++) {
                const int k = L_col_idxs.get_const_data()[l_idx];

                if (k > k_max) {
                    continue;
                }

                // find corresponding u at position k,col_index
                for (int u_idx = U_row_ptrs.get_const_data()[k];
                     u_idx < U_row_ptrs.get_const_data()[k + 1]; u_idx++) {
                    if (U_col_idxs.get_const_data()[u_idx] == col_index) {
                        dependencies.push_back(l_idx);
                        dependencies.push_back(u_idx);
                        num_dependencies += 2;
                    }
                }
            }


            if (row_index > col_index) {
                const int diag_loc = U_row_ptrs.get_const_data()[col_index];
                dependencies.push_back(diag_loc);
                num_dependencies++;
            }

            nz_ptrs.get_data()[loc + 1] =
                nz_ptrs.get_const_data()[loc] + num_dependencies + 1;
        }
    }
}


}  // namespace
}  // namespace batch_par_ilu


template <typename ValueType, typename IndexType>
void BatchParIlu<ValueType, IndexType>::generate_precond(
    const BatchLinOp* const system_matrix)
{
    using unbatch_type = matrix::Csr<ValueType, IndexType>;
    // generate entire batch of factorizations
    if (!system_matrix->get_size().stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }
    auto exec = this->get_executor();

    std::shared_ptr<matrix_type> csr_mat;
    const matrix_type* sys_csr = nullptr;
    if (auto temp_csr = dynamic_cast<const matrix_type*>(system_matrix)) {
        csr_mat = gko::share(gko::clone(exec, temp_csr));
    } else {
        csr_mat = gko::share(matrix_type::create(exec));
        as<ConvertibleTo<matrix_type>>(system_matrix)
            ->convert_to(csr_mat.get());
    }

    if (parameters_.skip_sorting != true) {
        csr_mat->sort_by_column_index();
    }

    const auto num_batch = csr_mat->get_num_batch_entries();
    const auto num_rows = csr_mat->get_size().at(0)[0];
    const auto num_nz = csr_mat->get_num_stored_elements() / num_batch;

    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size =
        gko::dim<2>{num_rows, csr_mat->get_size().at(0)[1]};
    auto sys_rows_view = array<IndexType>::const_view(
        exec, num_rows + 1, csr_mat->get_const_row_ptrs());
    auto sys_cols_view = array<IndexType>::const_view(
        exec, num_nz, csr_mat->get_const_col_idxs());
    auto sys_vals_view =
        array<ValueType>::const_view(exec, num_nz, csr_mat->get_const_values());
    auto first_csr = unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view));


    bool all_diags{false};
    exec->run(batch_par_ilu::make_nonbatch_check_diag_entries_exist(
        first_csr.get(), all_diags));
    if (!all_diags) {
        throw std::runtime_error("Matrix does not have all diagonal entries!");
    }

    // initialize L and U factors
    array<IndexType> l_row_ptrs(exec, num_rows + 1);
    array<IndexType> u_row_ptrs(exec, num_rows + 1);
    exec->run(batch_par_ilu::make_nonbatch_initialize_row_ptrs_l_u(
        first_csr.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));
    const auto l_nnz =
        exec->copy_val_to_host(&l_row_ptrs.get_const_data()[num_rows]);
    const auto u_nnz =
        exec->copy_val_to_host(&u_row_ptrs.get_const_data()[num_rows]);

    l_factor_ =
        gko::share(matrix_type::create(exec, num_batch, unbatch_size, l_nnz));
    u_factor_ =
        gko::share(matrix_type::create(exec, num_batch, unbatch_size, u_nnz));

    exec->copy(num_rows + 1, l_row_ptrs.get_const_data(),
               l_factor_->get_row_ptrs());
    exec->copy(num_rows + 1, u_row_ptrs.get_const_data(),
               u_factor_->get_row_ptrs());


    // fill batch_L and batch_U col_idxs and values
    array<IndexType> l_col_holders(exec, l_nnz);
    array<IndexType> u_col_holders(exec, u_nnz);

    exec->run(batch_par_ilu::make_generate_common_pattern_to_fill_l_and_u(
        first_csr.get(), l_factor_->get_const_row_ptrs(),
        u_factor_->get_const_row_ptrs(), l_col_holders.get_data(),
        u_col_holders.get_data()));

    exec->run(batch_par_ilu::make_initialize_batch_l_and_batch_u(
        csr_mat.get(), l_factor_.get(), u_factor_.get(),
        l_col_holders.get_const_data(), u_col_holders.get_const_data()));

    std::vector<IndexType> dependencies_vec;
    array<IndexType> nz_ptrs(exec->get_master(), num_nz + 1);

    batch_par_ilu::create_dependency_graph(exec, csr_mat.get(), l_factor_.get(),
                                           u_factor_.get(), dependencies_vec,
                                           nz_ptrs);

    array<IndexType> dependencies(exec, dependencies_vec.size());
    exec->copy_from(exec->get_master().get(), dependencies_vec.size(),
                    dependencies_vec.data(), dependencies.get_data());

    nz_ptrs.set_executor(exec);

    exec->run(batch_par_ilu::make_compute_par_ilu0(
        csr_mat.get(), l_factor_.get(), u_factor_.get(), parameters_.num_sweeps,
        dependencies.get_const_data(), nz_ptrs.get_const_data()));
}


#define GKO_DECLARE_BATCH_PAR_ILU(ValueType) class BatchParIlu<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_PAR_ILU);


}  // namespace preconditioner
}  // namespace gko
