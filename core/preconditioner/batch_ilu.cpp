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

#include <ginkgo/core/preconditioner/batch_ilu.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/batch_ilu_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace batch_ilu {
namespace {


GKO_REGISTER_OPERATION(nonbatch_check_diag_entries_exist,
                       csr::check_diagonal_entries_exist);
GKO_REGISTER_OPERATION(nonbatch_find_diag_locs,
                       csr::find_diagonal_entries_locations);
GKO_REGISTER_OPERATION(nonbatch_initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(compute_ilu0_factorization,
                       batch_ilu::compute_ilu0_factorization);
GKO_REGISTER_OPERATION(compute_parilu0_factorization,
                       batch_ilu::compute_parilu0_factorization);
GKO_REGISTER_OPERATION(generate_common_pattern_to_fill_l_and_u,
                       batch_ilu::generate_common_pattern_to_fill_l_and_u);
GKO_REGISTER_OPERATION(initialize_batch_l_and_batch_u,
                       batch_ilu::initialize_batch_l_and_batch_u);

}  // namespace
}  // namespace batch_ilu


namespace detail {

template <typename ValueType, typename IndexType>
void create_dependency_graph(
    std::shared_ptr<const Executor> exec, const IndexType* const diag_locations,
    const matrix::BatchCsr<ValueType, IndexType>* sys_mat,
    std::vector<IndexType>& dependencies, array<IndexType>& nz_ptrs)
{
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto nrows = sys_mat->get_size().at(0)[0];
    const auto nnz = sys_mat->get_num_stored_elements() / nbatch;

    array<IndexType> A_row_ptrs(exec->get_master(), nrows + 1);
    exec->get_master()->copy_from(exec.get(), nrows + 1,
                                  sys_mat->get_const_row_ptrs(),
                                  A_row_ptrs.get_data());
    array<IndexType> A_col_idxs(exec->get_master(), nnz);
    exec->get_master()->copy_from(
        exec.get(), nnz, sys_mat->get_const_col_idxs(), A_col_idxs.get_data());

    array<IndexType> diag_ptrs(exec->get_master(), nrows);
    exec->get_master()->copy_from(exec.get(), nrows, diag_locations,
                                  diag_ptrs.get_data());

    nz_ptrs.get_data()[0] = 0;

    for (IndexType row_index = 0; row_index < nrows; row_index++) {
        const auto row_start = A_row_ptrs.get_const_data()[row_index];
        const auto row_end = A_row_ptrs.get_const_data()[row_index + 1];

        for (IndexType loc = row_start; loc < row_end; loc++) {
            IndexType num_dependencies = 0;

            const auto col_index = A_col_idxs.get_const_data()[loc];

            const auto k_max = std::min(row_index, col_index) - 1;

            for (IndexType maybe_l_loc = row_start; maybe_l_loc < loc;
                 maybe_l_loc++)  // use loc instead of row_end as the matrix is
                                 // sorted
            {
                const int k = A_col_idxs.get_const_data()[maybe_l_loc];

                if (k > k_max) {
                    continue;
                }

                // find corresponding u at position k,col_index

                for (IndexType maybe_u_loc = A_row_ptrs.get_const_data()[k];
                     maybe_u_loc < A_row_ptrs.get_const_data()[k + 1];
                     maybe_u_loc++) {
                    if (A_col_idxs.get_const_data()[maybe_u_loc] == col_index) {
                        dependencies.push_back(maybe_l_loc);
                        dependencies.push_back(maybe_u_loc);
                        num_dependencies += 2;
                    }
                }
            }

            if (row_index > col_index) {
                const auto diag_loc = diag_ptrs.get_const_data()[col_index];
                dependencies.push_back(diag_loc);
                num_dependencies++;
            }

            nz_ptrs.get_data()[loc + 1] =
                nz_ptrs.get_const_data()[loc] + num_dependencies;
        }
    }
}

}  // namespace detail

template <typename ValueType, typename IndexType>
std::pair<std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>,
          std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>>
BatchIlu<ValueType, IndexType>::generate_split_factors_from_factored_matrix()
    const
{
    using unbatch_type = matrix::Csr<ValueType, IndexType>;
    std::pair<std::shared_ptr<const matrix_type>,
              std::shared_ptr<const matrix_type>>
        l_and_u_factors;
    auto exec = this->get_executor();

    const auto num_batch = this->mat_factored_->get_num_batch_entries();
    const auto num_rows = this->mat_factored_->get_size().at(0)[0];
    const auto num_nz =
        this->mat_factored_->get_num_stored_elements() / num_batch;

    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size =
        gko::dim<2>{num_rows, this->mat_factored_->get_size().at(0)[1]};
    auto sys_rows_view = array<IndexType>::const_view(
        exec, num_rows + 1, this->mat_factored_->get_const_row_ptrs());
    auto sys_cols_view = array<IndexType>::const_view(
        exec, num_nz, this->mat_factored_->get_const_col_idxs());
    auto sys_vals_view = array<ValueType>::const_view(
        exec, num_nz, this->mat_factored_->get_const_values());
    auto first_csr = unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view));

    // initialize L and U factors
    array<IndexType> l_row_ptrs(exec, num_rows + 1);
    array<IndexType> u_row_ptrs(exec, num_rows + 1);
    exec->run(batch_ilu::make_nonbatch_initialize_row_ptrs_l_u(
        first_csr.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));
    const auto l_nnz =
        exec->copy_val_to_host(&l_row_ptrs.get_const_data()[num_rows]);
    const auto u_nnz =
        exec->copy_val_to_host(&u_row_ptrs.get_const_data()[num_rows]);

    auto l_factor =
        gko::share(matrix_type::create(exec, num_batch, unbatch_size, l_nnz));
    auto u_factor =
        gko::share(matrix_type::create(exec, num_batch, unbatch_size, u_nnz));

    exec->copy(num_rows + 1, l_row_ptrs.get_const_data(),
               l_factor->get_row_ptrs());
    exec->copy(num_rows + 1, u_row_ptrs.get_const_data(),
               u_factor->get_row_ptrs());

    // fill batch_L and batch_U col_idxs and values
    array<IndexType> l_col_holders(exec, l_nnz);
    array<IndexType> u_col_holders(exec, u_nnz);

    exec->run(batch_ilu::make_generate_common_pattern_to_fill_l_and_u(
        first_csr.get(), l_factor->get_const_row_ptrs(),
        u_factor->get_const_row_ptrs(), l_col_holders.get_data(),
        u_col_holders.get_data()));

    exec->run(batch_ilu::make_initialize_batch_l_and_batch_u(
        this->mat_factored_.get(), l_factor.get(), u_factor.get(),
        l_col_holders.get_const_data(), u_col_holders.get_const_data()));

    l_and_u_factors.first = l_factor;
    l_and_u_factors.second = u_factor;

    return l_and_u_factors;
}

template <typename ValueType, typename IndexType>
void BatchIlu<ValueType, IndexType>::generate_precond()
{
    using unbatch_type = matrix::Csr<ValueType, IndexType>;
    // generate entire batch of factorizations
    if (!this->system_matrix_->get_size().stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }
    auto exec = this->get_executor();

    if (auto temp_csr =
            dynamic_cast<const matrix_type*>(this->system_matrix_.get())) {
        mat_factored_ = gko::share(gko::clone(exec, temp_csr));
    } else {
        mat_factored_ = gko::share(matrix_type::create(exec));
        as<ConvertibleTo<matrix_type>>(this->system_matrix_.get())
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
    exec->run(batch_ilu::make_nonbatch_check_diag_entries_exist(first_csr.get(),
                                                                all_diags));
    if (!all_diags) {
        throw std::runtime_error("Matrix does not have all diagonal entries!");
    }

    diag_locations_ = array<IndexType>(exec, num_rows);
    exec->run(batch_ilu::make_nonbatch_find_diag_locs(
        first_csr.get(), diag_locations_.get_data()));

    if (parameters_.ilu_type == batch_ilu_type::exact_ilu) {
        // Now given that the matrix is in csr form, sorted with all diagonal
        // entries, the following algo. computes exact ILU0 factorization
        exec->run(batch_ilu::make_compute_ilu0_factorization(
            diag_locations_.get_const_data(), mat_factored_.get()));
    } else if (parameters_.ilu_type == batch_ilu_type::parilu) {
        // create dependency graph
        std::vector<IndexType> dependencies_vec;
        array<IndexType> nz_ptrs(exec->get_master(), num_nz + 1);

        detail::create_dependency_graph(exec, diag_locations_.get_const_data(),
                                        mat_factored_.get(), dependencies_vec,
                                        nz_ptrs);

        array<IndexType> dependencies(exec, dependencies_vec.size());
        exec->copy_from(exec->get_master().get(), dependencies_vec.size(),
                        dependencies_vec.data(), dependencies.get_data());

        nz_ptrs.set_executor(exec);

        auto csr_sys_mat = gko::clone(exec, mat_factored_);

        // compute parilu0
        exec->run(batch_ilu::make_compute_parilu0_factorization(
            csr_sys_mat.get(), mat_factored_.get(),
            parameters_.parilu_num_sweeps, dependencies.get_const_data(),
            nz_ptrs.get_const_data()));
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


#define GKO_DECLARE_BATCH_ILU(ValueType) class BatchIlu<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_ILU);


}  // namespace preconditioner
}  // namespace gko
