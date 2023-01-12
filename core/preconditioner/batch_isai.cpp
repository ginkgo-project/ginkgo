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


#include <ginkgo/core/preconditioner/batch_ilu.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>
#include <ginkgo/core/solver/batch_bicgstab.hpp>
#include <ginkgo/core/solver/batch_gmres.hpp>
#include <ginkgo/core/solver/batch_lower_trs.hpp>
#include <ginkgo/core/solver/batch_upper_trs.hpp>


#include "core/components/prefix_sum_kernels.hpp"
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
GKO_REGISTER_OPERATION(prefix_sum, components::prefix_sum);
GKO_REGISTER_OPERATION(extract_csr_sys_pattern,
                       batch_isai::extract_csr_sys_pattern);
GKO_REGISTER_OPERATION(fill_batch_csr_sys_with_values,
                       batch_isai::fill_batch_csr_sys_with_values);
GKO_REGISTER_OPERATION(initialize_b_and_x_vectors,
                       batch_isai::initialize_b_and_x_vectors);
GKO_REGISTER_OPERATION(write_large_sys_solution_to_inverse,
                       batch_isai::write_large_sys_solution_to_inverse);


}  // namespace
}  // namespace batch_isai


namespace detail {

template <typename ValueType, typename IndexType>
void batch_isai_extension(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> first_sys_csr,
    std::shared_ptr<matrix::Csr<ValueType, IndexType>> first_approx_inv,
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>> sys_csr,
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>> approx_inv,
    const gko::array<IndexType>& sizes,
    const gko::array<IndexType>& rhs_one_idxs,
    const gko::array<IndexType>& num_matches_per_row_for_each_csr_sys,
    const gko::preconditioner::batch_isai_input_matrix_type&
        input_matrix_type_isai)
{
    using mtx_type = matrix::BatchCsr<ValueType, IndexType>;
    using unbatch_type = matrix::Csr<ValueType, IndexType>;
    using BDense = matrix::BatchDense<ValueType>;
    using lower_trs = solver::BatchLowerTrs<ValueType>;
    using upper_trs = solver::BatchUpperTrs<ValueType>;
    using bicgstab = solver::BatchBicgstab<ValueType>;
    using gmres = solver::BatchGmres<ValueType>;
    using RealValueType = gko::remove_complex<ValueType>;

    const auto nrows = sys_csr->get_size().at(0)[0];
    const auto nbatch = sys_csr->get_num_batch_entries();

    gko::array<IndexType> sizes_host(exec->get_master(), nrows);
    exec->get_master()->copy_from(exec.get(), nrows, sizes.get_const_data(),
                                  sizes_host.get_data());

    gko::array<IndexType> rhs_one_idxs_host(exec->get_master(), nrows);
    exec->get_master()->copy_from(exec.get(), nrows,
                                  rhs_one_idxs.get_const_data(),
                                  rhs_one_idxs_host.get_data());

    for (int lin_sys_row = 0; lin_sys_row < sizes_host.get_num_elems();
         lin_sys_row++) {
        const auto size = sizes_host.get_const_data()[lin_sys_row];
        const auto rhs_one_idx =
            rhs_one_idxs_host.get_const_data()[lin_sys_row];

        if (size <= gko::preconditioner::batch_isai::row_size_limit) {
            continue;
        }

        // row_ptrs for csr pattern
        array<IndexType> csr_pattern_row_ptrs_arr(exec, size + 1);

        const auto offset = exec->copy_val_to_host(
            first_approx_inv->get_const_row_ptrs() + lin_sys_row);

        exec->copy(
            size,
            num_matches_per_row_for_each_csr_sys.get_const_data() + offset,
            csr_pattern_row_ptrs_arr.get_data());

        exec->run(batch_isai::make_prefix_sum(
            csr_pattern_row_ptrs_arr.get_data(), size + 1));

        // extract csr pattern
        IndexType csr_nnz = exec->copy_val_to_host(
            csr_pattern_row_ptrs_arr.get_const_data() + size);
        array<IndexType> csr_pattern_col_idxs_arr(exec, csr_nnz);
        array<RealValueType> csr_pattern_values_arr(exec, csr_nnz);
        std::shared_ptr<matrix::Csr<RealValueType, IndexType>> csr_pattern =
            gko::share(matrix::Csr<RealValueType, IndexType>::create(
                exec, gko::dim<2>(size, size),
                std::move(csr_pattern_values_arr),
                std::move(csr_pattern_col_idxs_arr),
                std::move(csr_pattern_row_ptrs_arr)));

        // Now extract csr pattern
        exec->run(batch_isai::make_extract_csr_sys_pattern(
            lin_sys_row, size, first_approx_inv.get(), first_sys_csr.get(),
            csr_pattern.get()));

        auto csr_pattern_transposed = as<matrix::Csr<RealValueType, IndexType>>(
            gko::share(csr_pattern->transpose()));

        // Now create a batched csr matrix and fill it with values
        auto batch_csr_mats = gko::share(
            mtx_type::create(exec, nbatch, gko::dim<2>(size, size), csr_nnz));
        exec->copy(size + 1, csr_pattern_transposed->get_const_row_ptrs(),
                   batch_csr_mats->get_row_ptrs());
        exec->copy(csr_nnz, csr_pattern_transposed->get_const_col_idxs(),
                   batch_csr_mats->get_col_idxs());

        exec->run(batch_isai::make_fill_batch_csr_sys_with_values(
            csr_pattern_transposed.get(), sys_csr.get(), batch_csr_mats.get()));

        auto b = gko::share(BDense::create(
            exec, gko::batch_dim<2>(nbatch, gko::dim<2>(size, 1))));
        auto x = gko::share(BDense::create(
            exec, gko::batch_dim<2>(nbatch, gko::dim<2>(size, 1))));

        exec->run(batch_isai::make_initialize_b_and_x_vectors(
            rhs_one_idx, b.get(), x.get()));

        if (input_matrix_type_isai ==
            gko::preconditioner::batch_isai_input_matrix_type::lower_tri) {
            auto solver =
                upper_trs::build().with_skip_sorting(true).on(exec)->generate(
                    batch_csr_mats);

            solver->apply(b.get(), x.get());

        } else if (input_matrix_type_isai ==
                   gko::preconditioner::batch_isai_input_matrix_type::
                       upper_tri) {
            auto solver =
                lower_trs::build().with_skip_sorting(true).on(exec)->generate(
                    batch_csr_mats);

            solver->apply(b.get(), x.get());

        } else if (input_matrix_type_isai ==
                   gko::preconditioner::batch_isai_input_matrix_type::general) {
            const RealValueType reduction_factor{1e-10};
            auto solver = gmres::build()
                              .with_default_max_iterations(1000)
                              .with_tolerance_type(
                                  gko::stop::batch::ToleranceType::absolute)
                              .with_default_residual_tol(reduction_factor)
                              .with_preconditioner(
                                  preconditioner::BatchIlu<ValueType>::build()
                                      .with_skip_sorting(true)
                                      .on(exec))
                              .on(exec)
                              ->generate(batch_csr_mats);
            solver->apply(b.get(), x.get());
        } else {
            GKO_NOT_SUPPORTED(input_matrix_type_isai);
        }

        // write solution back to approx inv
        exec->run(batch_isai::make_write_large_sys_solution_to_inverse(
            lin_sys_row, x.get(), approx_inv.get()));
    }
}

}  // namespace detail


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

    // Say spy(laiA) = S
    // laiA * A = I on S
    // For each row i => solve laiA[i,J] * A[J,J] = I[i,J] where J = set of non
    // zero columns in ith row of laiA i.e. S(i,:)

    // Instead of doing repeated computations for each matrix in the batch to
    // extract a submatrix corresponding to a particular row,
    // we first extract and store the common pattern and then simply create the
    // submatrices based on that common pattern

    // arrays to store extracted submatrix pattern
    gko::array<IndexType> dense_mat_pattern(
        exec,
        num_rows * batch_isai::row_size_limit * batch_isai::row_size_limit);
    dense_mat_pattern.fill(static_cast<IndexType>(-1));
    gko::array<IndexType> rhs_one_idxs(exec, num_rows);
    rhs_one_idxs.fill(static_cast<IndexType>(-1));
    gko::array<IndexType> sizes(exec, num_rows);
    gko::array<IndexType> num_matches_per_row_for_each_csr_sys(
        exec, first_approx_inv->get_num_stored_elements());
    num_matches_per_row_for_each_csr_sys.fill(static_cast<IndexType>(-1));

    // For rows of the inverse having nnz <= 32, we extract and store the
    // corresponding linear system in dense format
    exec->run(batch_isai::make_extract_dense_linear_sys_pattern(
        first_sys_csr.get(), first_approx_inv.get(),
        dense_mat_pattern.get_data(), rhs_one_idxs.get_data(), sizes.get_data(),
        num_matches_per_row_for_each_csr_sys.get_data()));

    exec->run(batch_isai::make_fill_values_dense_mat_and_solve(
        sys_csr.get(), this->approx_inv_.get(),
        dense_mat_pattern.get_const_data(), rhs_one_idxs.get_const_data(),
        sizes.get_const_data(), this->parameters_.isai_input_matrix_type));

    // For rows of the inverse having nnz > 32, we extract and store the
    // corresponding linear system in csr format
    detail::batch_isai_extension(exec, first_sys_csr, first_approx_inv, sys_csr,
                                 approx_inv_, sizes, rhs_one_idxs,
                                 num_matches_per_row_for_each_csr_sys,
                                 this->parameters_.isai_input_matrix_type);
}


#define GKO_DECLARE_BATCH_ISAI(ValueType) class BatchIsai<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_ISAI);


}  // namespace preconditioner
}  // namespace gko
