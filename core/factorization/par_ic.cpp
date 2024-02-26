// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/par_ic.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/array_access.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/par_ic_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace factorization {
namespace par_ic_factorization {
namespace {


GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l,
                       factorization::initialize_row_ptrs_l);
GKO_REGISTER_OPERATION(initialize_l, factorization::initialize_l);
GKO_REGISTER_OPERATION(init_factor, par_ic_factorization::init_factor);
GKO_REGISTER_OPERATION(compute_factor, par_ic_factorization::compute_factor);
GKO_REGISTER_OPERATION(csr_transpose, csr::transpose);
GKO_REGISTER_OPERATION(convert_ptrs_to_idxs, components::convert_ptrs_to_idxs);


}  // anonymous namespace
}  // namespace par_ic_factorization


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> ParIc<ValueType, IndexType>::generate(
    const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
    bool both_factors) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto csr_system_matrix = CsrMatrix::create(exec);
    as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
        ->convert_to(csr_system_matrix);
    // If necessary, sort it
    if (!skip_sorting) {
        csr_system_matrix->sort_by_column_index();
    }

    // Add explicit diagonal zero elements if they are missing
    exec->run(par_ic_factorization::make_add_diagonal_elements(
        csr_system_matrix.get(), true));

    const auto matrix_size = csr_system_matrix->get_size();
    const auto number_rows = matrix_size[0];
    array<IndexType> l_row_ptrs{exec, number_rows + 1};
    exec->run(par_ic_factorization::make_initialize_row_ptrs_l(
        csr_system_matrix.get(), l_row_ptrs.get_data()));

    // Get nnz from device memory
    auto l_nnz = static_cast<size_type>(get_element(l_row_ptrs, number_rows));

    // Since `row_ptrs` of L is already created, the matrix can be
    // directly created with it
    array<IndexType> l_col_idxs{exec, l_nnz};
    array<ValueType> l_vals{exec, l_nnz};
    std::shared_ptr<CsrMatrix> l_factor = matrix_type::create(
        exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
        std::move(l_row_ptrs), parameters_.l_strategy);

    exec->run(par_ic_factorization::make_initialize_l(csr_system_matrix.get(),
                                                      l_factor.get(), false));

    // build COO representation of lower factor
    array<IndexType> l_row_idxs{exec, l_nnz};
    // copy values from l_factor, which are the lower triangular values of A
    auto l_vals_view = make_array_view(exec, l_nnz, l_factor->get_values());
    auto a_vals = array<ValueType>{exec, l_vals_view};
    auto a_row_idxs = array<IndexType>{exec, l_nnz};
    auto a_col_idxs = make_array_view(exec, l_nnz, l_factor->get_col_idxs());
    auto a_lower_coo =
        CooMatrix::create(exec, matrix_size, std::move(a_vals),
                          std::move(a_col_idxs), std::move(a_row_idxs));

    // compute sqrt of diagonal entries
    exec->run(par_ic_factorization::make_init_factor(l_factor.get()));

    // execute sweeps
    exec->run(par_ic_factorization::make_compute_factor(
        parameters_.iterations, a_lower_coo.get(), l_factor.get()));

    if (both_factors) {
        auto lh_factor = l_factor->conj_transpose();
        return Composition<ValueType>::create(std::move(l_factor),
                                              std::move(lh_factor));
    } else {
        return Composition<ValueType>::create(std::move(l_factor));
    }
}


#define GKO_DECLARE_PAR_IC(ValueType, IndexType) \
    class ParIc<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_IC);


}  // namespace factorization
}  // namespace gko
