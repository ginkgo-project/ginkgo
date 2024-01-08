// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/ilu.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/array_access.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/factorization/par_ilu_kernels.hpp"


namespace gko {
namespace factorization {
namespace ilu_factorization {
namespace {


GKO_REGISTER_OPERATION(compute_ilu, ilu_factorization::compute_lu);
GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, factorization::initialize_l_u);


}  // anonymous namespace
}  // namespace ilu_factorization


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> Ilu<ValueType, IndexType>::generate_l_u(
    const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto local_system_matrix = matrix_type::create(exec);
    as<ConvertibleTo<matrix_type>>(system_matrix.get())
        ->convert_to(local_system_matrix);

    if (!skip_sorting) {
        local_system_matrix->sort_by_column_index();
    }

    // Add explicit diagonal zero elements if they are missing
    exec->run(ilu_factorization::make_add_diagonal_elements(
        local_system_matrix.get(), false));

    // Compute LU factorization
    exec->run(ilu_factorization::make_compute_ilu(local_system_matrix.get()));

    // Separate L and U factors: nnz
    const auto matrix_size = local_system_matrix->get_size();
    const auto num_rows = matrix_size[0];
    array<IndexType> l_row_ptrs{exec, num_rows + 1};
    array<IndexType> u_row_ptrs{exec, num_rows + 1};
    exec->run(ilu_factorization::make_initialize_row_ptrs_l_u(
        local_system_matrix.get(), l_row_ptrs.get_data(),
        u_row_ptrs.get_data()));

    // Get nnz from device memory
    auto l_nnz = static_cast<size_type>(get_element(l_row_ptrs, num_rows));
    auto u_nnz = static_cast<size_type>(get_element(u_row_ptrs, num_rows));

    // Init arrays
    array<IndexType> l_col_idxs{exec, l_nnz};
    array<ValueType> l_vals{exec, l_nnz};
    std::shared_ptr<matrix_type> l_factor = matrix_type::create(
        exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
        std::move(l_row_ptrs), parameters_.l_strategy);
    array<IndexType> u_col_idxs{exec, u_nnz};
    array<ValueType> u_vals{exec, u_nnz};
    std::shared_ptr<matrix_type> u_factor = matrix_type::create(
        exec, matrix_size, std::move(u_vals), std::move(u_col_idxs),
        std::move(u_row_ptrs), parameters_.u_strategy);

    // Separate L and U: columns and values
    exec->run(ilu_factorization::make_initialize_l_u(
        local_system_matrix.get(), l_factor.get(), u_factor.get()));

    return Composition<ValueType>::create(std::move(l_factor),
                                          std::move(u_factor));
}


#define GKO_DECLARE_ILU(ValueType, IndexType) class Ilu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU);


}  // namespace factorization
}  // namespace gko
