// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/ic.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/array_access.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ic_kernels.hpp"


namespace gko {
namespace factorization {
namespace ic_factorization {
namespace {


GKO_REGISTER_OPERATION(compute, ic_factorization::compute);
GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l,
                       factorization::initialize_row_ptrs_l);
GKO_REGISTER_OPERATION(initialize_l, factorization::initialize_l);


}  // anonymous namespace
}  // namespace ic_factorization


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> Ic<ValueType, IndexType>::generate(
    const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
    bool both_factors) const
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
    exec->run(ic_factorization::make_add_diagonal_elements(
        local_system_matrix.get(), false));

    // Compute LC factorization
    exec->run(ic_factorization::make_compute(local_system_matrix.get()));

    // Extract lower factor: compute non-zeros
    const auto matrix_size = local_system_matrix->get_size();
    const auto num_rows = matrix_size[0];
    array<IndexType> l_row_ptrs{exec, num_rows + 1};
    exec->run(ic_factorization::make_initialize_row_ptrs_l(
        local_system_matrix.get(), l_row_ptrs.get_data()));

    // Get nnz from device memory
    auto l_nnz = static_cast<size_type>(get_element(l_row_ptrs, num_rows));

    // Init arrays
    array<IndexType> l_col_idxs{exec, l_nnz};
    array<ValueType> l_vals{exec, l_nnz};
    std::shared_ptr<matrix_type> l_factor = matrix_type::create(
        exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
        std::move(l_row_ptrs), parameters_.l_strategy);

    // Extract lower factor: columns and values
    exec->run(ic_factorization::make_initialize_l(local_system_matrix.get(),
                                                  l_factor.get(), false));

    if (both_factors) {
        auto lh_factor = l_factor->conj_transpose();
        return Composition<ValueType>::create(std::move(l_factor),
                                              std::move(lh_factor));
    } else {
        return Composition<ValueType>::create(std::move(l_factor));
    }
}


#define GKO_DECLARE_IC(ValueType, IndexType) class Ic<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IC);


}  // namespace factorization
}  // namespace gko
