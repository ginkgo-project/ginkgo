// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/factorization/ilu.hpp"

#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/factorization/lu_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace factorization {
namespace ilu_factorization {
namespace {


GKO_REGISTER_OPERATION(sparselib_ilu, ilu_factorization::sparselib_ilu);
GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, factorization::initialize_l_u);
// for gko syncfree implementation
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(initialize, lu_factorization::initialize);
GKO_REGISTER_OPERATION(factorize, lu_factorization::factorize);


}  // anonymous namespace
}  // namespace ilu_factorization


template <typename ValueType, typename IndexType>
typename Ilu<ValueType, IndexType>::parameters_type
Ilu<ValueType, IndexType>::parse(const config::pnode& config,
                                 const config::registry& context,
                                 const config::type_descriptor& td_for_child)
{
    auto params = factorization::Ilu<ValueType, IndexType>::build();
    if (auto& obj = config.get("l_strategy")) {
        params.with_l_strategy(config::get_strategy<matrix_type>(obj));
    }
    if (auto& obj = config.get("u_strategy")) {
        params.with_u_strategy(config::get_strategy<matrix_type>(obj));
    }
    if (auto& obj = config.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    if (auto& obj = config.get("algorithm")) {
        using gko::factorization::incomplete_algorithm;
        auto str = obj.get_string();
        if (str == "sparselib") {
            params.with_algorithm(incomplete_algorithm::sparselib);
        } else if (str == "syncfree") {
            params.with_algorithm(incomplete_algorithm::syncfree);
        } else {
            GKO_INVALID_CONFIG_VALUE("algorithm", str);
        }
    }
    return params;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> Ilu<ValueType, IndexType>::generate_l_u(
    const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto local_system_matrix = share(matrix_type::create(exec));
    as<ConvertibleTo<matrix_type>>(system_matrix.get())
        ->convert_to(local_system_matrix);

    if (!skip_sorting) {
        local_system_matrix->sort_by_column_index();
    }

    // Add explicit diagonal zero elements if they are missing
    exec->run(ilu_factorization::make_add_diagonal_elements(
        local_system_matrix.get(), false));

    std::shared_ptr<const matrix_type> ilu;
    // Compute ILU factorization
    if (parameters_.algorithm == incomplete_algorithm::syncfree ||
        (!std::dynamic_pointer_cast<const ReferenceExecutor>(exec) &&
         exec == exec->get_master())) {
        const auto nnz = local_system_matrix->get_num_stored_elements();
        const auto num_rows = local_system_matrix->get_size()[0];
        auto factors = share(
            matrix_type::create(exec, local_system_matrix->get_size(), nnz));
        exec->copy_from(exec, nnz, local_system_matrix->get_const_col_idxs(),
                        factors->get_col_idxs());
        exec->copy_from(exec, num_rows + 1,
                        local_system_matrix->get_const_row_ptrs(),
                        factors->get_row_ptrs());
        // update srow to be safe
        factors->set_strategy(factors->get_strategy());

        // setup lookup structure on factors
        const auto lookup = matrix::csr::build_lookup(factors.get());
        array<IndexType> diag_idxs{exec, num_rows};
        exec->run(ilu_factorization::make_initialize(
            local_system_matrix.get(), lookup.storage_offsets.get_const_data(),
            lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
            diag_idxs.get_data(), factors.get()));
        // run numerical factorization
        array<int> tmp{exec};
        exec->run(ilu_factorization::make_factorize(
            lookup.storage_offsets.get_const_data(),
            lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
            diag_idxs.get_const_data(), factors.get(), false, tmp));
        ilu = factors;
    } else {
        exec->run(
            ilu_factorization::make_sparselib_ilu(local_system_matrix.get()));
        ilu = local_system_matrix;
    }
    // Separate L and U factors: nnz
    const auto matrix_size = ilu->get_size();
    const auto num_rows = matrix_size[0];
    array<IndexType> l_row_ptrs{exec, num_rows + 1};
    array<IndexType> u_row_ptrs{exec, num_rows + 1};
    exec->run(ilu_factorization::make_initialize_row_ptrs_l_u(
        ilu.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));

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
    exec->run(ilu_factorization::make_initialize_l_u(ilu.get(), l_factor.get(),
                                                     u_factor.get()));

    return Composition<ValueType>::create(std::move(l_factor),
                                          std::move(u_factor));
}


#define GKO_DECLARE_ILU(ValueType, IndexType) class Ilu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU);


}  // namespace factorization
}  // namespace gko
