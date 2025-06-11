// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/factorization/ic.hpp"

#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/cholesky.hpp>

#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ic_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace factorization {
namespace ic_factorization {
namespace {


GKO_REGISTER_OPERATION(sparselib_ic, ic_factorization::sparselib_ic);
GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l,
                       factorization::initialize_row_ptrs_l);
GKO_REGISTER_OPERATION(initialize_l, factorization::initialize_l);
// for gko syncfree implementation
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(from_factor, elimination_forest::from_factor);
GKO_REGISTER_OPERATION(initialize, cholesky::initialize);
GKO_REGISTER_OPERATION(factorize, cholesky::factorize);


}  // anonymous namespace
}  // namespace ic_factorization


template <typename ValueType, typename IndexType>
typename Ic<ValueType, IndexType>::parameters_type
Ic<ValueType, IndexType>::parse(const config::pnode& config,
                                const config::registry& context,
                                const config::type_descriptor& td_for_child)
{
    auto params = factorization::Ic<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("l_strategy")) {
        params.with_l_strategy(config::get_strategy<matrix_type>(obj));
    }
    if (auto& obj = config_check.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    if (auto& obj = config_check.get("both_factors")) {
        params.with_both_factors(config::get_value<bool>(obj));
    }
    if (auto& obj = config_check.get("algorithm")) {
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
std::unique_ptr<Composition<ValueType>> Ic<ValueType, IndexType>::generate(
    const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting,
    bool both_factors) const
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
    exec->run(ic_factorization::make_add_diagonal_elements(
        local_system_matrix.get(), false));

    std::shared_ptr<const matrix_type> ic;
    // Compute IC factorization
    if (parameters_.algorithm == incomplete_algorithm::syncfree ||
        (!std::dynamic_pointer_cast<const ReferenceExecutor>(exec) &&
         exec == exec->get_master())) {
        std::unique_ptr<gko::factorization::elimination_forest<IndexType>>
            forest;
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
        forest =
            std::make_unique<gko::factorization::elimination_forest<IndexType>>(
                exec, num_rows);
        exec->run(ic_factorization::make_from_factor(factors.get(), *forest));

        // setup lookup structure on factors
        const auto lookup = matrix::csr::build_lookup(factors.get());
        array<IndexType> diag_idxs{exec, num_rows};
        array<IndexType> transpose_idxs{exec,
                                        factors->get_num_stored_elements()};
        // initialize factors
        exec->run(ic_factorization::make_fill_array(
            factors->get_values(), factors->get_num_stored_elements(),
            zero<ValueType>()));
        exec->run(ic_factorization::make_initialize(
            local_system_matrix.get(), lookup.storage_offsets.get_const_data(),
            lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
            diag_idxs.get_data(), transpose_idxs.get_data(), factors.get()));
        // run numerical factorization
        array<int> tmp{exec};
        exec->run(ic_factorization::make_factorize(
            lookup.storage_offsets.get_const_data(),
            lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
            diag_idxs.get_const_data(), transpose_idxs.get_const_data(),
            *forest, factors.get(), false, tmp));
        ic = factors;
    } else {
        exec->run(
            ic_factorization::make_sparselib_ic(local_system_matrix.get()));
        ic = local_system_matrix;
    }

    // Extract lower factor: compute non-zeros
    const auto matrix_size = ic->get_size();
    const auto num_rows = matrix_size[0];
    array<IndexType> l_row_ptrs{exec, num_rows + 1};
    exec->run(ic_factorization::make_initialize_row_ptrs_l(
        ic.get(), l_row_ptrs.get_data()));

    // Get nnz from device memory
    auto l_nnz = static_cast<size_type>(get_element(l_row_ptrs, num_rows));

    // Init arrays
    array<IndexType> l_col_idxs{exec, l_nnz};
    array<ValueType> l_vals{exec, l_nnz};
    std::shared_ptr<matrix_type> l_factor = matrix_type::create(
        exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
        std::move(l_row_ptrs), parameters_.l_strategy);

    // Extract lower factor: columns and values
    exec->run(
        ic_factorization::make_initialize_l(ic.get(), l_factor.get(), false));

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
