// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/factorization/cholesky.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>

#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"

namespace gko {
namespace experimental {
namespace factorization {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(from_factor, elimination_forest::from_factor);
GKO_REGISTER_OPERATION(initialize, cholesky::initialize);
GKO_REGISTER_OPERATION(factorize, cholesky::factorize);


}  // namespace


template <typename ValueType, typename IndexType>
typename Cholesky<ValueType, IndexType>::parameters_type
Cholesky<ValueType, IndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Cholesky<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("symbolic_factorization")) {
        params.with_symbolic_factorization(
            config::get_stored_obj<const sparsity_pattern_type>(obj, context));
    }
    if (auto& obj = config_check.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    return params;
}


template <typename ValueType, typename IndexType>
Cholesky<ValueType, IndexType>::Cholesky(std::shared_ptr<const Executor> exec,
                                         const parameters_type& params)
    : EnablePolymorphicObject<Cholesky, LinOpFactory>(std::move(exec)),
      parameters_(params)
{}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Cholesky<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product =
        std::unique_ptr<factorization_type>(static_cast<factorization_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Cholesky<ValueType, IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();
    const auto mtx = copy_and_convert_to<matrix_type>(exec, system_matrix);
    const auto num_rows = mtx->get_size()[0];
    std::unique_ptr<matrix_type> factors;
    std::unique_ptr<gko::factorization::elimination_forest<IndexType>> forest;
    if (!parameters_.symbolic_factorization) {
        gko::factorization::symbolic_cholesky(mtx.get(), true, factors, forest);
    } else {
        const auto& symbolic = parameters_.symbolic_factorization;
        const auto factor_nnz = symbolic->get_num_nonzeros();
        factors = matrix_type::create(exec, mtx->get_size(), factor_nnz);
        const auto symbolic_exec = symbolic->get_executor();
        exec->copy_from(symbolic_exec.get(), factor_nnz,
                        symbolic->get_const_col_idxs(),
                        factors->get_col_idxs());
        exec->copy_from(symbolic_exec.get(), num_rows + 1,
                        symbolic->get_const_row_ptrs(),
                        factors->get_row_ptrs());
        // update srow to be safe
        factors->set_strategy(factors->get_strategy());
        forest =
            std::make_unique<gko::factorization::elimination_forest<IndexType>>(
                exec, num_rows);
        exec->run(make_from_factor(factors.get(), *forest));
    }
    // setup lookup structure on factors
    const auto lookup = matrix::csr::build_lookup(factors.get());
    array<IndexType> diag_idxs{exec, num_rows};
    array<IndexType> transpose_idxs{exec, factors->get_num_stored_elements()};
    // initialize factors
    exec->run(make_fill_array(factors->get_values(),
                              factors->get_num_stored_elements(),
                              zero<ValueType>()));
    exec->run(make_initialize(
        mtx.get(), lookup.storage_offsets.get_const_data(),
        lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
        diag_idxs.get_data(), transpose_idxs.get_data(), factors.get()));
    // run numerical factorization
    array<int> tmp{exec};
    exec->run(make_factorize(
        lookup.storage_offsets.get_const_data(),
        lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
        diag_idxs.get_const_data(), transpose_idxs.get_const_data(), *forest,
        factors.get(), true, tmp));
    return factorization_type::create_from_combined_cholesky(
        std::move(factors));
}


#define GKO_DECLARE_CHOLESKY(ValueType, IndexType) \
    class Cholesky<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY);


}  // namespace factorization
}  // namespace experimental
}  // namespace gko
