// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/factorization/lu.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>

#include "core/components/fill_array_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace experimental {
namespace factorization {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(initialize, lu_factorization::initialize);
GKO_REGISTER_OPERATION(factorize, lu_factorization::factorize);
GKO_REGISTER_HOST_OPERATION(symbolic_cholesky,
                            gko::factorization::symbolic_cholesky);
GKO_REGISTER_HOST_OPERATION(symbolic_lu, gko::factorization::symbolic_lu);
GKO_REGISTER_HOST_OPERATION(symbolic_lu_near_symm,
                            gko::factorization::symbolic_lu_near_symm);


}  // namespace


template <typename ValueType, typename IndexType>
typename Lu<ValueType, IndexType>::parameters_type
Lu<ValueType, IndexType>::parse(const config::pnode& config,
                                const config::registry& context,
                                const config::type_descriptor& td_for_child)
{
    auto params =
        experimental::factorization::Lu<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("symbolic_factorization")) {
        params.with_symbolic_factorization(
            config::get_stored_obj<const sparsity_pattern_type>(obj, context));
    }
    if (auto& obj = config_check.get("symbolic_algorithm")) {
        auto str = obj.get_string();
        if (str == "general") {
            params.with_symbolic_algorithm(symbolic_type::general);
        } else if (str == "near_symmetric") {
            params.with_symbolic_algorithm(symbolic_type::near_symmetric);
        } else if (str == "symmetric") {
            params.with_symbolic_algorithm(symbolic_type::symmetric);
        } else {
            GKO_INVALID_CONFIG_VALUE("symbolic_type", str);
        }
    }
    if (auto& obj = config_check.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }

    return params;
}


template <typename ValueType, typename IndexType>
class Lu<ValueType, IndexType>::LuReuseData : public BaseReuseData {
    friend class Lu;
    friend class Factory;

public:
    LuReuseData() = default;

    bool is_empty() const { return lookup_data_ == nullptr; }

private:
    std::unique_ptr<matrix::csr::lookup_data<IndexType>> lookup_data_;
    std::unique_ptr<matrix::SparsityCsr<ValueType, IndexType>>
        symbolic_factors_;
    // diag_idxs are cheap to recompute, ignore them
};


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOpFactory::ReuseData>
Lu<ValueType, IndexType>::create_empty_reuse_data() const
{
    return std::make_unique<LuReuseData>();
}


template <typename ValueType, typename IndexType>
void Lu<ValueType, IndexType>::check_reuse_consistent(
    const LinOp* input, BaseReuseData& reuse_data) const
{
    auto& lrd = *as<LuReuseData>(&reuse_data);
    if (lrd.is_empty()) {
        return;
    }
    GKO_ASSERT_IS_SQUARE_MATRIX(input);
}


template <typename ValueType, typename IndexType>
Lu<ValueType, IndexType>::Lu(std::shared_ptr<const Executor> exec,
                             const parameters_type& params)
    : EnablePolymorphicObject<Lu, LinOpFactory>(std::move(exec)),
      parameters_(params)
{}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Lu<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    return as<factorization_type>(
        this->LinOpFactory::generate(std::move(system_matrix)));
}


template <typename ValueType, typename IndexType>
auto Lu<ValueType, IndexType>::generate_reuse(
    std::shared_ptr<const LinOp> input, BaseReuseData& reuse_data) const
    -> std::unique_ptr<factorization_type>
{
    return as<factorization_type>(
        LinOpFactory::generate_reuse(input, reuse_data));
}


template <typename ValueType, typename IndexType>
static std::unique_ptr<matrix::Csr<ValueType, IndexType>> symbolic_factorize(
    symbolic_type algorithm, const matrix::Csr<ValueType, IndexType>* mtx)
{
    auto exec = mtx->get_executor();
    std::unique_ptr<matrix::Csr<ValueType, IndexType>> factors;
    switch (algorithm) {
    case symbolic_type::general:
        exec->run(make_symbolic_lu(mtx, factors));
        break;
    case symbolic_type::near_symmetric:
        exec->run(make_symbolic_lu_near_symm(mtx, factors));
        break;
    case symbolic_type::symmetric: {
        std::unique_ptr<gko::factorization::elimination_forest<IndexType>>
            forest;
        exec->run(make_symbolic_cholesky(mtx, true, factors, forest));
        break;
    }
    default:
        GKO_INVALID_STATE("Invalid symbolic factorization algorithm");
    }
    return factors;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Lu<ValueType, IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();
    const auto mtx = copy_and_convert_to<matrix_type>(exec, system_matrix);
    const auto num_rows = mtx->get_size()[0];
    std::unique_ptr<matrix_type> factors;
    if (!parameters_.symbolic_factorization) {
        factors = symbolic_factorize(parameters_.symbolic_algorithm, mtx.get());
    } else {
        const auto& symbolic = parameters_.symbolic_factorization;
        const auto factor_nnz = symbolic->get_num_nonzeros();
        factors = matrix_type::create(exec, mtx->get_size(), factor_nnz);
        const auto symbolic_exec = symbolic->get_executor();
        exec->copy_from(symbolic_exec, factor_nnz,
                        symbolic->get_const_col_idxs(),
                        factors->get_col_idxs());
        exec->copy_from(symbolic_exec, num_rows + 1,
                        symbolic->get_const_row_ptrs(),
                        factors->get_row_ptrs());
        // update srow to be safe
        factors->set_strategy(factors->get_strategy());
    }
    // setup lookup structure on factors
    const auto lookup = matrix::csr::build_lookup(factors.get());
    array<IndexType> diag_idxs{exec, num_rows};
    exec->run(make_initialize(
        mtx.get(), lookup.storage_offsets.get_const_data(),
        lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
        diag_idxs.get_data(), factors.get()));
    // run numerical factorization
    array<int> tmp{exec};
    exec->run(make_factorize(
        lookup.storage_offsets.get_const_data(),
        lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
        diag_idxs.get_const_data(), factors.get(), true, tmp));
    return factorization_type::create_from_combined_lu(std::move(factors));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Lu<ValueType, IndexType>::generate_reuse_impl(
    std::shared_ptr<const LinOp> system_matrix, BaseReuseData& reuse_data) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();
    const auto mtx = copy_and_convert_to<matrix_type>(exec, system_matrix);
    const auto num_rows = mtx->get_size()[0];
    auto& lurd = *as<LuReuseData>(&reuse_data);
    if (lurd.is_empty()) {
        if (!parameters_.symbolic_factorization) {
            auto tmp_factors =
                symbolic_factorize(parameters_.symbolic_algorithm, mtx.get());
            lurd.symbolic_factors_ = sparsity_pattern_type::create(exec);
            tmp_factors->move_to(lurd.symbolic_factors_);
        } else {
            lurd.symbolic_factors_ =
                parameters_.symbolic_factorization->clone();
        }
        // setup lookup structure on factors
        lurd.lookup_data_ =
            std::make_unique<matrix::csr::lookup_data<IndexType>>(
                matrix::csr::build_lookup(lurd.symbolic_factors_.get()));
    }
    auto pattern = lurd.symbolic_factors_.get();
    auto pattern_nnz = pattern->get_num_nonzeros();
    auto factors = matrix_type::create(
        exec, mtx->get_size(), array<ValueType>{exec, pattern_nnz},
        make_array_view(exec, pattern_nnz, pattern->get_col_idxs()),
        make_array_view(exec, num_rows + 1, pattern->get_row_ptrs()));
    auto& lookup = *lurd.lookup_data_;
    array<IndexType> diag_idxs{exec, num_rows};
    exec->run(make_initialize(
        mtx.get(), lookup.storage_offsets.get_const_data(),
        lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
        diag_idxs.get_data(), factors.get()));
    // run numerical factorization
    array<int> tmp{exec};
    exec->run(make_factorize(
        lookup.storage_offsets.get_const_data(),
        lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
        diag_idxs.get_const_data(), factors.get(), true, tmp));
    return factorization_type::create_from_combined_lu(std::move(factors));
}


#define GKO_DECLARE_LU(ValueType, IndexType) class Lu<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU);


}  // namespace factorization
}  // namespace experimental
}  // namespace gko
