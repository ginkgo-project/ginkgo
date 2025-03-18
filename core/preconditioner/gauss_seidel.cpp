// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/preconditioner/gauss_seidel.hpp"

#include <set>
#include <string>

#include <ginkgo/core/preconditioner/sor.hpp>

#include "core/config/config_helper.hpp"


namespace gko {
namespace preconditioner {


template <typename ValueType, typename IndexType>
typename GaussSeidel<ValueType, IndexType>::parameters_type
GaussSeidel<ValueType, IndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = GaussSeidel::build();
    std::set<std::string> allowed_keys;
    if (auto& obj =
            config::get_config_node(config, "skip_sorting", allowed_keys)) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    if (auto& obj =
            config::get_config_node(config, "symmetric", allowed_keys)) {
        params.with_symmetric(config::get_value<bool>(obj));
    }
    if (auto& obj = config::get_config_node(config, "l_solver", allowed_keys)) {
        params.with_l_solver(config::parse_or_get_factory<const LinOpFactory>(
            obj, context, td_for_child));
    }
    if (auto& obj = config::get_config_node(config, "u_solver", allowed_keys)) {
        params.with_u_solver(config::parse_or_get_factory<const LinOpFactory>(
            obj, context, td_for_child));
    }
    config::check_allowed_keys(config, allowed_keys);

    return params;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename GaussSeidel<ValueType, IndexType>::composition_type>
GaussSeidel<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product =
        std::unique_ptr<composition_type>(static_cast<composition_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> GaussSeidel<ValueType, IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    return Sor<ValueType, IndexType>::build()
        .with_skip_sorting(parameters_.skip_sorting)
        .with_symmetric(parameters_.symmetric)
        .with_relaxation_factor(static_cast<remove_complex<ValueType>>(1.0))
        .with_l_solver(parameters_.l_solver)
        .with_u_solver(parameters_.u_solver)
        .on(this->get_executor())
        ->generate(std::move(system_matrix));
}


#define GKO_DECLARE_GAUSS_SEIDEL(ValueType, IndexType) \
    class GaussSeidel<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GAUSS_SEIDEL);


}  // namespace preconditioner
}  // namespace gko
