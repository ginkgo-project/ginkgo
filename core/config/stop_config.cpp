// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/config/stop_config.hpp"

#include <string>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/registry_accessor.hpp"
#include "core/config/type_descriptor_helper.hpp"

namespace gko {
namespace config {


deferred_factory_parameter<stop::CriterionFactory> configure_time(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto params = stop::Time::build();
    config_check_decorator config_check(config);
    if (auto& obj = config_check.get("time_limit")) {
        params.with_time_limit(get_value<long long int>(obj));
    }
    return params;
}


deferred_factory_parameter<stop::CriterionFactory> configure_iter(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto params = stop::Iteration::build();
    config_check_decorator config_check(config);
    if (auto& obj = config_check.get("max_iters")) {
        params.with_max_iters(get_value<size_type>(obj));
    }
    return params;
}


inline stop::mode get_mode(const std::string& str)
{
    if (str == "absolute") {
        return stop::mode::absolute;
    } else if (str == "initial_resnorm") {
        return stop::mode::initial_resnorm;
    } else if (str == "rhs_norm") {
        return stop::mode::rhs_norm;
    }
    GKO_INVALID_CONFIG_VALUE("baseline", str);
}


template <typename ValueType>
class ResidualNormConfigurer {
public:
    static deferred_factory_parameter<
        typename stop::ResidualNorm<ValueType>::Factory>
    parse(const gko::config::pnode& config,
          const gko::config::registry& context,
          const gko::config::type_descriptor& td_for_child)
    {
        config_check_decorator config_check(config);
        auto params = stop::ResidualNorm<ValueType>::build();
        if (auto& obj = config_check.get("reduction_factor")) {
            params.with_reduction_factor(
                get_value<remove_complex<ValueType>>(obj));
        }
        if (auto& obj = config_check.get("baseline")) {
            params.with_baseline(get_mode(obj.get_string()));
        }
        return params;
    }
};


deferred_factory_parameter<stop::CriterionFactory> configure_residual(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<stop::CriterionFactory, ResidualNormConfigurer>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()));
}


template <typename ValueType>
class ImplicitResidualNormConfigurer {
public:
    static deferred_factory_parameter<
        typename stop::ImplicitResidualNorm<ValueType>::Factory>
    parse(const gko::config::pnode& config,
          const gko::config::registry& context,
          const gko::config::type_descriptor& td_for_child)
    {
        config_check_decorator config_check(config);
        auto params = stop::ImplicitResidualNorm<ValueType>::build();
        if (auto& obj = config_check.get("reduction_factor")) {
            params.with_reduction_factor(
                get_value<remove_complex<ValueType>>(obj));
        }
        if (auto& obj = config_check.get("baseline")) {
            params.with_baseline(get_mode(obj.get_string()));
        }
        return params;
    }
};


deferred_factory_parameter<stop::CriterionFactory> configure_implicit_residual(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<stop::CriterionFactory, ImplicitResidualNormConfigurer>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()));
}


}  // namespace config
}  // namespace gko
