// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


inline deferred_factory_parameter<stop::CriterionFactory> configure_time(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto factory = stop::Time::build();
    if (auto& obj = config.get("time_limit")) {
        factory.with_time_limit(gko::config::get_value<long long int>(obj));
    }
    return factory;
}


inline deferred_factory_parameter<stop::CriterionFactory> configure_iter(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto factory = stop::Iteration::build();
    if (auto& obj = config.get("max_iters")) {
        factory.with_max_iters(gko::config::get_value<size_type>(obj));
    }
    return factory;
}


stop::mode get_mode(const std::string& str)
{
    if (str == "absolute") {
        return stop::mode::absolute;
    } else if (str == "initial_resnorm") {
        return stop::mode::initial_resnorm;
    } else if (str == "rhs_norm") {
        return stop::mode::rhs_norm;
    }
    GKO_INVALID_STATE("Not valid " + str);
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
        auto params = stop::ResidualNorm<ValueType>::build();
        if (auto& obj = config.get("reduction_factor")) {
            params.with_reduction_factor(
                gko::config::get_value<remove_complex<ValueType>>(obj));
        }
        if (auto& obj = config.get("baseline")) {
            params.with_baseline(get_mode(obj.get_string()));
        }
        return params;
    }
};


inline deferred_factory_parameter<stop::CriterionFactory> configure_residual(
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
        auto params = stop::ImplicitResidualNorm<ValueType>::build();
        if (auto& obj = config.get("reduction_factor")) {
            params.with_reduction_factor(
                gko::config::get_value<remove_complex<ValueType>>(obj));
        }
        if (auto& obj = config.get("baseline")) {
            params.with_baseline(get_mode(obj.get_string()));
        }
        return params;
    }
};


inline deferred_factory_parameter<stop::CriterionFactory>
configure_implicit_residual(const pnode& config, const registry& context,
                            const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<stop::CriterionFactory, ImplicitResidualNormConfigurer>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()));
}


template <>
deferred_factory_parameter<const stop::CriterionFactory>
get_factory<const stop::CriterionFactory>(const pnode& config,
                                          const registry& context,
                                          const type_descriptor& td)
{
    deferred_factory_parameter<const stop::CriterionFactory> ptr;
    if (config.get_tag() == pnode::tag_t::string) {
        return context.search_data<stop::CriterionFactory>(config.get_string());
    } else if (config.get_tag() == pnode::tag_t::map) {
        static std::map<std::string,
                        std::function<deferred_factory_parameter<
                            gko::stop::CriterionFactory>(
                            const pnode&, const registry&, type_descriptor)>>
            criterion_map{
                {{"Time", configure_time},
                 {"Iteration", configure_iter},
                 {"ResidualNorm", configure_residual},
                 {"ImplicitResidualNorm", configure_implicit_residual}}};
        return criterion_map.at(config.get("Type").get_string())(config,
                                                                 context, td);
    }
    GKO_THROW_IF_INVALID(!ptr.is_empty(), "Parse get nullptr in the end");
    return ptr;
}


}  // namespace config
}  // namespace gko
