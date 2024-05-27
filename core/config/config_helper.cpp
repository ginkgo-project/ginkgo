// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/registry.hpp>


#include "core/config/config_helper.hpp"
#include "core/config/registry_accessor.hpp"
#include "core/config/stop_config.hpp"

namespace gko {
namespace config {


template <>
deferred_factory_parameter<const LinOpFactory>
parse_or_get_factory<const LinOpFactory>(const pnode& config,
                                         const registry& context,
                                         const type_descriptor& td)
{
    if (config.get_tag() == pnode::tag_t::string) {
        return detail::registry_accessor::get_data<LinOpFactory>(
            context, config.get_string());
    } else if (config.get_tag() == pnode::tag_t::map) {
        return parse(config, context, td);
    } else {
        GKO_INVALID_STATE("The data of config is not valid.");
    }
}


template <>
deferred_factory_parameter<const stop::CriterionFactory>
parse_or_get_factory<const stop::CriterionFactory>(const pnode& config,
                                                   const registry& context,
                                                   const type_descriptor& td)
{
    if (config.get_tag() == pnode::tag_t::string) {
        return detail::registry_accessor::get_data<stop::CriterionFactory>(
            context, config.get_string());
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
        return criterion_map.at(config.get("type").get_string())(config,
                                                                 context, td);
    } else {
        GKO_INVALID_STATE("The data of config is not valid.");
    }
}

}  // namespace config
}  // namespace gko
