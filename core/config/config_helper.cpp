// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/config/config_helper.hpp"

#include <type_traits>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/registry.hpp>

#include "core/config/registry_accessor.hpp"
#include "core/config/stop_config.hpp"
#include "type_descriptor_helper.hpp"

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
        GKO_INVALID_STATE("The type of config is not valid.");
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
    }

    if (config.get_tag() == pnode::tag_t::map) {
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
    }

    GKO_INVALID_STATE(
        "Criteria must either be defined as a string or an array.");
}


std::vector<deferred_factory_parameter<const stop::CriterionFactory>>
parse_minimal_criteria(const pnode& config, const registry& context,
                       const type_descriptor& td)
{
    auto map_time = [](const pnode& config, const registry& context,
                       const type_descriptor& td) {
        pnode time_config{{{"time_limit", config.get("time")}}};
        return configure_time(time_config, context, td);
    };
    auto map_iteration = [](const pnode& config, const registry& context,
                            const type_descriptor& td) {
        pnode iter_config{{{"max_iters", config.get("iteration")}}};
        return configure_iter(iter_config, context, td);
    };
    auto create_residual_mapping = [](const std::string& key,
                                      const std::string& baseline,
                                      auto configure_fn) {
        return std::make_pair(
            key, [=](const pnode& config, const registry& context,
                     const type_descriptor& td) {
                pnode res_config{{{"baseline", pnode{baseline}},
                                  {"reduction_factor", config.get(key)}}};
                return configure_fn(res_config, context, td);
            });
    };
    std::map<
        std::string,
        std::function<deferred_factory_parameter<gko::stop::CriterionFactory>(
            const pnode&, const registry&, type_descriptor)>>
        criterion_map{
            {{"time", map_time},
             {"iteration", map_iteration},
             create_residual_mapping("relative_residual_norm", "rhs_norm",
                                     configure_residual),
             create_residual_mapping("initial_residual_norm", "initial_resnorm",
                                     configure_residual),
             create_residual_mapping("absolute_residual_norm", "absolute",
                                     configure_residual),
             create_residual_mapping("relative_implicit_residual_norm",
                                     "rhs_norm", configure_implicit_residual),
             create_residual_mapping("initial_implicit_residual_norm",
                                     "initial_resnorm",
                                     configure_implicit_residual),
             create_residual_mapping("absolute_implicit_residual_norm",
                                     "absolute", configure_implicit_residual)}};

    type_descriptor updated_td = update_type(config, td);

    // Although it can still be caught in the following map, it gives consistent
    // error exception when some keys are not allowed.
    std::set<std::string> allowed_keys{"time",
                                       "iteration",
                                       "relative_residual_norm",
                                       "initial_residual_norm",
                                       "absolute_residual_norm",
                                       "relative_implicit_residual_norm",
                                       "initial_implicit_residual_norm",
                                       "absolute_implicit_residual_norm"};
    check_allowed_keys(config, allowed_keys);

    std::vector<deferred_factory_parameter<const stop::CriterionFactory>> res;
    for (const auto& it : config.get_map()) {
        if (it.first == "value_type") {
            continue;
        }
        res.emplace_back(
            criterion_map.at(it.first)(config, context, updated_td));
    }
    return res;
}


std::vector<deferred_factory_parameter<const stop::CriterionFactory>>
parse_or_get_criteria(const pnode& config, const registry& context,
                      const type_descriptor& td)
{
    if (config.get_tag() == pnode::tag_t::array ||
        (config.get_tag() == pnode::tag_t::map && config.get("type"))) {
        return parse_or_get_factory_vector<const stop::CriterionFactory>(
            config, context, td);
    }

    if (config.get_tag() == pnode::tag_t::map) {
        return parse_minimal_criteria(config, context, td);
    }

    if (config.get_tag() == pnode::tag_t::string) {
        return {detail::registry_accessor::get_data<stop::CriterionFactory>(
            context, config.get_string())};
    }

    GKO_INVALID_STATE(
        "Criteria must either be defined as a string, an array,"
        "or an map.");
}

void check_allowed_keys(const pnode& config,
                        const std::set<std::string>& allowed_keys)
{
    if (config.get_tag() != pnode::tag_t::map) {
        // we only check the key in the map
        return;
    }
    const auto& map = config.get_map();
    auto set_output = [](auto& set) {
        std::string output = "[";
        for (const auto& item : set) {
            output = output + " " + item;
        }
        output += " ]";
        return output;
    };
    for (const auto& item : map) {
        if (item.first == "value_type" || item.first == "type") {
            // We always allow value_type in any class and use type to choose
            // the class
            continue;
        }
        auto search = allowed_keys.find(item.first);
        GKO_THROW_IF_INVALID(
            search != allowed_keys.end(),
            item.first + " is not a allowed key. The allowed keys here is " +
                set_output(allowed_keys));
    }
}


const pnode& get_config_node(const pnode& config, const std::string& key,
                             std::set<std::string>& allowed_keys)
{
    allowed_keys.insert(key);
    return config.get(key);
}

}  // namespace config
}  // namespace gko
