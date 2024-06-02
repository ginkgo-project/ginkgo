// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_STOP_CONFIG_HPP_
#define GKO_CORE_CONFIG_STOP_CONFIG_HPP_


#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace config {


deferred_factory_parameter<stop::CriterionFactory> configure_time(
    const pnode& config, const registry& context, const type_descriptor& td);

deferred_factory_parameter<stop::CriterionFactory> configure_iter(
    const pnode& config, const registry& context, const type_descriptor& td);

deferred_factory_parameter<stop::CriterionFactory> configure_residual(
    const pnode& config, const registry& context, const type_descriptor& td);

deferred_factory_parameter<stop::CriterionFactory> configure_implicit_residual(
    const pnode& config, const registry& context, const type_descriptor& td);

}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_STOP_CONFIG_HPP_
