// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_SOLVER_CONFIG_HPP_
#define GKO_CORE_CONFIG_SOLVER_CONFIG_HPP_

#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"

namespace gko {
namespace config {


template <typename SolverParam>
inline void common_solver_parse(SolverParam& params,
                                config_check_decorator& config_check,
                                const registry& context,
                                type_descriptor td_for_child)
{
    if (auto& obj = config_check.get("generated_preconditioner")) {
        params.with_generated_preconditioner(
            gko::config::get_stored_obj<const LinOp>(obj, context));
    }
    if (auto& obj = config_check.get("criteria")) {
        params.with_criteria(
            gko::config::parse_or_get_criteria(obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("preconditioner")) {
        params.with_preconditioner(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
}


}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_CONFIG_SOLVER_CONFIG_HPP_
