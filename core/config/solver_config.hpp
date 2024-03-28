// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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


template <typename SolverFactory>
inline void common_solver_configure(SolverFactory& params, const pnode& config,
                                    const registry& context,
                                    type_descriptor td_for_child)
{
    if (auto& obj = config.get("generated_preconditioner")) {
        params.with_generated_preconditioner(
            gko::config::get_stored_obj<const LinOp>(obj, context));
    }
    if (auto& obj = config.get("criteria")) {
        params.with_criteria(
            gko::config::build_or_get_factory_vector<
                const stop::CriterionFactory>(obj, context, td_for_child));
    }
    if (auto& obj = config.get("preconditioner")) {
        params.with_preconditioner(
            gko::config::build_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
}


}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_CONFIG_SOLVER_CONFIG_HPP_
