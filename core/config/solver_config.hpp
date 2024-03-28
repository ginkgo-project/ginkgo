// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_SOLVER_CONFIG_HPP_
#define GKO_CORE_CONFIG_SOLVER_CONFIG_HPP_


#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>


#include "core/config/config.hpp"
#include "core/config/dispatch.hpp"

namespace gko {
namespace config {


template <typename SolverFactory>
inline void common_solver_configure(SolverFactory& factory, const pnode& config,
                                    const registry& context,
                                    type_descriptor td_for_child)
{
    if (auto& obj = config.get("generated_preconditioner")) {
        factory.with_generated_preconditioner(
            gko::config::get_pointer<const LinOp>(obj, context, td_for_child));
    }
    if (auto& obj = config.get("criteria")) {
        factory.with_criteria(
            gko::config::get_factory_vector<const stop::CriterionFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config.get("preconditioner")) {
        factory.with_preconditioner(
            gko::config::get_factory<const LinOpFactory>(obj, context,
                                                         td_for_child));
    }
}


}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_CONFIG_SOLVER_CONFIG_HPP_
