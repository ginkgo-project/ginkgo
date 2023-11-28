// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
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
    SET_POINTER(factory, const LinOp, generated_preconditioner, config, context,
                td_for_child);
    SET_FACTORY_VECTOR(factory, const stop::CriterionFactory, criteria, config,
                       context, td_for_child);
    SET_FACTORY(factory, const LinOpFactory, preconditioner, config, context,
                td_for_child);
}


}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_CONFIG_SOLVER_CONFIG_HPP_
