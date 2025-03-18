// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_TRISOLVER_CONFIG_HPP_
#define GKO_CORE_CONFIG_TRISOLVER_CONFIG_HPP_

#include <set>
#include <string>

#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"

namespace gko {
namespace config {


template <typename SolverParam>
inline void common_trisolver_parse(SolverParam& params, const pnode& config,
                                   const registry& context,
                                   type_descriptor td_for_child,
                                   std::set<std::string>& allowed_keys)
{
    if (auto& obj = get_config_node(config, "num_rhs", allowed_keys)) {
        params.with_num_rhs(gko::config::get_value<size_type>(obj));
    }
    if (auto& obj = get_config_node(config, "unit_diagonal", allowed_keys)) {
        params.with_unit_diagonal(gko::config::get_value<bool>(obj));
    }
    if (auto& obj = get_config_node(config, "algorithm", allowed_keys)) {
        using gko::solver::trisolve_algorithm;
        auto str = obj.get_string();
        if (str == "sparselib") {
            params.with_algorithm(trisolve_algorithm::sparselib);
        } else if (str == "syncfree") {
            params.with_algorithm(trisolve_algorithm::syncfree);
        } else {
            GKO_INVALID_CONFIG_VALUE("algorithm", str);
        }
    }
}


}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_CONFIG_TRISOLVER_CONFIG_HPP_
