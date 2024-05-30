// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_TRISOLVER_CONFIG_HPP_
#define GKO_CORE_CONFIG_TRISOLVER_CONFIG_HPP_


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
                                   type_descriptor td_for_child)
{
    if (auto& obj = config.get("num_rhs")) {
        params.with_num_rhs(gko::config::get_value<size_type>(obj));
    }
    if (auto& obj = config.get("unit_diagonal")) {
        params.with_unit_diagonal(gko::config::get_value<bool>(obj));
    }
    if (auto& obj = config.get("algorithm")) {
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
