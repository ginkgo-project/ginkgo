/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/multigrid/fixed_coarsening.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/solver/multigrid.hpp>

#include "core/config/config.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace config {


template <typename ValueType, typename IndexType>
class PgmConfigurator {
public:
    static std::unique_ptr<
        typename multigrid::Pgm<ValueType, IndexType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        auto factory = multigrid::Pgm<ValueType, IndexType>::build();
        SET_VALUE(factory, unsigned, max_iterations, config);
        SET_VALUE(factory, double, max_unassigned_ratio, config);
        SET_VALUE(factory, bool, deterministic, config);
        SET_VALUE(factory, bool, skip_sorting, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Pgm>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, PgmConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


template <typename ValueType, typename IndexType>
class FixedCoarseningConfigurator {
public:
    static std::unique_ptr<
        typename multigrid::FixedCoarsening<ValueType, IndexType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        auto factory =
            multigrid::FixedCoarsening<ValueType, IndexType>::build();
        SET_VALUE_ARRAY(factory, array<IndexType>, coarse_rows, config, exec);
        SET_VALUE(factory, bool, skip_sorting, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory>
build_from_config<LinOpFactoryType::FixedCoarsening>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, FixedCoarseningConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


std::function<size_type(const size_type, const LinOp*)> get_selector(
    std::string key)
{
    static std::map<std::string,
                    std::function<size_type(const size_type, const LinOp*)>>
        selector_map{
            {{"first_for_top", [](const size_type level, const LinOp*) {
                  return (level == 0) ? 0 : 1;
              }}}};
    return selector_map.at(key);
}


class MultigridConfigurator {
public:
    static std::unique_ptr<typename solver::Multigrid::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        auto factory = solver::Multigrid::build();
        SET_POINTER_VECTOR(factory, const stop::CriterionFactory, criteria,
                           config, context, exec, td_for_child);
        SET_POINTER_VECTOR(factory, const gko::LinOpFactory, mg_level, config,
                           context, exec, td_for_child);
        if (config.contains("level_selector")) {
            auto str = config.at("level_selector").get_data<std::string>();
            factory.with_level_selector(get_selector(str));
        }
        SET_POINTER_VECTOR(factory, const LinOpFactory, pre_smoother, config,
                           context, exec, td_for_child);
        SET_POINTER_VECTOR(factory, const LinOpFactory, post_smoother, config,
                           context, exec, td_for_child);
        SET_POINTER_VECTOR(factory, const LinOpFactory, mid_smoother, config,
                           context, exec, td_for_child);
        SET_VALUE(factory, bool, post_uses_pre, config);
        if (config.contains("mid_case")) {
            auto str = config.at("mid_case").get_data<std::string>();
            if (str == "both") {
                factory.with_mid_case(solver::multigrid::mid_smooth_type::both);
            } else if (str == "post_smoother") {
                factory.with_mid_case(
                    solver::multigrid::mid_smooth_type::post_smoother);
            } else if (str == "pre_smoother") {
                factory.with_mid_case(
                    solver::multigrid::mid_smooth_type::pre_smoother);
            } else if (str == "standalone") {
                factory.with_mid_case(
                    solver::multigrid::mid_smooth_type::standalone);
            } else {
                GKO_INVALID_STATE("Not valid mid_smooth_type value");
            }
        }
        SET_VALUE(factory, size_type, max_levels, config);
        SET_VALUE(factory, size_type, min_coarse_rows, config);
        SET_POINTER_VECTOR(factory, const LinOpFactory, coarsest_solver, config,
                           context, exec, td_for_child);
        if (config.contains("solver_selector")) {
            auto str = config.at("solver_selector").get_data<std::string>();
            factory.with_solver_selector(get_selector(str));
        }
        if (config.contains("cycle")) {
            auto str = config.at("cycle").get_data<std::string>();
            if (str == "v") {
                factory.with_cycle(solver::multigrid::cycle::v);
            } else if (str == "w") {
                factory.with_cycle(solver::multigrid::cycle::w);
            } else if (str == "f") {
                factory.with_cycle(solver::multigrid::cycle::f);
            } else {
                GKO_INVALID_STATE("Not valid cycle value");
            }
        }
        SET_VALUE(factory, size_type, kcycle_base, config);
        SET_VALUE(factory, double, kcycle_rel_tol, config);
        SET_VALUE(factory, std::complex<double>, smoother_relax, config);
        SET_VALUE(factory, size_type, smoother_iters, config);
        SET_VALUE(factory, solver::initial_guess_mode, default_initial_guess,
                  config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Multigrid>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return MultigridConfigurator::build_from_config(config, context, exec,
                                                    updated);
}


}  // namespace config
}  // namespace gko
