/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_SOLVER_MULTIGRID_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_SOLVER_MULTIGRID_HPP_


#include <ginkgo/core/solver/multigrid.hpp>


#include "resource_manager/base/function_map.hpp"
#include "resource_manager/base/generic_constructor.hpp"
#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"
#include "resource_manager/base/type_default.hpp"
#include "resource_manager/base/type_pack.hpp"
#include "resource_manager/base/type_resolving.hpp"
#include "resource_manager/base/type_string.hpp"
#include "resource_manager/base/types.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


template <>
struct Generic<typename gko::solver::Multigrid::Factory,
               gko::solver::Multigrid> {
    using type = std::shared_ptr<typename gko::solver::Multigrid::Factory>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY(gko::solver::Multigrid, manager, item, exec, linop);
            SET_POINTER_VECTOR(const stop::CriterionFactory, criteria);
            SET_POINTER_VECTOR(const gko::LinOpFactory, mg_level);
            SET_FUNCTION(
                std::function<size_type(const size_type, const LinOp*)>,
                level_selector);
            SET_POINTER_VECTOR(const LinOpFactory, pre_smoother);
            SET_POINTER_VECTOR(const LinOpFactory, post_smoother);
            SET_POINTER_VECTOR(const LinOpFactory, mid_smoother);
            SET_VALUE(bool, post_uses_pre);
            SET_VALUE(gko::solver::multigrid::mid_smooth_type, mid_case);
            SET_VALUE(size_type, max_levels);
            SET_VALUE(size_type, min_coarse_rows);
            SET_POINTER_VECTOR(const LinOpFactory, coarsest_solver);
            SET_FUNCTION(
                std::function<size_type(const size_type, const LinOp*)>,
                solver_selector);
            SET_VALUE(gko::solver::multigrid::cycle, cycle);
            SET_VALUE(size_type, kcycle_base);
            SET_VALUE(double, kcycle_rel_tol);
            SET_VALUE(std::complex<double>, smoother_relax);
            SET_VALUE(size_type, smoother_iters);
            SET_VALUE(bool, zero_guess);
            SET_EXECUTOR;
        }();
        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};

SIMPLE_LINOP_WITH_FACTORY_IMPL_BASE(gko::solver::Multigrid);


IMPLEMENT_BRIDGE(RM_LinOpFactory, MultigridFactory,
                 gko::solver::Multigrid::Factory);
IMPLEMENT_BRIDGE(RM_LinOp, Multigrid, gko::solver::Multigrid);


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_SOLVER_MULTIGRID_HPP_
