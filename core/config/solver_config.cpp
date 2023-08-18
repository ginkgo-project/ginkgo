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
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/fcg.hpp>


#include "core/config/config.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/solver_config.hpp"

namespace gko {
namespace config {


// It can also be directly in solver (or in proteced part) if we also allow
// the executor as input there.
template <template <class> class Solver>
class solver_helper {
public:
    template <typename ValueType>
    class configurator {
    public:
        static std::unique_ptr<typename Solver<ValueType>::Factory>
        parse(const pnode& config, const registry& context,
                          std::shared_ptr<const Executor> exec,
                          type_descriptor td_for_child)
        {
            auto factory = Solver<ValueType>::build();
            common_solver_configure(factory, config, context, exec,
                                    td_for_child);
            return factory.on(exec);
        }
    };
};


template <>
std::unique_ptr<gko::LinOpFactory> parse<LinOpFactoryType::Cg>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, solver_helper<solver::Cg>::configurator>(
        updated.first, config, context, exec, updated, value_type_list());
}

template <>
std::unique_ptr<gko::LinOpFactory> parse<LinOpFactoryType::Bicg>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory,
                    solver_helper<solver::Bicg>::configurator>(
        updated.first, config, context, exec, updated, value_type_list());
}

template <>
std::unique_ptr<gko::LinOpFactory>
parse<LinOpFactoryType::Bicgstab>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory,
                    solver_helper<solver::Bicgstab>::configurator>(
        updated.first, config, context, exec, updated, value_type_list());
}

template <>
std::unique_ptr<gko::LinOpFactory> parse<LinOpFactoryType::Cgs>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory,
                    solver_helper<solver::Cgs>::configurator>(
        updated.first, config, context, exec, updated, value_type_list());
}

template <>
std::unique_ptr<gko::LinOpFactory> parse<LinOpFactoryType::Fcg>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory,
                    solver_helper<solver::Fcg>::configurator>(
        updated.first, config, context, exec, updated, value_type_list());
}


}  // namespace config
}  // namespace gko
