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
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>


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


template <typename ValueType>
class IrConfigurator {
public:
    static std::unique_ptr<typename solver::Ir<ValueType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        auto factory = solver::Ir<ValueType>::build();
        SET_POINTER_VECTOR(factory, const stop::CriterionFactory, criteria,
                           config, context, exec, td_for_child);
        SET_POINTER(factory, const LinOpFactory, solver, config, context, exec,
                    td_for_child);

        SET_POINTER(factory, const LinOp, generated_solver, config, context,
                    exec, td_for_child);
        SET_VALUE(factory, ValueType, relaxation_factor, config);
        SET_VALUE(factory, solver::initial_guess_mode, default_initial_guess,
                  config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Ir>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, IrConfigurator>(
        updated.first, config, context, exec, updated, value_type_list());
}


template <typename ValueType>
class IdrConfigurator {
public:
    static std::unique_ptr<typename solver::Idr<ValueType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        auto factory = solver::Idr<ValueType>::build();
        common_solver_configure(factory, config, context, exec, td_for_child);
        SET_VALUE(factory, size_type, subspace_dim, config);
        SET_VALUE(factory, remove_complex<ValueType>, kappa, config);
        SET_VALUE(factory, bool, deterministic, config);
        SET_VALUE(factory, bool, complex_subspace, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Idr>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, IdrConfigurator>(
        updated.first, config, context, exec, updated, value_type_list());
}


template <typename ValueType>
class GcrConfigurator {
public:
    static std::unique_ptr<typename solver::Gcr<ValueType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        auto factory = solver::Gcr<ValueType>::build();
        common_solver_configure(factory, config, context, exec, td_for_child);
        SET_VALUE(factory, size_type, krylov_dim, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Gcr>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, GcrConfigurator>(
        updated.first, config, context, exec, updated, value_type_list());
}


template <typename ValueType>
class GmresConfigurator {
public:
    static std::unique_ptr<typename solver::Gmres<ValueType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        auto factory = solver::Gmres<ValueType>::build();
        common_solver_configure(factory, config, context, exec, td_for_child);
        SET_VALUE(factory, size_type, krylov_dim, config);
        SET_VALUE(factory, bool, flexible, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Gmres>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, GmresConfigurator>(
        updated.first, config, context, exec, updated, value_type_list());
}


template <typename ValueType>
class CbGmresConfigurator {
public:
    static std::unique_ptr<typename solver::CbGmres<ValueType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        auto factory = solver::CbGmres<ValueType>::build();
        common_solver_configure(factory, config, context, exec, td_for_child);
        SET_VALUE(factory, size_type, krylov_dim, config);
        if (config.contains("storage_precision")) {
            auto get_storage_precision = [](std::string str) {
                using gko::solver::cb_gmres::storage_precision;
                if (str == "keep") {
                    return storage_precision::keep;
                } else if (str == "reduce1") {
                    return storage_precision::reduce1;
                } else if (str == "reduce2") {
                    return storage_precision::reduce2;
                } else if (str == "integer") {
                    return storage_precision::integer;
                } else if (str == "ireduce1") {
                    return storage_precision::ireduce1;
                } else if (str == "ireduce2") {
                    return storage_precision::ireduce2;
                }
                GKO_INVALID_STATE("Wrong value for storage_precision");
            };
            factory.with_storage_precision(get_storage_precision(
                config.at("storage_precision").get_data<std::string>()));
        }
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::CbGmres>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, CbGmresConfigurator>(
        updated.first, config, context, exec, updated, value_type_list());
}


template <typename ValueType, typename IndexType>
class DirectConfigurator {
public:
    static std::unique_ptr<
        typename experimental::solver::Direct<ValueType, IndexType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        auto factory =
            experimental::solver::Direct<ValueType, IndexType>::build();
        SET_VALUE(factory, size_type, num_rhs, config);
        SET_POINTER(factory, const LinOpFactory, factorization, config, context,
                    exec, td_for_child);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Direct>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, DirectConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


template <template <class, class> class Trs>
class trs_helper {
public:
    template <typename ValueType, typename IndexType>
    class configurator {
    public:
        static std::unique_ptr<typename Trs<ValueType, IndexType>::Factory>
        build_from_config(const pnode& config, const registry& context,
                          std::shared_ptr<const Executor> exec,
                          type_descriptor td_for_child)
        {
            auto factory = Trs<ValueType, IndexType>::build();
            SET_VALUE(factory, size_type, num_rhs, config);
            SET_VALUE(factory, bool, unit_diagonal, config);
            if (config.contains("algorithm")) {
                using gko::solver::trisolve_algorithm;
                auto str = config.at("algorithm").get_data<std::string>();
                if (str == "sparselib") {
                    factory.with_algorithm(trisolve_algorithm::sparselib);
                } else if (str == "syncfree") {
                    factory.with_algorithm(trisolve_algorithm::syncfree);
                } else {
                    GKO_INVALID_STATE("Wrong value for algorithm");
                }
            }
            return factory.on(exec);
        }
    };
};


template <>
std::unique_ptr<gko::LinOpFactory>
build_from_config<LinOpFactoryType::LowerTrs>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory,
                    trs_helper<solver::LowerTrs>::configurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}

template <>
std::unique_ptr<gko::LinOpFactory>
build_from_config<LinOpFactoryType::UpperTrs>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory,
                    trs_helper<solver::UpperTrs>::configurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


}  // namespace config
}  // namespace gko
