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
#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/config/config.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace config {


// For Ic and Ilu, we use additional ValueType to help Solver type decision

template <typename Solver>
class IcSolverHelper {
public:
    template <typename ValueType, typename IndexType>
    class Configurator {
    public:
        static std::unique_ptr<
            typename preconditioner::Ic<Solver, IndexType>::Factory>
        build_from_config(const pnode& config, const registry& context,
                          std::shared_ptr<const Executor> exec,
                          type_descriptor td_for_child)
        {
            auto factory = preconditioner::Ic<Solver, IndexType>::build();
            SET_POINTER(factory, typename Solver::Factory, l_solver_factory,
                        config, context, exec, td_for_child);
            SET_POINTER(factory, LinOpFactory, factorization_factory, config,
                        context, exec, td_for_child);
            return factory.on(exec);
        }
    };
};

template <template <typename...> class>
class IcHelper {};

template <template <typename V> class SolverBase>
class IcHelper<SolverBase> {
public:
    template <typename ValueType, typename IndexType>
    class Configurator
        : public IcSolverHelper<SolverBase<ValueType>>::template Configurator<
              ValueType, IndexType> {};
};

template <template <typename V, typename I> class SolverBase>
class IcHelper<SolverBase> {
public:
    template <typename ValueType, typename IndexType>
    class Configurator
        : public IcSolverHelper<SolverBase<ValueType, IndexType>>::
              template Configurator<ValueType, IndexType> {};
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Ic>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    if (config.contains("LSolverType")) {
        auto str = config.at("LSolverType").get_data<std::string>();
        if (str == "LowerTrs") {
            return dispatch<gko::LinOpFactory,
                            IcHelper<solver::LowerTrs>::Configurator>(
                updated.first + "," + updated.second, config, context, exec,
                updated, value_type_list(), index_type_list());
        } else if (str == "Ir") {
            return dispatch<gko::LinOpFactory,
                            IcHelper<solver::Ir>::Configurator>(
                updated.first + "," + updated.second, config, context, exec,
                updated, value_type_list(), index_type_list());
        } else if (str == "LowerIsai") {
            return dispatch<gko::LinOpFactory,
                            IcHelper<preconditioner::LowerIsai>::Configurator>(
                updated.first + "," + updated.second, config, context, exec,
                updated, value_type_list(), index_type_list());
        } else if (str == "Gmres") {
            return dispatch<gko::LinOpFactory,
                            IcHelper<solver::Gmres>::Configurator>(
                updated.first + "," + updated.second, config, context, exec,
                updated, value_type_list(), index_type_list());
        } else {
            GKO_INVALID_STATE("does not have valid LSolverType");
        }
    } else {
        GKO_INVALID_STATE("does not contain LSolverType");
    }
}


// template <typename LSolver, typename USolver, bool ReverseApply>
// class IluSolverHelper {
// public:
//     template <typename ValueType, typename IndexType>
//     class Configurator {
//     public:
//         static std::unique_ptr<typename preconditioner::Ilu<
//             LSolver, USolver, ReverseApply, IndexType>::Factory>
//         build_from_config(const pnode& config, const registry& context,
//                           std::shared_ptr<const Executor> exec,
//                           type_descriptor td_for_child)
//         {
//             auto factory = preconditioner::Ilu<LSolver, USolver,
//             ReverseApply,
//                                                IndexType>::build();
//             SET_POINTER(factory, typename LSolver::Factory, l_solver_factory,
//                         config, context, exec, td_for_child);
//             SET_POINTER(factory, typename USolver::Factory, u_solver_factory,
//                         config, context, exec, td_for_child);
//             SET_POINTER(factory, LinOpFactory, factorization_factory, config,
//                         context, exec, td_for_child);
//             return factory.on(exec);
//         }
//     };
// };

// template <template <typename...> class, >
// class IluHelper {};

// template <template <typename V> class SolverBase>
// class IcHelper<SolverBase> {
// public:
//     template <typename ValueType, typename IndexType>
//     class Configurator
//         : public IcSolverHelper<SolverBase<ValueType>>::template
//         Configurator<
//               ValueType, IndexType> {};
// };

// template <template <typename V, typename I> class SolverBase>
// class IcHelper<SolverBase> {
// public:
//     template <typename ValueType, typename IndexType>
//     class Configurator
//         : public IcSolverHelper<SolverBase<ValueType, IndexType>>::
//               template Configurator<ValueType, IndexType> {};
// };


// template <>
// std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Ic>(
//     const pnode& config, const registry& context,
//     std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
// {
//     auto updated = update_type(config, td);
//     if (config.contains("LSolverType")) {
//         auto str = config.at("LSolverType").get_data<std::string>();
//         if (str == "LowerTrs") {
//             return dispatch<gko::LinOpFactory,
//                             IcHelper<solver::LowerTrs>::Configurator>(
//                 updated.first + "," + updated.second, config, context, exec,
//                 updated, value_type_list(), index_type_list());
//         } else if (str == "Ir") {
//             return dispatch<gko::LinOpFactory,
//                             IcHelper<solver::Ir>::Configurator>(
//                 updated.first + "," + updated.second, config, context, exec,
//                 updated, value_type_list(), index_type_list());
//         } else if (str == "LowerIsai") {
//             return dispatch<gko::LinOpFactory,
//                             IcHelper<preconditioner::LowerIsai>::Configurator>(
//                 updated.first + "," + updated.second, config, context, exec,
//                 updated, value_type_list(), index_type_list());
//         } else if (str == "Gmres") {
//             return dispatch<gko::LinOpFactory,
//                             IcHelper<solver::Gmres>::Configurator>(
//                 updated.first + "," + updated.second, config, context, exec,
//                 updated, value_type_list(), index_type_list());
//         } else {
//             GKO_INVALID_STATE("does not have valid LSolverType");
//         }
//     } else {
//         GKO_INVALID_STATE("does not contain LSolverType");
//     }
// }


template <preconditioner::isai_type IsaiType>
class IsaiHelper {
public:
    template <typename ValueType, typename IndexType>
    class Configurator {
    public:
        static std::unique_ptr<typename preconditioner::Isai<
            IsaiType, ValueType, IndexType>::Factory>
        build_from_config(const pnode& config, const registry& context,
                          std::shared_ptr<const Executor> exec,
                          type_descriptor td_for_child)
        {
            auto factory =
                preconditioner::Isai<IsaiType, ValueType, IndexType>::build();
            SET_VALUE(factory, bool, skip_sorting, config);
            SET_VALUE(factory, int, sparsity_power, config);
            SET_VALUE(factory, size_type, excess_limit, config);
            SET_POINTER(factory, LinOpFactory, excess_solver_factory, config,
                        context, exec, td_for_child);
            SET_VALUE(factory, remove_complex<ValueType>,
                      excess_solver_reduction, config);
            return factory.on(exec);
        }
    };
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Isai>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    if (config.contains("IsaiType")) {
        auto str = config.at("IsaiType").get_data<std::string>();
        if (str == "lower") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::lower>::Configurator>(
                updated.first + "," + updated.second, config, context, exec,
                updated, value_type_list(), index_type_list());
        } else if (str == "upper") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::upper>::Configurator>(
                updated.first + "," + updated.second, config, context, exec,
                updated, value_type_list(), index_type_list());
        } else if (str == "general") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::general>::Configurator>(
                updated.first + "," + updated.second, config, context, exec,
                updated, value_type_list(), index_type_list());
        } else if (str == "spd") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::spd>::Configurator>(
                updated.first + "," + updated.second, config, context, exec,
                updated, value_type_list(), index_type_list());
        } else {
            GKO_INVALID_STATE("does not have valid IsaiType");
        }
    } else {
        GKO_INVALID_STATE("does not contain IsaiType");
    }
}


template <typename ValueType, typename IndexType>
class JacobiConfigurator {
public:
    static std::unique_ptr<
        typename preconditioner::Jacobi<ValueType, IndexType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        auto factory = preconditioner::Jacobi<ValueType, IndexType>::build();
        SET_VALUE(factory, uint32, max_block_size, config);
        SET_VALUE(factory, uint32, max_block_stride, config);
        SET_VALUE(factory, bool, skip_sorting, config);
        SET_VALUE_ARRAY(factory, gko::array<IndexType>, block_pointers, config,
                        exec);
        // storage_optimization_type is not public. It uses precision_reduction
        // as input.
        SET_VALUE(factory, precision_reduction, storage_optimization, config);
        SET_VALUE(factory, remove_complex<ValueType>, accuracy, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Jacobi>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, JacobiConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


}  // namespace config
}  // namespace gko
