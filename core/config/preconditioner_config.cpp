// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
namespace preconditioner {
// handle Ic and Ilu here.
template <typename LSolverType, typename IndexType>
typename Ic<LSolverType, IndexType>::parameters_type
Ic<LSolverType, IndexType>::build_from_config(
    const config::pnode& config, const config::registry& context,
    config::type_descriptor td_for_child)
{
    auto factory = preconditioner::Ic<LSolverType, IndexType>::build();
    SET_FACTORY(factory, const typename LSolverType::Factory, l_solver, config,
                context, td_for_child);
    SET_FACTORY(factory, const LinOpFactory, factorization, config, context,
                td_for_child);
    return factory;
}


template <typename LSolverType, typename USolverType, bool ReverseApply,
          typename IndexType>
typename Ilu<LSolverType, USolverType, ReverseApply, IndexType>::parameters_type
Ilu<LSolverType, USolverType, ReverseApply, IndexType>::build_from_config(
    const config::pnode& config, const config::registry& context,
    config::type_descriptor td_for_child)
{
    auto factory = preconditioner::Ilu<LSolverType, USolverType, ReverseApply,
                                       IndexType>::build();
    SET_FACTORY(factory, const typename LSolverType::Factory, l_solver, config,
                context, td_for_child);
    SET_FACTORY(factory, const typename USolverType::Factory, u_solver, config,
                context, td_for_child);
    SET_FACTORY(factory, const LinOpFactory, factorization, config, context,
                td_for_child);
    return factory;
}

}  // namespace preconditioner


namespace config {


// For Ic and Ilu, we use additional ValueType to help Solver type decision

template <typename Solver>
class IcSolverHelper {
public:
    template <typename ValueType, typename IndexType>
    class Configurator {
    public:
        static
            typename gko::preconditioner::Ic<Solver, IndexType>::parameters_type
            build_from_config(const pnode& config, const registry& context,
                              type_descriptor td_for_child)
        {
            return gko::preconditioner::Ic<
                Solver, IndexType>::build_from_config(config, context,
                                                      td_for_child);
        }
    };
};


// Do not use the partial specialization for SolverBase<V> and SolverBase<V, I>
// because the default template arguments are allowed for a template template
// argument (detail: CWG 150 after c++17
// https://en.cppreference.com/w/cpp/language/template_parameters#Template_template_arguments)
template <template <typename V> class SolverBase>
class IcHelper1 {
public:
    template <typename ValueType, typename IndexType>
    class Configurator
        : public IcSolverHelper<SolverBase<ValueType>>::template Configurator<
              ValueType, IndexType> {};
};


template <template <typename V, typename I> class SolverBase>
class IcHelper2 {
public:
    template <typename ValueType, typename IndexType>
    class Configurator
        : public IcSolverHelper<SolverBase<ValueType, IndexType>>::
              template Configurator<ValueType, IndexType> {};
};


template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Ic>(const pnode& config,
                                        const registry& context,
                                        gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    std::string str("LowerTrs");
    if (config.contains("LSolverType")) {
        str = config.at("LSolverType").get_data<std::string>();
    }
    if (str == "LowerTrs") {
        return dispatch<gko::LinOpFactory,
                        IcHelper2<solver::LowerTrs>::Configurator>(
            updated.first + "," + updated.second, config, context, updated,
            value_type_list(), index_type_list());
    } else if (str == "Ir") {
        return dispatch<gko::LinOpFactory, IcHelper1<solver::Ir>::Configurator>(
            updated.first + "," + updated.second, config, context, updated,
            value_type_list(), index_type_list());
    } else if (str == "LowerIsai") {
        return dispatch<gko::LinOpFactory,
                        IcHelper2<preconditioner::LowerIsai>::Configurator>(
            updated.first + "," + updated.second, config, context, updated,
            value_type_list(), index_type_list());
    } else if (str == "Gmres") {
        return dispatch<gko::LinOpFactory,
                        IcHelper1<solver::Gmres>::Configurator>(
            updated.first + "," + updated.second, config, context, updated,
            value_type_list(), index_type_list());
    } else {
        GKO_INVALID_STATE("does not have valid LSolverType");
    }
}


template <typename LSolver, typename USolver, bool ReverseApply>
class IluSolverHelper {
public:
    template <typename ValueType, typename IndexType>
    class Configurator {
    public:
        static typename preconditioner::Ilu<LSolver, USolver, ReverseApply,
                                            IndexType>::parameters_type
        build_from_config(const pnode& config, const registry& context,
                          type_descriptor td_for_child)
        {
            return preconditioner::Ilu<
                LSolver, USolver, ReverseApply,
                IndexType>::build_from_config(config, context, td_for_child);
        }
    };
};


template <template <typename V> class LSolverBase,
          template <typename V> class USolverBase, bool ReverseApply>
class IluHelper1 {
public:
    template <typename ValueType, typename IndexType>
    class Configurator
        : public IluSolverHelper<
              LSolverBase<ValueType>, USolverBase<ValueType>,
              ReverseApply>::template Configurator<ValueType, IndexType> {};
};


template <template <typename V, typename I> class LSolverBase,
          template <typename V, typename I> class USolverBase,
          bool ReverseApply>
class IluHelper2 {
public:
    template <typename ValueType, typename IndexType>
    class Configurator
        : public IluSolverHelper<
              LSolverBase<ValueType, IndexType>,
              USolverBase<ValueType, IndexType>,
              ReverseApply>::template Configurator<ValueType, IndexType> {};
};


template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Ilu>(const pnode& config,
                                         const registry& context,
                                         gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    auto dispatch_solver = [&](auto reverse_apply)
        -> deferred_factory_parameter<gko::LinOpFactory> {
        using ReverseApply = decltype(reverse_apply);
        // always use symmetric solver for USolverType
        std::string str("LowerTrs");
        if (config.contains("LSolverType")) {
            str = config.at("LSolverType").get_data<std::string>();
        }
        if (str == "LowerTrs") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper2<solver::LowerTrs, solver::UpperTrs,
                           ReverseApply::value>::template Configurator>(
                updated.first + "," + updated.second, config, context, updated,
                value_type_list(), index_type_list());
        } else if (str == "Ir") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper1<solver::Ir, solver::Ir,
                           ReverseApply::value>::template Configurator>(
                updated.first + "," + updated.second, config, context, updated,
                value_type_list(), index_type_list());
        } else if (str == "LowerIsai") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper2<preconditioner::LowerIsai, preconditioner::UpperIsai,
                           ReverseApply::value>::template Configurator>(
                updated.first + "," + updated.second, config, context, updated,
                value_type_list(), index_type_list());
        } else if (str == "Gmres") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper1<solver::Gmres, solver::Gmres,
                           ReverseApply::value>::template Configurator>(
                updated.first + "," + updated.second, config, context, updated,
                value_type_list(), index_type_list());
        } else {
            GKO_INVALID_STATE("does not have valid LSolverType");
        }
    };
    bool reverse_apply = false;
    if (config.contains("ReverseApply")) {
        reverse_apply = config.at("ReverseApply").get_data<bool>();
    }
    if (reverse_apply) {
        return dispatch_solver(std::true_type{});
    } else {
        return dispatch_solver(std::false_type{});
    }
}


template <preconditioner::isai_type IsaiType>
class IsaiHelper {
public:
    template <typename ValueType, typename IndexType>
    class Configurator {
    public:
        static typename preconditioner::Isai<IsaiType, ValueType,
                                             IndexType>::parameters_type
        build_from_config(const pnode& config, const registry& context,
                          type_descriptor td_for_child)
        {
            return preconditioner::Isai<IsaiType, ValueType, IndexType>::
                build_from_config(config, context, td_for_child);
        }
    };
};


template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Isai>(const pnode& config,
                                          const registry& context,
                                          gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    if (config.contains("IsaiType")) {
        auto str = config.at("IsaiType").get_data<std::string>();
        if (str == "lower") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::lower>::Configurator>(
                updated.first + "," + updated.second, config, context, updated,
                value_type_list(), index_type_list());
        } else if (str == "upper") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::upper>::Configurator>(
                updated.first + "," + updated.second, config, context, updated,
                value_type_list(), index_type_list());
        } else if (str == "general") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::general>::Configurator>(
                updated.first + "," + updated.second, config, context, updated,
                value_type_list(), index_type_list());
        } else if (str == "spd") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::spd>::Configurator>(
                updated.first + "," + updated.second, config, context, updated,
                value_type_list(), index_type_list());
        } else {
            GKO_INVALID_STATE("does not have valid IsaiType");
        }
    } else {
        GKO_INVALID_STATE("does not contain IsaiType");
    }
}


template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Jacobi>(const pnode& config,
                                            const registry& context,
                                            gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::preconditioner::Jacobi>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}


}  // namespace config
}  // namespace gko
