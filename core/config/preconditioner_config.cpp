// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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


#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
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
            parse(const pnode& config, const registry& context,
                  const type_descriptor& td_for_child)
        {
            return gko::preconditioner::Ic<Solver, IndexType>::parse(
                config, context, td_for_child);
        }
    };
};


template <typename LSolver, typename USolver, bool ReverseApply>
class IluSolverHelper {
public:
    template <typename ValueType, typename IndexType>
    class Configurator {
    public:
        static typename preconditioner::Ilu<LSolver, USolver, ReverseApply,
                                            IndexType>::parameters_type
        parse(const pnode& config, const registry& context,
              const type_descriptor& td_for_child)
        {
            return preconditioner::Ilu<LSolver, USolver, ReverseApply,
                                       IndexType>::parse(config, context,
                                                         td_for_child);
        }
    };
};


template <preconditioner::isai_type IsaiType>
class IsaiHelper {
public:
    template <typename ValueType, typename IndexType>
    class Configurator {
    public:
        static typename preconditioner::Isai<IsaiType, ValueType,
                                             IndexType>::parameters_type
        parse(const pnode& config, const registry& context,
              const type_descriptor& td_for_child)
        {
            return preconditioner::Isai<IsaiType, ValueType, IndexType>::parse(
                config, context, td_for_child);
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
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::Ic>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    std::string str("solver::LowerTrs");
    if (auto& obj = config.get("l_solver_type")) {
        str = obj.get_string();
    }
    if (str == "solver::LowerTrs") {
        return dispatch<gko::LinOpFactory,
                        IcHelper2<solver::LowerTrs>::Configurator>(
            config, context, updated,
            make_type_selector(updated.get_value_typestr(), value_type_list()),
            make_type_selector(updated.get_index_typestr(), index_type_list()));
    } else if (str == "solver::Ir") {
        return dispatch<gko::LinOpFactory, IcHelper1<solver::Ir>::Configurator>(
            config, context, updated,
            make_type_selector(updated.get_value_typestr(), value_type_list()),
            make_type_selector(updated.get_index_typestr(), index_type_list()));
    } else if (str == "preconditioner::LowerIsai") {
        return dispatch<gko::LinOpFactory,
                        IcHelper2<preconditioner::LowerIsai>::Configurator>(
            config, context, updated,
            make_type_selector(updated.get_value_typestr(), value_type_list()),
            make_type_selector(updated.get_index_typestr(), index_type_list()));
    } else if (str == "solver::Gmres") {
        return dispatch<gko::LinOpFactory,
                        IcHelper1<solver::Gmres>::Configurator>(
            config, context, updated,
            make_type_selector(updated.get_value_typestr(), value_type_list()),
            make_type_selector(updated.get_index_typestr(), index_type_list()));
    } else {
        GKO_INVALID_STATE("does not have valid LSolverType");
    }
}


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
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::Ilu>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    auto dispatch_solver = [&](auto reverse_apply)
        -> deferred_factory_parameter<gko::LinOpFactory> {
        using ReverseApply = decltype(reverse_apply);
        // always use symmetric solver for USolverType
        std::string str("solver::LowerTrs");
        if (auto& obj = config.get("l_solver_type")) {
            str = obj.get_string();
        }
        if (str == "solver::LowerTrs") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper2<solver::LowerTrs, solver::UpperTrs,
                           ReverseApply::value>::template Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else if (str == "solver::Ir") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper1<solver::Ir, solver::Ir,
                           ReverseApply::value>::template Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else if (str == "preconditioner::LowerIsai") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper2<preconditioner::LowerIsai, preconditioner::UpperIsai,
                           ReverseApply::value>::template Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else if (str == "solver::Gmres") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper1<solver::Gmres, solver::Gmres,
                           ReverseApply::value>::template Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else {
            GKO_INVALID_STATE("does not have valid LSolverType");
        }
    };
    bool reverse_apply = false;
    if (auto& obj = config.get("reverse_apply")) {
        reverse_apply = obj.get_boolean();
    }
    if (reverse_apply) {
        return dispatch_solver(std::true_type{});
    } else {
        return dispatch_solver(std::false_type{});
    }
}


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::Isai>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    if (auto& obj = config.get("isai_type")) {
        auto str = obj.get_string();
        if (str == "lower") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::lower>::Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else if (str == "upper") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::upper>::Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else if (str == "general") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::general>::Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else if (str == "spd") {
            return dispatch<
                gko::LinOpFactory,
                IsaiHelper<preconditioner::isai_type::spd>::Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else {
            GKO_INVALID_STATE("does not have valid IsaiType");
        }
    } else {
        GKO_INVALID_STATE("does not contain IsaiType");
    }
}


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::Jacobi>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, preconditioner::Jacobi>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}


}  // namespace config
}  // namespace gko
