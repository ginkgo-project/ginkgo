// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/parse_macro.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


// For Ic and Ilu, we use additional ValueType to help Solver type decision
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
        if (config.get("u_solver_type")) {
            GKO_INVALID_STATE(
                "preconditioner::Ilu only allows l_solver_type. The "
                "u_solver_type automatically uses the transposed type of "
                "l_solver_type.");
        }
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
                                   value_type_list_with_half()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else if (str == "solver::Ir") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper1<solver::Ir, solver::Ir,
                           ReverseApply::value>::template Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list_with_half()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else if (str == "preconditioner::LowerIsai") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper2<preconditioner::LowerIsai, preconditioner::UpperIsai,
                           ReverseApply::value>::template Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list_with_half()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else if (str == "solver::Gmres") {
            return dispatch<
                gko::LinOpFactory,
                IluHelper1<solver::Gmres, solver::Gmres,
                           ReverseApply::value>::template Configurator>(
                config, context, updated,
                make_type_selector(updated.get_value_typestr(),
                                   value_type_list_with_half()),
                make_type_selector(updated.get_index_typestr(),
                                   index_type_list()));
        } else {
            GKO_INVALID_CONFIG_VALUE("l_solver_type", str);
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


}  // namespace config
}  // namespace gko
