// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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
    } else if (str == "LinOp") {
        return dispatch<gko::LinOpFactory, IcSolverHelper<LinOp>::Configurator>(
            config, context, updated,
            // no effect
            make_type_selector(updated.get_value_typestr(), value_type_list()),
            make_type_selector(updated.get_index_typestr(), index_type_list()));
    } else {
        GKO_INVALID_CONFIG_VALUE("l_solver_type", str);
    }
}


}  // namespace config
}  // namespace gko
