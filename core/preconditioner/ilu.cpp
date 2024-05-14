// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/ilu.hpp>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>


#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace preconditioner {


template <typename LSolverType, typename USolverType, bool ReverseApply,
          typename IndexType>
typename Ilu<LSolverType, USolverType, ReverseApply, IndexType>::parameters_type
Ilu<LSolverType, USolverType, ReverseApply, IndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = preconditioner::Ilu<LSolverType, USolverType, ReverseApply,
                                      IndexType>::build();

    if (auto& obj = config.get("l_solver")) {
        params.with_l_solver(
            gko::config::get_specific_factory<const LSolverType>(obj, context,
                                                                 td_for_child));
    }
    if (auto& obj = config.get("u_solver")) {
        params.with_u_solver(
            gko::config::get_specific_factory<const USolverType>(obj, context,
                                                                 td_for_child));
    }
    if (auto& obj = config.get("factorization")) {
        params.with_factorization(
            gko::config::build_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }

    return params;
}

#define GKO_DECLARE_TRS_ILU(ValueType, IndexType)                       \
    class Ilu<solver::LowerTrs<ValueType, IndexType>,                   \
              solver::UpperTrs<ValueType, IndexType>, true, IndexType>; \
    template class Ilu<solver::LowerTrs<ValueType, IndexType>,          \
                       solver::UpperTrs<ValueType, IndexType>, false,   \
                       IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_TRS_ILU);

#define GKO_DECLARE_IR_ILU(ValueType, IndexType)                              \
    class Ilu<solver::Ir<ValueType>, solver::Ir<ValueType>, true, IndexType>; \
    template class Ilu<solver::Ir<ValueType>, solver::Ir<ValueType>, false,   \
                       IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IR_ILU);

#define GKO_DECLARE_GMRES_ILU(ValueType, IndexType)                        \
    class Ilu<solver::Gmres<ValueType>, solver::Gmres<ValueType>, true,    \
              IndexType>;                                                  \
    template class Ilu<solver::Gmres<ValueType>, solver::Gmres<ValueType>, \
                       false, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GMRES_ILU);

#define GKO_DECLARE_ISAI_ILU(ValueType, IndexType)                             \
    class Ilu<preconditioner::LowerIsai<ValueType, IndexType>,                 \
              preconditioner::UpperIsai<ValueType, IndexType>, true,           \
              IndexType>;                                                      \
    template class Ilu<preconditioner::LowerIsai<ValueType, IndexType>,        \
                       preconditioner::UpperIsai<ValueType, IndexType>, false, \
                       IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ISAI_ILU);


}  // namespace preconditioner
}  // namespace gko
