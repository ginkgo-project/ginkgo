// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/ic.hpp>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>


#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace preconditioner {


template <typename LSolverType, typename IndexType>
typename Ic<LSolverType, IndexType>::parameters_type
Ic<LSolverType, IndexType>::parse(const config::pnode& config,
                                  const config::registry& context,
                                  const config::type_descriptor& td_for_child)
{
    auto params = preconditioner::Ic<LSolverType, IndexType>::build();

    if (auto& obj = config.get("l_solver")) {
        params.with_l_solver(
            gko::config::get_specific_factory<const LSolverType>(obj, context,
                                                                 td_for_child));
    }
    if (auto& obj = config.get("factorization")) {
        params.with_factorization(
            gko::config::build_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }

    return params;
}

#define GKO_DECLARE_LOWERTRS_IC(ValueType, IndexType) \
    class Ic<solver::LowerTrs<ValueType, IndexType>, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWERTRS_IC);

#define GKO_DECLARE_IR_IC(ValueType, IndexType) \
    class Ic<solver::Ir<ValueType>, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IR_IC);

#define GKO_DECLARE_GMRES_IC(ValueType, IndexType) \
    class Ic<solver::Gmres<ValueType>, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GMRES_IC);

#define GKO_DECLARE_LOWERISAI_IC(ValueType, IndexType) \
    class Ic<preconditioner::LowerIsai<ValueType, IndexType>, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWERISAI_IC);


}  // namespace preconditioner
}  // namespace gko
