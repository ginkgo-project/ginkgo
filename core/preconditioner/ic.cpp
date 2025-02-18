// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/preconditioner/ic.hpp"

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/utils.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace preconditioner {
namespace detail {


template <typename Ic,
          std::enable_if_t<support_ic_parse<typename Ic::l_solver_type>>*>
typename Ic::parameters_type ic_parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Ic::build();

    if (auto& obj = config.get("l_solver")) {
        params.with_l_solver(
            gko::config::parse_or_get_specific_factory<
                const typename Ic::l_solver_type>(obj, context, td_for_child));
    }
    if (auto& obj = config.get("factorization")) {
        params.with_factorization(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }

    return params;
}


#define GKO_DECLARE_LOWERTRS_IC_PARSE(ValueType, IndexType)          \
    typename Ic<solver::LowerTrs<ValueType, IndexType>,              \
                IndexType>::parameters_type                          \
    ic_parse<Ic<solver::LowerTrs<ValueType, IndexType>, IndexType>>( \
        const config::pnode&, const config::registry&,               \
        const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWERTRS_IC_PARSE);

#define GKO_DECLARE_IR_IC_PARSE(ValueType, IndexType)              \
    typename Ic<solver::Ir<ValueType>, IndexType>::parameters_type \
    ic_parse<Ic<solver::Ir<ValueType>, IndexType>>(                \
        const config::pnode&, const config::registry&,             \
        const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IR_IC_PARSE);

#define GKO_DECLARE_GMRES_IC_PARSE(ValueType, IndexType)              \
    typename Ic<solver::Gmres<ValueType>, IndexType>::parameters_type \
    ic_parse<Ic<solver::Gmres<ValueType>, IndexType>>(                \
        const config::pnode&, const config::registry&,                \
        const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GMRES_IC_PARSE);

#define GKO_DECLARE_LOWERISAI_IC_PARSE(ValueType, IndexType)                 \
    typename Ic<LowerIsai<ValueType, IndexType>, IndexType>::parameters_type \
    ic_parse<Ic<LowerIsai<ValueType, IndexType>, IndexType>>(                \
        const config::pnode&, const config::registry&,                       \
        const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWERISAI_IC_PARSE);

}  // namespace detail
}  // namespace preconditioner
}  // namespace gko
