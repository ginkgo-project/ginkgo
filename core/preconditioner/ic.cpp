// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/preconditioner/ic.hpp"

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace preconditioner {
namespace detail {


template <typename Ic, std::enable_if_t<support_ic_parse<Ic>>*>
typename Ic::parameters_type ic_parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Ic::build();
    config::config_check_decorator config_check(config);

    using l_solver_type = typename Ic::l_solver_type;
    static_assert(std::is_same_v<l_solver_type, LinOp>,
                  "only support IC parse when l_solver_type is LinOp.");

    if (auto& obj = config_check.get("l_solver")) {
        params.with_l_solver(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("factorization")) {
        params.with_factorization(
            config::parse_or_get_factory<const LinOpFactory>(obj, context,
                                                             td_for_child));
    }

    return params;
}


#define GKO_DECLARE_IC_PARSE(ValueType, IndexType)              \
    typename Ic<ValueType, IndexType>::parameters_type          \
    ic_parse<Ic<ValueType, IndexType>>(const config::pnode&,    \
                                       const config::registry&, \
                                       const config::type_descriptor&)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IC_PARSE);


}  // namespace detail


// only instantiate the value type variants of IC, whose solver is LinOp.
#define GKO_DECLARE_IC(ValueType, IndexType) class Ic<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IC);


}  // namespace preconditioner
}  // namespace gko
