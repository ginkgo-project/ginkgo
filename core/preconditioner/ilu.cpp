// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/preconditioner/ilu.hpp"

#include <set>
#include <string>

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace preconditioner {
namespace detail {


template <typename Ilu, std::enable_if_t<support_ilu_parse<Ilu>>*>
typename Ilu::parameters_type ilu_parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    // additional l_solver_type and reverse_apply because we allow user select
    // one of instantiation.
    std::set<std::string> allowed_keys{"l_solver", "u_solver", "factorization",
                                       "l_solver_type", "reverse_apply"};
    gko::config::check_allowed_keys(config, allowed_keys);

    auto params = Ilu::build();
    using l_solver_type = typename Ilu::l_solver_type;
    using u_solver_type = typename Ilu::u_solver_type;
    static_assert(std::is_same_v<l_solver_type, LinOp>,
                  "only support ILU parse when l_solver_type is LinOp.");
    static_assert(std::is_same_v<u_solver_type, LinOp>,
                  "only support ILU parse when u_solver_type is LinOp.");
    if (auto& obj = config.get("l_solver")) {
        params.with_l_solver(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config.get("u_solver")) {
        params.with_u_solver(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config.get("factorization")) {
        params.with_factorization(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }

    return params;
}


#define GKO_DECLARE_ILU_PARSE_FALSE(ValueType, IndexType)                 \
    typename Ilu<ValueType, ValueType, false, IndexType>::parameters_type \
    ilu_parse<Ilu<ValueType, ValueType, false, IndexType>>(               \
        const config::pnode&, const config::registry&,                    \
        const config::type_descriptor&)
#define GKO_DECLARE_ILU_PARSE_TRUE(ValueType, IndexType)                 \
    typename Ilu<ValueType, ValueType, true, IndexType>::parameters_type \
    ilu_parse<Ilu<ValueType, ValueType, true, IndexType>>(               \
        const config::pnode&, const config::registry&,                   \
        const config::type_descriptor&)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU_PARSE_FALSE);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU_PARSE_TRUE);


}  // namespace detail


// only instantiate the value type variants of ILU, whose solver is LinOp.
#define GKO_DECLARE_ILU_FALSE(ValueType, IndexType) \
    class Ilu<ValueType, ValueType, false, IndexType>
#define GKO_DECLARE_ILU_TRUE(ValueType, IndexType) \
    class Ilu<ValueType, ValueType, true, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU_FALSE);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU_TRUE);


}  // namespace preconditioner
}  // namespace gko
