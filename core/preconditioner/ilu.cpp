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
namespace detail {


template <typename Ilu,
          std::enable_if_t<support_ilu_parse<typename Ilu::l_solver_type,
                                             typename Ilu::u_solver_type>>*>
typename Ilu::parameters_type ilu_parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Ilu::build();

    if (auto& obj = config.get("l_solver")) {
        params.with_l_solver(
            gko::config::parse_or_get_specific_factory<
                const typename Ilu::l_solver_type>(obj, context, td_for_child));
    }
    if (auto& obj = config.get("u_solver")) {
        params.with_u_solver(
            gko::config::parse_or_get_specific_factory<
                const typename Ilu::u_solver_type>(obj, context, td_for_child));
    }
    if (auto& obj = config.get("factorization")) {
        params.with_factorization(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }

    return params;
}


#define GKO_DECLARE_TRS_ILU_FALSE_PARSE(ValueType, IndexType)                 \
    typename Ilu<solver::LowerTrs<ValueType, IndexType>,                      \
                 solver::UpperTrs<ValueType, IndexType>, false,               \
                 IndexType>::parameters_type                                  \
    ilu_parse<Ilu<solver::LowerTrs<ValueType, IndexType>,                     \
                  solver::UpperTrs<ValueType, IndexType>, false, IndexType>>( \
        const config::pnode&, const config::registry&,                        \
        const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_TRS_ILU_FALSE_PARSE);

#define GKO_DECLARE_TRS_ILU_TRUE_PARSE(ValueType, IndexType)                 \
    typename Ilu<solver::LowerTrs<ValueType, IndexType>,                     \
                 solver::UpperTrs<ValueType, IndexType>, true,               \
                 IndexType>::parameters_type                                 \
    ilu_parse<Ilu<solver::LowerTrs<ValueType, IndexType>,                    \
                  solver::UpperTrs<ValueType, IndexType>, true, IndexType>>( \
        const config::pnode&, const config::registry&,                       \
        const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_TRS_ILU_TRUE_PARSE);

#define GKO_DECLARE_GMRES_ILU_FALSE_PARSE(ValueType, IndexType)              \
    typename Ilu<solver::Gmres<ValueType>, solver::Gmres<ValueType>, false,  \
                 IndexType>::parameters_type                                 \
    ilu_parse<Ilu<solver::Gmres<ValueType>, solver::Gmres<ValueType>, false, \
                  IndexType>>(const config::pnode&, const config::registry&, \
                              const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GMRES_ILU_FALSE_PARSE);

#define GKO_DECLARE_GMRES_ILU_TRUE_PARSE(ValueType, IndexType)               \
    typename Ilu<solver::Gmres<ValueType>, solver::Gmres<ValueType>, true,   \
                 IndexType>::parameters_type                                 \
    ilu_parse<Ilu<solver::Gmres<ValueType>, solver::Gmres<ValueType>, true,  \
                  IndexType>>(const config::pnode&, const config::registry&, \
                              const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GMRES_ILU_TRUE_PARSE);

#define GKO_DECLARE_IR_ILU_FALSE_PARSE(ValueType, IndexType)                  \
    typename Ilu<solver::Ir<ValueType>, solver::Ir<ValueType>, false,         \
                 IndexType>::parameters_type                                  \
    ilu_parse<                                                                \
        Ilu<solver::Ir<ValueType>, solver::Ir<ValueType>, false, IndexType>>( \
        const config::pnode&, const config::registry&,                        \
        const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IR_ILU_FALSE_PARSE);

#define GKO_DECLARE_IR_ILU_TRUE_PARSE(ValueType, IndexType)                  \
    typename Ilu<solver::Ir<ValueType>, solver::Ir<ValueType>, true,         \
                 IndexType>::parameters_type                                 \
    ilu_parse<                                                               \
        Ilu<solver::Ir<ValueType>, solver::Ir<ValueType>, true, IndexType>>( \
        const config::pnode&, const config::registry&,                       \
        const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IR_ILU_TRUE_PARSE);

#define GKO_DECLARE_ISAI_ILU_FALSE_PARSE(ValueType, IndexType)         \
    typename Ilu<LowerIsai<ValueType, IndexType>,                      \
                 UpperIsai<ValueType, IndexType>, false,               \
                 IndexType>::parameters_type                           \
    ilu_parse<Ilu<LowerIsai<ValueType, IndexType>,                     \
                  UpperIsai<ValueType, IndexType>, false, IndexType>>( \
        const config::pnode&, const config::registry&,                 \
        const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ISAI_ILU_FALSE_PARSE);

#define GKO_DECLARE_ISAI_ILU_TRUE_PARSE(ValueType, IndexType)         \
    typename Ilu<LowerIsai<ValueType, IndexType>,                     \
                 UpperIsai<ValueType, IndexType>, true,               \
                 IndexType>::parameters_type                          \
    ilu_parse<Ilu<LowerIsai<ValueType, IndexType>,                    \
                  UpperIsai<ValueType, IndexType>, true, IndexType>>( \
        const config::pnode&, const config::registry&,                \
        const config::type_descriptor&)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ISAI_ILU_TRUE_PARSE);


}  // namespace detail
}  // namespace preconditioner
}  // namespace gko
