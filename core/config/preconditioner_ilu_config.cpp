// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/export.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/parse_macro.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


template <bool ReverseApply>
class IluSolverHelper {
public:
    template <typename ValueType, typename IndexType>
    class Configurator {
    public:
        static typename preconditioner::Ilu<ValueType, ReverseApply,
                                            IndexType>::parameters_type
        parse(const pnode& config, const registry& context,
              const type_descriptor& td_for_child)
        {
            return preconditioner::Ilu<ValueType, ReverseApply,
                                       IndexType>::parse(config, context,
                                                         td_for_child);
        }
    };
};


template <>
GKO_EXPORT deferred_factory_parameter<gko::LinOpFactory>
parse<LinOpFactoryType::Ilu>(const pnode& config, const registry& context,
                             const type_descriptor& td)
{
    auto updated = update_type(config, td);
    bool reverse_apply = false;
    if (auto& obj = config.get("reverse_apply")) {
        reverse_apply = obj.get_boolean();
    }
    if (reverse_apply) {
        return dispatch<gko::LinOpFactory, IluSolverHelper<true>::Configurator>(
            config, context, updated,
            make_type_selector(updated.get_value_typestr(), value_type_list()),
            make_type_selector(updated.get_index_typestr(), index_type_list()));
    } else {
        return dispatch<gko::LinOpFactory,
                        IluSolverHelper<false>::Configurator>(
            config, context, updated,
            make_type_selector(updated.get_value_typestr(), value_type_list()),
            make_type_selector(updated.get_index_typestr(), index_type_list()));
    }
}


}  // namespace config
}  // namespace gko
