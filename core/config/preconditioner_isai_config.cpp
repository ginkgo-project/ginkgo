// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/parse_macro.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


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
            GKO_INVALID_CONFIG_VALUE("isai_type", str);
        }
    } else {
        GKO_MISSING_CONFIG_ENTRY("isai_type");
    }
}


}  // namespace config
}  // namespace gko
