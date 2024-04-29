// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/registry.hpp>


#include "core/config/config_helper.hpp"


namespace gko {
namespace config {


type_descriptor update_type(const pnode& config, const type_descriptor& td)
{
    auto value_typestr = td.get_value_typestr();
    auto index_typestr = td.get_index_typestr();

    if (auto& obj = config.get("ValueType")) {
        value_typestr = obj.get_string();
    }
    if (auto& obj = config.get("IndexType")) {
        index_typestr = obj.get_string();
    }
    return type_descriptor{value_typestr, index_typestr};
}


template <>
deferred_factory_parameter<const LinOpFactory> get_factory<const LinOpFactory>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    deferred_factory_parameter<const LinOpFactory> ptr;
    if (config.get_tag() == pnode::tag_t::string) {
        ptr = context.search_data<LinOpFactory>(config.get_string());
    } else if (config.get_tag() == pnode::tag_t::map) {
        ptr = parse(config, context, td);
    } else {
        GKO_INVALID_STATE("The data of config is not valid.");
    }
    GKO_THROW_IF_INVALID(!ptr.is_empty(), "Parse get nullptr in the end");

    return ptr;
}


}  // namespace config
}  // namespace gko
