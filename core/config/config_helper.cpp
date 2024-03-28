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
    type_descriptor updated = td;

    if (auto& obj = config.get("ValueType")) {
        updated.first = obj.get_string();
    }
    if (auto& obj = config.get("IndexType")) {
        updated.second = obj.get_string();
    }
    return updated;
}


template <>
deferred_factory_parameter<const LinOpFactory> get_factory<const LinOpFactory>(
    const pnode& config, const registry& context, type_descriptor td)
{
    deferred_factory_parameter<const LinOpFactory> ptr;
    if (config.get_status() == pnode::status_t::string) {
        ptr = context.search_data<LinOpFactory>(config.get_string());
    } else if (config.get_status() == pnode::status_t::map) {
        ptr = build_from_config(config, context, td);
    }
    assert(!ptr.is_empty());
    return std::move(ptr);
}


}  // namespace config
}  // namespace gko
