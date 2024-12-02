// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_PARSE_MACRO_HPP_
#define GKO_CORE_CONFIG_PARSE_MACRO_HPP_


#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/type_descriptor_helper.hpp"


// for value_type only
#define GKO_PARSE_VALUE_TYPE_(_type, _configurator, _value_type_list)        \
    template <>                                                              \
    deferred_factory_parameter<gko::LinOpFactory>                            \
    parse<gko::config::LinOpFactoryType::_type>(                             \
        const gko::config::pnode& config,                                    \
        const gko::config::registry& context,                                \
        const gko::config::type_descriptor& td)                              \
    {                                                                        \
        auto updated = gko::config::update_type(config, td);                 \
        return gko::config::dispatch<gko::LinOpFactory, _configurator>(      \
            config, context, updated,                                        \
            gko::config::make_type_selector(updated.get_value_typestr(),     \
                                            _value_type_list));              \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#define GKO_PARSE_VALUE_TYPE_BASE(_type, _configurator) \
    GKO_PARSE_VALUE_TYPE_(_type, _configurator,         \
                          gko::config::value_type_list_base())

#define GKO_PARSE_VALUE_TYPE(_type, _configurator) \
    GKO_PARSE_VALUE_TYPE_(_type, _configurator, gko::config::value_type_list())

// for value_type and index_type
#define GKO_PARSE_VALUE_AND_INDEX_TYPE_(_type, _configurator,                 \
                                        _value_type_list)                     \
    template <>                                                               \
    deferred_factory_parameter<gko::LinOpFactory>                             \
    parse<gko::config::LinOpFactoryType::_type>(                              \
        const gko::config::pnode& config,                                     \
        const gko::config::registry& context,                                 \
        const gko::config::type_descriptor& td)                               \
    {                                                                         \
        auto updated = gko::config::update_type(config, td);                  \
        return gko::config::dispatch<gko::LinOpFactory, _configurator>(       \
            config, context, updated,                                         \
            gko::config::make_type_selector(updated.get_value_typestr(),      \
                                            _value_type_list),                \
            gko::config::make_type_selector(updated.get_index_typestr(),      \
                                            gko::config::index_type_list())); \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_PARSE_VALUE_AND_INDEX_TYPE_BASE(_type, _configurator) \
    GKO_PARSE_VALUE_AND_INDEX_TYPE_(_type, _configurator,         \
                                    gko::config::value_type_list_base())

#define GKO_PARSE_VALUE_AND_INDEX_TYPE(_type, _configurator) \
    GKO_PARSE_VALUE_AND_INDEX_TYPE_(_type, _configurator,    \
                                    gko::config::value_type_list())


#endif  // GKO_CORE_CONFIG_PARSE_MACRO_HPP_
