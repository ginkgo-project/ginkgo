// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::Schwarz>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    // We can not directly dispatch the global index type without consider local
    // index type, which leads the invalid index type <int64, int32> in
    // compile time.
    if (updated.get_index_typestr() == type_string<int32>::str()) {
        return dispatch<
            gko::LinOpFactory,
            gko::experimental::distributed::preconditioner::Schwarz>(
            config, context, updated,
            make_type_selector(updated.get_value_typestr(),
                               value_type_list_base()),
            make_type_selector(updated.get_index_typestr(),
                               syn::type_list<int32>()),
            make_type_selector(updated.get_global_index_typestr(),
                               index_type_list()));
    } else {
        return dispatch<
            gko::LinOpFactory,
            gko::experimental::distributed::preconditioner::Schwarz>(
            config, context, updated,
            make_type_selector(updated.get_value_typestr(),
                               value_type_list_base()),
            make_type_selector(updated.get_index_typestr(),
                               syn::type_list<int64>()),
            make_type_selector(updated.get_global_index_typestr(),
                               syn::type_list<int64>()));
    }
}


}  // namespace config
}  // namespace gko
