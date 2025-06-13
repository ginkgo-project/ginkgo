// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/preconditioner/ic.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/parse_macro.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


template <>
deferred_factory_parameter<gko::LinOpFactory>
parse<gko::config::LinOpFactoryType::Ic>(const gko::config::pnode& config,
                                         const gko::config::registry& context,
                                         const gko::config::type_descriptor& td)
{
    auto updated = gko::config::update_type(config, td);
    if (config.get("l_solver_type_or_value_type")) {
        GKO_INVALID_STATE(
            "preconditioner::Ic only allows value_type from "
            "l_solver_type_or_value_type. Please use value_type key to set the "
            "value type used by the preconditioner and the l_lover  key to set "
            "the solvers used for the lower triangular systems.");
    }
    return gko::config::dispatch<gko::LinOpFactory, gko::preconditioner::Ic>(
        config, context, updated,
        gko::config::make_type_selector(updated.get_value_typestr(),
                                        gko::config::value_type_list()),
        gko::config::make_type_selector(updated.get_index_typestr(),
                                        gko::config::index_type_list()));
}
static_assert(true,
              "This assert is used to counter the false positive extra "
              "semi-colon warnings");


}  // namespace config
}  // namespace gko
