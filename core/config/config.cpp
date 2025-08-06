// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/config/config.hpp"

#include <map>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/solver/solver_base.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/registry_accessor.hpp"


namespace gko {
namespace config {


deferred_factory_parameter<gko::LinOpFactory> parse(const pnode& config,
                                                    const registry& context,
                                                    const type_descriptor& td)
{
    if (auto& obj = config.get("type")) {
        auto func = detail::registry_accessor::get_build_map(context).at(
            obj.get_string());
        return func(config, context, td);
    }
    GKO_MISSING_CONFIG_ENTRY("type");
}


}  // namespace config
}  // namespace gko
