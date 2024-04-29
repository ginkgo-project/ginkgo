// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/config.hpp>


#include <map>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


#include "core/config/config_helper.hpp"


namespace gko {
namespace config {


configuration_map generate_config_map()
{
    return {{"solver::Cg", parse<LinOpFactoryType::Cg>}};
}


deferred_factory_parameter<gko::LinOpFactory> parse(const pnode& config,
                                                    const registry& context,
                                                    const type_descriptor& td)
{
    if (auto& obj = config.get("Type")) {
        auto func = context.get_build_map().at(obj.get_string());
        return func(config, context, td);
    }
    GKO_INVALID_STATE("Should contain Type property");
}


}  // namespace config
}  // namespace gko
