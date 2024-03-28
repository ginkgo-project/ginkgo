// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/config.hpp>


#include <map>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace config {


buildfromconfig_map generate_config_map()
{
    return {{"Cg", build_from_config<LinOpFactoryType::Cg>}};
}


deferred_factory_parameter<gko::LinOpFactory> build_from_config(
    const pnode& config, const registry& context, type_descriptor td)
{
    if (auto& obj = config.get("Type")) {
        auto func = context.get_build_map().at(obj.get_string());
        return func(config, context, td);
    }
    GKO_INVALID_STATE("Should contain Type property");
}


}  // namespace config
}  // namespace gko
