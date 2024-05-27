// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/registry.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>


#include "core/config/config_helper.hpp"


namespace gko {
namespace config {


configuration_map generate_config_map()
{
    return {{"solver::Cg", parse<LinOpFactoryType::Cg>}};
}


registry::registry(const configuration_map& additional_map)
    : registry({}, additional_map)
{}


registry::registry(
    const std::unordered_map<std::string, detail::allowed_ptr>& stored_map,
    const configuration_map& additional_map)
    : stored_map_(stored_map), build_map_(generate_config_map())
{
    // merge additional_map into build_map_
    for (auto& item : additional_map) {
        auto res = build_map_.emplace(item.first, item.second);
        GKO_THROW_IF_INVALID(res.second,
                             "failed when adding the key " + item.first);
    }
}


}  // namespace config
}  // namespace gko
