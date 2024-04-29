// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/registry.hpp>


#include <ginkgo/core/config/config.hpp>


#include "core/config/config_helper.hpp"


namespace gko {
namespace config {

registry::registry(configuration_map build_map) : build_map_(build_map) {}

registry::registry(
    std::unordered_map<std::string, detail::allowed_ptr> stored_map,
    configuration_map build_map)
    : stored_map_(stored_map), build_map_(build_map)
{}


}  // namespace config
}  // namespace gko
