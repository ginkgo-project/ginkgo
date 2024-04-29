// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/registry.hpp>


#include "core/config/config_helper.hpp"


namespace gko {
namespace config {


using linop_map = std::unordered_map<std::string, std::shared_ptr<LinOp>>;
using linopfactory_map =
    std::unordered_map<std::string, std::shared_ptr<LinOpFactory>>;
using criterionfactory_map =
    std::unordered_map<std::string, std::shared_ptr<stop::CriterionFactory>>;


template <>
linop_map& registry::get_map_impl<linop_map>()
{
    return linop_map_;
}

template <>
linopfactory_map& registry::get_map_impl<linopfactory_map>()
{
    return linopfactory_map_;
}

template <>
criterionfactory_map& registry::get_map_impl<criterionfactory_map>()
{
    return criterionfactory_map_;
}

template <>
const linop_map& registry::get_map_impl<linop_map>() const
{
    return linop_map_;
}

template <>
const linopfactory_map& registry::get_map_impl<linopfactory_map>() const
{
    return linopfactory_map_;
}

template <>
const criterionfactory_map& registry::get_map_impl<criterionfactory_map>() const
{
    return criterionfactory_map_;
}


}  // namespace config
}  // namespace gko
