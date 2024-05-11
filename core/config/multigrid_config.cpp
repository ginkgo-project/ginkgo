// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/multigrid/fixed_coarsening.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/solver/multigrid.hpp>


#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::Pgm>(
    const pnode& config, const registry& context,
    const gko::config::type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, multigrid::Pgm>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
parse<LinOpFactoryType::FixedCoarsening>(const pnode& config,
                                         const registry& context,
                                         const gko::config::type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, multigrid::FixedCoarsening>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
parse<LinOpFactoryType::Multigrid>(const pnode& config, const registry& context,
                                   const gko::config::type_descriptor& td)
{
    auto updated = update_type(config, td);
    return solver::Multigrid::parse(config, context, updated);
}


}  // namespace config
}  // namespace gko
