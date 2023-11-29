// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/multigrid/fixed_coarsening.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/solver/multigrid.hpp>


#include "core/config/config.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace config {


template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Pgm>(const pnode& config,
                                         const registry& context,
                                         gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, multigrid::Pgm>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::FixedCoarsening>(
    const pnode& config, const registry& context,
    gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, multigrid::FixedCoarsening>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Multigrid>(const pnode& config,
                                               const registry& context,
                                               gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return solver::Multigrid::build_from_config(config, context, updated);
}


}  // namespace config
}  // namespace gko
