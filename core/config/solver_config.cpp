// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/config/config.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/solver_config.hpp"


namespace gko {
namespace config {

// for valuetype only
#define BUILD_FROM_CONFIG(_type)                                         \
    template <>                                                          \
    deferred_factory_parameter<gko::LinOpFactory>                        \
    build_from_config<LinOpFactoryType::_type>(                          \
        const pnode& config, const registry& context,                    \
        gko::config::type_descriptor td)                                 \
    {                                                                    \
        auto updated = update_type(config, td);                          \
        return dispatch<gko::LinOpFactory, gko::solver::_type>(          \
            updated.first, config, context, updated, value_type_list()); \
    }

BUILD_FROM_CONFIG(Cg)
BUILD_FROM_CONFIG(Bicg)
BUILD_FROM_CONFIG(Bicgstab)
BUILD_FROM_CONFIG(Cgs)
BUILD_FROM_CONFIG(Fcg)
BUILD_FROM_CONFIG(Ir)
BUILD_FROM_CONFIG(Idr)
BUILD_FROM_CONFIG(Gcr)
BUILD_FROM_CONFIG(Gmres)
BUILD_FROM_CONFIG(CbGmres)


template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Direct>(const pnode& config,
                                            const registry& context,
                                            gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::experimental::solver::Direct>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}


template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::LowerTrs>(const pnode& config,
                                              const registry& context,
                                              gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::solver::LowerTrs>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::UpperTrs>(const pnode& config,
                                              const registry& context,
                                              gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::solver::UpperTrs>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}


}  // namespace config
}  // namespace gko
