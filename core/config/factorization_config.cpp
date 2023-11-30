// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/factorization/par_ict.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>
#include <ginkgo/core/factorization/par_ilut.hpp>


#include "core/config/config.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace config {


template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Factorization_Ic>(
    const pnode& config, const registry& context,
    gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::Ic>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}


template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Factorization_Ilu>(
    const pnode& config, const registry& context,
    gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::Ilu>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}


template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Cholesky>(const pnode& config,
                                              const registry& context,
                                              gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory,
                    gko::experimental::factorization::Cholesky>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Lu>(const pnode& config,
                                        const registry& context,
                                        gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::experimental::factorization::Lu>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::ParIlu>(const pnode& config,
                                            const registry& context,
                                            gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::ParIlu>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::ParIlut>(const pnode& config,
                                             const registry& context,
                                             gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::ParIlut>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::ParIc>(const pnode& config,
                                           const registry& context,
                                           gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::ParIc>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
build_from_config<LinOpFactoryType::ParIct>(const pnode& config,
                                            const registry& context,
                                            gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::ParIct>(
        updated.first + "," + updated.second, config, context, updated,
        value_type_list(), index_type_list());
}


}  // namespace config
}  // namespace gko
