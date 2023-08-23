// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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


#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace config {


template <>
deferred_factory_parameter<gko::LinOpFactory>
parse<LinOpFactoryType::Factorization_Ic>(const pnode& config,
                                          const registry& context,
                                          const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::Ic>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}


template <>
deferred_factory_parameter<gko::LinOpFactory>
parse<LinOpFactoryType::Factorization_Ilu>(const pnode& config,
                                           const registry& context,
                                           const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::Ilu>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::Cholesky>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory,
                    gko::experimental::factorization::Cholesky>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::Lu>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::experimental::factorization::Lu>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::ParIlu>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::ParIlu>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::ParIlut>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::ParIlut>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::ParIc>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::ParIc>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::ParIct>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::factorization::ParIct>(
        config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}


}  // namespace config
}  // namespace gko
