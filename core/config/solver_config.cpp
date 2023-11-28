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
#define PARSE(_type)                                                \
    template <>                                                     \
    deferred_factory_parameter<gko::LinOpFactory>                   \
    parse<LinOpFactoryType::_type>(const pnode& config,             \
                                   const registry& context,         \
                                   gko::config::type_descriptor td) \
    {                                                               \
        auto updated = update_type(config, td);                     \
        return dispatch<gko::LinOpFactory, gko::solver::_type>(     \
            config, context, updated,                               \
            make_type_selector(updated.get_value_typestr(),         \
                               value_type_list()));                 \
    }

PARSE(Cg)
PARSE(Bicg)
PARSE(Bicgstab)
PARSE(Cgs)
PARSE(Fcg)
PARSE(Ir)
PARSE(Idr)
PARSE(Gcr)
PARSE(Gmres)
PARSE(CbGmres)


template <>
deferred_factory_parameter<gko::LinOpFactory>
parse<LinOpFactoryType::Direct>(const pnode& config,
                                            const registry& context,
                                            gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::experimental::solver::Direct>(
        config, context, updated,  make_type_selector(updated.get_value_typestr(), value_type_list()),  make_type_selector(updated.get_index_typestr(), index_type_list())
        );
}


template <>
deferred_factory_parameter<gko::LinOpFactory>
parse<LinOpFactoryType::LowerTrs>(const pnode& config,
                                              const registry& context,
                                              gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::solver::LowerTrs>(
         config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),  make_type_selector(updated.get_index_typestr(), index_type_list()));
}

template <>
deferred_factory_parameter<gko::LinOpFactory>
parse<LinOpFactoryType::UpperTrs>(const pnode& config,
                                              const registry& context,
                                              gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, gko::solver::UpperTrs>(
         config, context, updated,
        make_type_selector(updated.get_value_typestr(), value_type_list()),  make_type_selector(updated.get_index_typestr(), index_type_list()));
}


}  // namespace config
}  // namespace gko
