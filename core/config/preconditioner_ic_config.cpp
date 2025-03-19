// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/config/parse_macro.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


class IcSolverHelper {
public:
    template <typename IndexType>
    class Configurator {
    public:
        static typename gko::preconditioner::Ic<gko::LinOp,
                                                IndexType>::parameters_type
        parse(const pnode& config, const registry& context,
              const type_descriptor& td_for_child)
        {
            return gko::preconditioner::Ic<gko::LinOp, IndexType>::parse(
                config, context, td_for_child);
        }
    };
};


template <>
deferred_factory_parameter<gko::LinOpFactory> parse<LinOpFactoryType::Ic>(
    const pnode& config, const registry& context, const type_descriptor& td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, IcSolverHelper::Configurator>(
        config, context, updated,
        make_type_selector(updated.get_index_typestr(), index_type_list()));
}


}  // namespace config
}  // namespace gko
