// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/solver/cg.hpp>

#include "core/config/write_helper.hpp"


namespace gko {
namespace config {


// mirrors Cg::parse in core/solver/cg.cpp
template <typename ValueType>
pnode parameter_writer<solver::Cg<ValueType>>::write(
    const typename solver::Cg<ValueType>::parameters_type& params)
{
    auto map = make_config_map<ValueType>("solver::Cg");
    write_common_solver(map, params);
    return pnode{map};
}

#define GKO_DECLARE_CG_WRITER(ValueType) \
    struct parameter_writer<solver::Cg<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_WRITER);


}  // namespace config
}  // namespace gko
