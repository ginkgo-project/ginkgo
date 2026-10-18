// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>

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


// mirrors Bicgstab::parse in core/solver/bicgstab.cpp
template <typename ValueType>
pnode parameter_writer<solver::Bicgstab<ValueType>>::write(
    const typename solver::Bicgstab<ValueType>::parameters_type& params)
{
    auto map = make_config_map<ValueType>("solver::Bicgstab");
    write_common_solver(map, params);
    return pnode{map};
}

#define GKO_DECLARE_BICGSTAB_WRITER(ValueType) \
    struct parameter_writer<solver::Bicgstab<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_WRITER);


// mirrors Gmres::parse in core/solver/gmres.cpp
template <typename ValueType>
pnode parameter_writer<solver::Gmres<ValueType>>::write(
    const typename solver::Gmres<ValueType>::parameters_type& params)
{
    auto map = make_config_map<ValueType>("solver::Gmres");
    write_common_solver(map, params);
    map["krylov_dim"] = write_value(params.krylov_dim);
    map["restart_ratio"] = write_value(params.restart_ratio);
    map["flexible"] = write_value(params.flexible);
    map["ortho_method"] = write_value(params.ortho_method);
    return pnode{map};
}

#define GKO_DECLARE_GMRES_WRITER(ValueType) \
    struct parameter_writer<solver::Gmres<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_WRITER);


// mirrors Ir::parse in core/solver/ir.cpp. Ir does not read a preconditioner.
template <typename ValueType>
pnode parameter_writer<solver::Ir<ValueType>>::write(
    const typename solver::Ir<ValueType>::parameters_type& params)
{
    auto map = make_config_map<ValueType>("solver::Ir");
    map["criteria"] = write_criteria(params.criteria);
    // an unset solver is the identity (see the Ir constructor)
    write_factory_with_identity_default(map, "solver", params.solver);
    write_object(map, "generated_solver", params.generated_solver);
    map["relaxation_factor"] = write_value(params.relaxation_factor);
    map["default_initial_guess"] = write_value(params.default_initial_guess);
    map["inner_initial_guess"] = write_value(params.inner_initial_guess);
    return pnode{map};
}

#define GKO_DECLARE_IR_WRITER(ValueType) \
    struct parameter_writer<solver::Ir<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IR_WRITER);


// mirrors LowerTrs::parse in core/solver/lower_trs.cpp
template <typename ValueType, typename IndexType>
pnode parameter_writer<solver::LowerTrs<ValueType, IndexType>>::write(
    const typename solver::LowerTrs<ValueType, IndexType>::parameters_type&
        params)
{
    auto map = make_config_map<ValueType>("solver::LowerTrs");
    write_common_trisolver(map, params);
    return pnode{map};
}

#define GKO_DECLARE_LOWER_TRS_WRITER(ValueType, IndexType) \
    struct parameter_writer<solver::LowerTrs<ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_TRS_WRITER);


// mirrors UpperTrs::parse in core/solver/upper_trs.cpp
template <typename ValueType, typename IndexType>
pnode parameter_writer<solver::UpperTrs<ValueType, IndexType>>::write(
    const typename solver::UpperTrs<ValueType, IndexType>::parameters_type&
        params)
{
    auto map = make_config_map<ValueType>("solver::UpperTrs");
    write_common_trisolver(map, params);
    return pnode{map};
}

#define GKO_DECLARE_UPPER_TRS_WRITER(ValueType, IndexType) \
    struct parameter_writer<solver::UpperTrs<ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_TRS_WRITER);


}  // namespace config
}  // namespace gko
