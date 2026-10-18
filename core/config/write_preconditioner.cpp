// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>

#include "core/config/write_helper.hpp"


namespace gko {
namespace config {


// mirrors Jacobi::parse in core/preconditioner/jacobi.cpp
template <typename ValueType, typename IndexType>
pnode parameter_writer<preconditioner::Jacobi<ValueType, IndexType>>::write(
    const typename preconditioner::Jacobi<ValueType,
                                          IndexType>::parameters_type& params)
{
    // Jacobi::parse can read neither of these
    if (params.block_pointers.get_size() > 0) {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           "preconditioner::Jacobi with block_pointers");
    }
    if (params.storage_optimization.is_block_wise) {
        throw NotSupported(
            __FILE__, __LINE__, __func__,
            "preconditioner::Jacobi with a block-wise storage_optimization");
    }
    auto map = make_config_map<ValueType>("preconditioner::Jacobi");
    map["max_block_size"] = write_value(params.max_block_size);
    map["max_block_stride"] = write_value(params.max_block_stride);
    map["skip_sorting"] = write_value(params.skip_sorting);
    map["storage_optimization"] =
        write_value(params.storage_optimization.of_all_blocks);
    map["accuracy"] = write_value(params.accuracy);
    map["aggregate_l1"] = write_value(params.aggregate_l1);
    return pnode{map};
}

#define GKO_DECLARE_JACOBI_WRITER(ValueType, IndexType) \
    struct parameter_writer<preconditioner::Jacobi<ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_WRITER);


// mirrors Ilu::parse in core/preconditioner/ilu.cpp, plus the reverse_apply
// that core/config/preconditioner_ilu_config.cpp selects the class with.
// Unset solvers and factorization are not the identity, but a LowerTrs, an
// UpperTrs and a ParIlu (see the Ilu constructor), so they use write_factory.
template <typename ValueType, bool ReverseApply, typename IndexType>
pnode parameter_writer<
    preconditioner::Ilu<ValueType, ReverseApply, IndexType>>::
    write(const typename preconditioner::Ilu<
          ValueType, ReverseApply, IndexType>::parameters_type& params)
{
    auto map = make_config_map<ValueType>("preconditioner::Ilu");
    map["reverse_apply"] = write_value(ReverseApply);
    write_factory(map, "l_solver", params.l_solver_factory);
    write_factory(map, "u_solver", params.u_solver_factory);
    write_factory(map, "factorization", params.factorization_factory);
    return pnode{map};
}

#define GKO_DECLARE_ILU_FALSE_WRITER(ValueType, IndexType) \
    struct parameter_writer<preconditioner::Ilu<ValueType, false, IndexType>>
#define GKO_DECLARE_ILU_TRUE_WRITER(ValueType, IndexType) \
    struct parameter_writer<preconditioner::Ilu<ValueType, true, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU_FALSE_WRITER);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU_TRUE_WRITER);


// mirrors Ic::parse in core/preconditioner/ic.cpp. An unset solver and
// factorization are a LowerTrs and a ParIc (see the Ic constructor).
template <typename ValueType, typename IndexType>
pnode parameter_writer<preconditioner::Ic<ValueType, IndexType>>::write(
    const typename preconditioner::Ic<ValueType, IndexType>::parameters_type&
        params)
{
    auto map = make_config_map<ValueType>("preconditioner::Ic");
    write_factory(map, "l_solver", params.l_solver_factory);
    write_factory(map, "factorization", params.factorization_factory);
    return pnode{map};
}

#define GKO_DECLARE_IC_WRITER(ValueType, IndexType) \
    struct parameter_writer<preconditioner::Ic<ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IC_WRITER);


// mirrors Isai::parse in core/preconditioner/isai.cpp, plus the isai_type that
// core/config/preconditioner_isai_config.cpp selects the class with. An unset
// excess solver is a Gmres or a triangular solver (see generate_inverse).
template <preconditioner::isai_type IsaiType, typename ValueType,
          typename IndexType>
pnode parameter_writer<preconditioner::Isai<IsaiType, ValueType, IndexType>>::
    write(const typename preconditioner::Isai<
          IsaiType, ValueType, IndexType>::parameters_type& params)
{
    auto map = make_config_map<ValueType>("preconditioner::Isai");
    map["isai_type"] = write_value(IsaiType);
    map["skip_sorting"] = write_value(params.skip_sorting);
    map["sparsity_power"] = write_value(params.sparsity_power);
    map["excess_limit"] = write_value(params.excess_limit);
    write_factory(map, "excess_solver_factory", params.excess_solver_factory);
    map["excess_solver_reduction"] =
        write_value(params.excess_solver_reduction);
    return pnode{map};
}

#define GKO_DECLARE_LOWER_ISAI_WRITER(ValueType, IndexType) \
    struct parameter_writer<preconditioner::Isai<           \
        preconditioner::isai_type::lower, ValueType, IndexType>>
#define GKO_DECLARE_UPPER_ISAI_WRITER(ValueType, IndexType) \
    struct parameter_writer<preconditioner::Isai<           \
        preconditioner::isai_type::upper, ValueType, IndexType>>
#define GKO_DECLARE_GENERAL_ISAI_WRITER(ValueType, IndexType) \
    struct parameter_writer<preconditioner::Isai<             \
        preconditioner::isai_type::general, ValueType, IndexType>>
#define GKO_DECLARE_SPD_ISAI_WRITER(ValueType, IndexType) \
    struct parameter_writer<preconditioner::Isai<         \
        preconditioner::isai_type::spd, ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_ISAI_WRITER);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_ISAI_WRITER);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GENERAL_ISAI_WRITER);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPD_ISAI_WRITER);


}  // namespace config
}  // namespace gko
