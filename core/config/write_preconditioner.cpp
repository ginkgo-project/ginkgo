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

GKO_INSTANTIATE_WRITER_VALUE_AND_INDEX_TYPE(preconditioner::Jacobi);


// the unset solvers and factorization are not the identity
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

GKO_INSTANTIATE_WRITER_VALUE_AND_INDEX_TYPE(forward_ilu);
GKO_INSTANTIATE_WRITER_VALUE_AND_INDEX_TYPE(reverse_ilu);


// the unset solver and factorization are not the identity
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

GKO_INSTANTIATE_WRITER_VALUE_AND_INDEX_TYPE(preconditioner::Ic);


// an unset excess solver is not the identity
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

GKO_INSTANTIATE_WRITER_VALUE_AND_INDEX_TYPE(preconditioner::LowerIsai);
GKO_INSTANTIATE_WRITER_VALUE_AND_INDEX_TYPE(preconditioner::UpperIsai);
GKO_INSTANTIATE_WRITER_VALUE_AND_INDEX_TYPE(preconditioner::GeneralIsai);
GKO_INSTANTIATE_WRITER_VALUE_AND_INDEX_TYPE(preconditioner::SpdIsai);


}  // namespace config
}  // namespace gko
