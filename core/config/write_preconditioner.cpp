// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
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


}  // namespace config
}  // namespace gko
