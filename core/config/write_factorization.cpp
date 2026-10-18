// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/ilu.hpp>

#include "core/config/write_helper.hpp"


namespace gko {
namespace config {


// mirrors Ilu::parse in core/factorization/ilu.cpp
template <typename ValueType, typename IndexType>
pnode parameter_writer<factorization::Ilu<ValueType, IndexType>>::write(
    const typename factorization::Ilu<ValueType, IndexType>::parameters_type&
        params)
{
    auto map = make_config_map<ValueType>("factorization::Ilu");
    map["l_strategy"] = write_value(params.l_strategy);
    map["u_strategy"] = write_value(params.u_strategy);
    map["skip_sorting"] = write_value(params.skip_sorting);
    map["algorithm"] = write_value(params.algorithm);
    return pnode{map};
}

#define GKO_DECLARE_FACTORIZATION_ILU_WRITER(ValueType, IndexType) \
    struct parameter_writer<factorization::Ilu<ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_ILU_WRITER);


// mirrors Ic::parse in core/factorization/ic.cpp
template <typename ValueType, typename IndexType>
pnode parameter_writer<factorization::Ic<ValueType, IndexType>>::write(
    const typename factorization::Ic<ValueType, IndexType>::parameters_type&
        params)
{
    auto map = make_config_map<ValueType>("factorization::Ic");
    map["l_strategy"] = write_value(params.l_strategy);
    map["skip_sorting"] = write_value(params.skip_sorting);
    map["both_factors"] = write_value(params.both_factors);
    map["algorithm"] = write_value(params.algorithm);
    return pnode{map};
}

#define GKO_DECLARE_FACTORIZATION_IC_WRITER(ValueType, IndexType) \
    struct parameter_writer<factorization::Ic<ValueType, IndexType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_IC_WRITER);


}  // namespace config
}  // namespace gko
