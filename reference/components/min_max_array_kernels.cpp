// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/min_max_array_kernels.hpp"


#include <numeric>


namespace gko {
namespace kernels {
namespace reference {
namespace components {


template <typename IndexType>
void max_array(std::shared_ptr<const DefaultExecutor> exec,
               const array<IndexType>& arr, IndexType& val)
{
    if (arr.get_size() == 0) {
        val == std::numeric_limits<IndexType>::min();
    } else {
        val = *std::max_element(arr.get_const_data(),
                                arr.get_const_data() + arr.get_size());
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_MAX_ARRAY_KERNEL);


template <typename IndexType>
void min_array(std::shared_ptr<const DefaultExecutor> exec,
               const array<IndexType>& arr, IndexType& val)
{
    if (arr.get_size() == 0) {
        val == std::numeric_limits<IndexType>::max();
    } else {
        val = *std::min_element(arr.get_const_data(),
                                arr.get_const_data() + arr.get_size());
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_MIN_ARRAY_KERNEL);


}  // namespace components
}  // namespace reference
}  // namespace kernels
}  // namespace gko
