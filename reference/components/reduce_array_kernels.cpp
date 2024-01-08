// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/reduce_array_kernels.hpp"


#include <numeric>


namespace gko {
namespace kernels {
namespace reference {
namespace components {


template <typename ValueType>
void reduce_add_array(std::shared_ptr<const DefaultExecutor> exec,
                      const array<ValueType>& arr, array<ValueType>& val)
{
    val.get_data()[0] = std::accumulate(arr.get_const_data(),
                                        arr.get_const_data() + arr.get_size(),
                                        val.get_const_data()[0]);
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_REDUCE_ADD_ARRAY_KERNEL);


}  // namespace components
}  // namespace reference
}  // namespace kernels
}  // namespace gko
