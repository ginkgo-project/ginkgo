// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/reduce_array_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace components {


template <typename ValueType>
void reduce_add_array(std::shared_ptr<const DefaultExecutor> exec,
                      const array<ValueType>& arr, array<ValueType>& result)
{
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto arr, auto result) {
            return i == 0 ? (arr[i] + result[0]) : arr[i];
        },
        GKO_KERNEL_REDUCE_SUM(ValueType), result.get_data(), arr.get_size(),
        arr, result);
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_REDUCE_ADD_ARRAY_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
