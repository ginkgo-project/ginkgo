// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/min_max_array_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename IndexType>
void max_array(std::shared_ptr<const DefaultExecutor> exec,
               const array<IndexType>& arr, IndexType& result)
{
    array<IndexType> device_result(exec, 1);
    device_result.fill(std::numeric_limits<IndexType>::min());
    run_kernel_reduction(
        exec, [] GKO_KERNEL(auto i, auto arr) { return arr[i]; },
        GKO_KERNEL_REDUCE_MAX(IndexType), device_result.get_data(),
        arr.get_size(), arr);
    result = exec->copy_val_to_host(device_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_MAX_ARRAY_KERNEL);


template <typename IndexType>
void min_array(std::shared_ptr<const DefaultExecutor> exec,
               const array<IndexType>& arr, IndexType& result)
{
    array<IndexType> device_result(exec, 1);
    device_result.fill(std::numeric_limits<IndexType>::max());
    run_kernel_reduction(
        exec, [] GKO_KERNEL(auto i, auto arr) { return arr[i]; },
        GKO_KERNEL_REDUCE_MIN(IndexType), device_result.get_data(),
        arr.get_size(), arr);
    result = exec->copy_val_to_host(device_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_MIN_ARRAY_KERNEL);

}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
