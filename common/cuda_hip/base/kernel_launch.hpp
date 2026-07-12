// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch.hpp"
#endif


#include <thrust/tuple.h>

#include "common/cuda_hip/base/accessor.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


template <typename AccessorType>
struct to_device_type_impl<acc::range<AccessorType>&> {
    using type = std::decay_t<decltype(acc::as_device_range(
        std::declval<acc::range<AccessorType>>()))>;
    static type map_to_device(acc::range<AccessorType>& range)
    {
        return acc::as_device_range(range);
    }
};

template <typename AccessorType>
struct to_device_type_impl<const acc::range<AccessorType>&> {
    using type = std::decay_t<decltype(acc::as_device_range(
        std::declval<acc::range<AccessorType>>()))>;
    static type map_to_device(const acc::range<AccessorType>& range)
    {
        return acc::as_device_range(range);
    }
};


namespace device_std = thrust;


constexpr int default_block_size = 512;


template <typename KernelFunction, typename... KernelArgs>
__global__ __launch_bounds__(default_block_size) void generic_kernel_1d(
    int64 size, KernelFunction fn, KernelArgs... args)
{
    auto tidx = thread::get_thread_id_flat<int64>();
    if (tidx >= size) {
        return;
    }
    fn(tidx, args...);
}


template <typename KernelFunction, typename... KernelArgs>
__global__ __launch_bounds__(default_block_size) void generic_kernel_2d(
    int64 rows, int64 cols, KernelFunction fn, KernelArgs... args)
{
    auto tidx = thread::get_thread_id_flat<int64>();
    auto col = tidx % cols;
    auto row = tidx / cols;
    if (row >= rows) {
        return;
    }
    fn(row, col, args...);
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const DefaultExecutor> exec, KernelFunction fn,
                size_type size, KernelArgs&&... args)
{
    if (size > 0) {
        constexpr auto block_size = default_block_size;
        auto num_blocks = ceildiv(size, block_size);
        generic_kernel_1d<<<num_blocks, block_size, 0, exec->get_stream()>>>(
            static_cast<int64>(size), fn, map_to_device(args)...);
    }
}

template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const DefaultExecutor> exec, KernelFunction fn,
                dim<2> size, KernelArgs&&... args)
{
    if (size[0] > 0 && size[1] > 0) {
        constexpr auto block_size = default_block_size;
        auto num_blocks = ceildiv(size[0] * size[1], block_size);
        generic_kernel_2d<<<num_blocks, block_size, 0, exec->get_stream()>>>(
            static_cast<int64>(size[0]), static_cast<int64>(size[1]), fn,
            map_to_device(args)...);
    }
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
