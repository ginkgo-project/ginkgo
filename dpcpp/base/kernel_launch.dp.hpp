// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch.hpp"
#endif


#include <tuple>


#include <CL/sycl.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {


namespace device_std = std;


template <typename KernelFunction, typename... KernelArgs>
void generic_kernel_1d(sycl::handler& cgh, int64 size, KernelFunction fn,
                       KernelArgs... args)
{
    cgh.parallel_for(sycl::range<1>{static_cast<std::size_t>(size)},
                     [=](sycl::id<1> idx_id) {
                         auto idx = static_cast<int64>(idx_id[0]);
                         fn(idx, args...);
                     });
}


template <typename KernelFunction, typename... KernelArgs>
void generic_kernel_2d(sycl::handler& cgh, int64 rows, int64 cols,
                       KernelFunction fn, KernelArgs... args)
{
    cgh.parallel_for(sycl::range<1>{static_cast<std::size_t>(rows * cols)},
                     [=](sycl::id<1> idx) {
                         auto row = static_cast<int64>(idx[0]) / cols;
                         auto col = static_cast<int64>(idx[0]) % cols;
                         fn(row, col, args...);
                     });
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const DpcppExecutor> exec, KernelFunction fn,
                size_type size, KernelArgs&&... args)
{
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        generic_kernel_1d(cgh, static_cast<int64>(size), fn,
                          map_to_device(args)...);
    });
}

template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const DpcppExecutor> exec, KernelFunction fn,
                dim<2> size, KernelArgs&&... args)
{
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        generic_kernel_2d(cgh, static_cast<int64>(size[0]),
                          static_cast<int64>(size[1]), fn,
                          map_to_device(args)...);
    });
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
