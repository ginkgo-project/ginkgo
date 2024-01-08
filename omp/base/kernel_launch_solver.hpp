// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_SOLVER_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch_solver.hpp"
#endif


namespace gko {
namespace kernels {
namespace omp {


template <typename T>
typename device_unpack_solver_impl<typename to_device_type_impl<T>::type>::type
map_to_device_solver(T&& param, int64 default_stride)
{
    return device_unpack_solver_impl<typename to_device_type_impl<T>::type>::
        unpack(to_device_type_impl<T>::map_to_device(param), default_stride);
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel_solver(std::shared_ptr<const OmpExecutor> exec,
                       KernelFunction fn, dim<2> size, size_type default_stride,
                       KernelArgs&&... args)
{
    run_kernel_impl(
        exec, fn, size,
        map_to_device_solver(args, static_cast<int64>(default_stride))...);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
