// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_SOLVER_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch_solver.hpp"
#endif


namespace gko {
namespace kernels {
namespace omp {


template <typename KernelFunction, typename... KernelArgs>
void run_kernel_solver(std::shared_ptr<const OmpExecutor> exec,
                       KernelFunction fn, dim<2> size, size_type default_stride,
                       KernelArgs&&... args)
{
    run_kernel_impl(exec, fn, size,
                    device_unpack(map_to_device(std::forward<KernelArgs>(args)),
                                  static_cast<int64>(default_stride))...);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
