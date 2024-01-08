// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_SOLVER_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch_solver.hpp"
#endif


namespace gko {
namespace kernels {
namespace dpcpp {


template <typename KernelFunction, typename... KernelArgs>
void generic_kernel_2d_solver(sycl::handler& cgh, int64 rows, int64 cols,
                              int64 default_stride, KernelFunction fn,
                              KernelArgs... args)
{
    cgh.parallel_for(sycl::range<1>{static_cast<std::size_t>(rows * cols)},
                     [=](sycl::id<1> idx) {
                         auto row = static_cast<int64>(idx[0] / cols);
                         auto col = static_cast<int64>(idx[0] % cols);
                         fn(row, col,
                            device_unpack_solver_impl<KernelArgs>::unpack(
                                args, default_stride)...);
                     });
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel_solver(std::shared_ptr<const DpcppExecutor> exec,
                       KernelFunction fn, dim<2> size, size_type default_stride,
                       KernelArgs&&... args)
{
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        kernels::dpcpp::generic_kernel_2d_solver(
            cgh, static_cast<int64>(size[0]), static_cast<int64>(size[1]),
            static_cast<int64>(default_stride), fn,
            kernels::dpcpp::map_to_device(args)...);
    });
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
