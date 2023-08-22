/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_SOLVER_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch_solver.hpp"
#endif


namespace gko {
namespace kernels {
namespace sycl {


template <typename KernelFunction, typename... KernelArgs>
void generic_kernel_2d_solver(::sycl::handler& cgh, int64 rows, int64 cols,
                              int64 default_stride, KernelFunction fn,
                              KernelArgs... args)
{
    cgh.parallel_for(::sycl::range<1>{static_cast<std::size_t>(rows * cols)},
                     [=](::sycl::id<1> idx) {
                         auto row = static_cast<int64>(idx[0] / cols);
                         auto col = static_cast<int64>(idx[0] % cols);
                         fn(row, col,
                            device_unpack_solver_impl<KernelArgs>::unpack(
                                args, default_stride)...);
                     });
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel_solver(std::shared_ptr<const SyclExecutor> exec,
                       KernelFunction fn, dim<2> size, size_type default_stride,
                       KernelArgs&&... args)
{
    exec->get_queue()->submit([&](::sycl::handler& cgh) {
        kernels::sycl::generic_kernel_2d_solver(
            cgh, static_cast<int64>(size[0]), static_cast<int64>(size[1]),
            static_cast<int64>(default_stride), fn,
            kernels::sycl::map_to_device(args)...);
    });
}


}  // namespace sycl
}  // namespace kernels
}  // namespace gko
