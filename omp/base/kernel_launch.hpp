/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_COMMON_BASE_KERNEL_LAUNCH_HPP_
#error "This file can only be used from inside common/base/kernel_launch.hpp"
#endif


namespace gko {
namespace kernels {
namespace omp {


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                size_type size, KernelArgs &&... args)
{
#pragma omp parallel for
    for (int64 i = 0; i < static_cast<int64>(size); i++) {
        [&]() { fn(i, map_to_device(args)...); }();
    }
}


template <int64 cols, typename KernelFunction, typename... MappedKernelArgs>
void run_kernel_fixed_cols_impl(std::shared_ptr<const OmpExecutor> exec,
                                KernelFunction fn, dim<2> size,
                                MappedKernelArgs... args)
{
    const auto rows = static_cast<int64>(size[0]);
#pragma omp parallel for
    for (int64 row = 0; row < rows; row++) {
#pragma unroll
        for (int64 col = 0; col < cols; col++) {
            [&]() { fn(row, col, args...); }();
        }
    }
}

template <int64 remainder_cols, int64 block_size, typename KernelFunction,
          typename... MappedKernelArgs>
void run_kernel_blocked_cols_impl(std::shared_ptr<const OmpExecutor> exec,
                                  KernelFunction fn, dim<2> size,
                                  MappedKernelArgs... args)
{
    static_assert(remainder_cols < block_size, "remainder too large");
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto rounded_cols = cols / block_size * block_size;
    GKO_ASSERT(rounded_cols + remainder_cols == cols);
#pragma omp parallel for
    for (int64 row = 0; row < rows; row++) {
        for (int64 base_col = 0; base_col < rounded_cols;
             base_col += block_size) {
#pragma unroll
            for (int64 i = 0; i < block_size; i++) {
                [&]() { fn(row, base_col + i, args...); }();
            }
        }
#pragma unroll
        for (int64 i = 0; i < remainder_cols; i++) {
            [&]() { fn(row, rounded_cols + i, args...); }();
        }
    }
}

template <typename KernelFunction, typename... MappedKernelArgs>
void run_kernel_impl(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                     dim<2> size, MappedKernelArgs... args)
{
    const auto rows = size[0];
    const auto cols = size[1];
    constexpr int64 block_size = 4;
    if (cols <= 0) {
        return;
    }
    if (cols == 1) {
        run_kernel_fixed_cols_impl<1>(exec, fn, size, args...);
        return;
    }
    if (cols == 2) {
        run_kernel_fixed_cols_impl<2>(exec, fn, size, args...);
        return;
    }
    if (cols == 3) {
        run_kernel_fixed_cols_impl<3>(exec, fn, size, args...);
        return;
    }
    if (cols == 4) {
        run_kernel_fixed_cols_impl<4>(exec, fn, size, args...);
        return;
    }
    const auto rem_cols = cols % block_size;
    if (rem_cols == 0) {
        run_kernel_blocked_cols_impl<0, block_size>(exec, fn, size, args...);
        return;
    }
    if (rem_cols == 1) {
        run_kernel_blocked_cols_impl<1, block_size>(exec, fn, size, args...);
        return;
    }
    if (rem_cols == 2) {
        run_kernel_blocked_cols_impl<2, block_size>(exec, fn, size, args...);
        return;
    }
    if (rem_cols == 3) {
        run_kernel_blocked_cols_impl<3, block_size>(exec, fn, size, args...);
        return;
    }
    // should be unreachable
    GKO_ASSERT(false);
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                dim<2> size, KernelArgs &&... args)
{
    run_kernel_impl(exec, fn, size, map_to_device(args)...);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
