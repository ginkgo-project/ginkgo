// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch.hpp"
#endif


#include <tuple>


#include "core/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace omp {


namespace device_std = std;


namespace {


template <typename KernelFunction, typename... MappedKernelArgs>
void run_kernel_impl(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                     size_type size, MappedKernelArgs... args)
{
#pragma omp parallel for
    for (int64 i = 0; i < static_cast<int64>(size); i++) {
        [&]() { fn(i, args...); }();
    }
}


template <int block_size, int remainder_cols, typename KernelFunction,
          typename... MappedKernelArgs>
void run_kernel_sized_impl(syn::value_list<int, remainder_cols>,
                           std::shared_ptr<const OmpExecutor> exec,
                           KernelFunction fn, dim<2> size,
                           MappedKernelArgs... args)
{
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    static_assert(remainder_cols < block_size, "remainder too large");
    const auto rounded_cols = cols / block_size * block_size;
    GKO_ASSERT(rounded_cols + remainder_cols == cols);
    if (rounded_cols == 0 || cols == block_size) {
        // we group all sizes <= block_size here and unroll explicitly
        constexpr auto local_cols =
            remainder_cols == 0 ? block_size : remainder_cols;
#pragma omp parallel for
        for (int64 row = 0; row < rows; row++) {
#pragma unroll
            for (int64 col = 0; col < local_cols; col++) {
                [&]() { fn(row, col, args...); }();
            }
        }
    } else {
        // we operate in block_size blocks plus an explicitly unrolled remainder
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
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_run_kernel_sized,
                                    run_kernel_sized_impl);


template <typename KernelFunction, typename... MappedKernelArgs>
void run_kernel_impl(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                     dim<2> size, MappedKernelArgs... args)
{
    const auto cols = static_cast<int64>(size[1]);
    constexpr int block_size = 8;
    using remainders = syn::as_list<syn::range<0, block_size, 1>>;

    if (cols <= 0) {
        return;
    }
    select_run_kernel_sized(
        remainders(),
        [&](int remainder) { return remainder == cols % block_size; },
        syn::value_list<int, block_size>(), syn::type_list<>(), exec, fn, size,
        args...);
}


}  // namespace


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                size_type size, KernelArgs&&... args)
{
    run_kernel_impl(exec, fn, size, map_to_device(args)...);
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                dim<2> size, KernelArgs&&... args)
{
    run_kernel_impl(exec, fn, size, map_to_device(args)...);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
