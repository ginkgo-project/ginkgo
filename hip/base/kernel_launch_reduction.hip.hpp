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

#ifndef GKO_COMMON_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#error \
    "This file can only be used from inside common/base/kernel_launch_reduction.hpp"
#endif


#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/device_guard.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
__global__ __launch_bounds__(
    default_block_size) void generic_kernel_reduction_1d(int64 size,
                                                         KernelFunction fn,
                                                         ReductionOp op,
                                                         FinalizeOp finalize,
                                                         ValueType init,
                                                         ValueType *storage,
                                                         KernelArgs... args)
{
    __shared__
        UninitializedArray<ValueType, default_block_size / config::warp_size>
            warp_partial;
    static_assert(default_block_size / config::warp_size <= config::warp_size,
                  "needs third reduction level");
    auto tidx = thread::get_thread_id_flat<int64>();
    auto grid_size = thread::get_thread_num_flat<int64>();
    auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    auto partial = init;
    for (int64 i = tidx; i < size; i += grid_size) {
        partial = op(partial, fn(i, args...));
    }
    partial = reduce(warp, partial, op);
    if (warp.thread_rank() == 0) {
        warp_partial[threadIdx.x / config::warp_size] = partial;
    }
    __syncthreads();
    if (threadIdx.x < config::warp_size) {
        partial = reduce(warp, warp_partial[threadIdx.x], op);
        if (threadIdx.x == 0) {
            storage[blockIdx.x] = finalize(partial);
        }
    }
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
__global__ __launch_bounds__(
    default_block_size) void generic_kernel_reduction_2d(int64 rows, int64 cols,
                                                         KernelFunction fn,
                                                         ReductionOp op,
                                                         FinalizeOp finalize,
                                                         ValueType init,
                                                         ValueType *storage,
                                                         KernelArgs... args)
{
    __shared__
        UninitializedArray<ValueType, default_block_size / config::warp_size>
            warp_partial;
    static_assert(default_block_size / config::warp_size <= config::warp_size,
                  "needs third reduction level");
    auto tidx = thread::get_thread_id_flat<int64>();
    auto grid_size = thread::get_thread_num_flat<int64>();
    auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    auto partial = init;
    for (int64 i = tidx; i < rows * cols; i += grid_size) {
        const auto row = i / cols;
        const auto col = i % cols;
        partial = op(partial, fn(row, col, args...));
    }
    partial = reduce(warp, partial, op);
    if (warp.thread_rank() == 0) {
        warp_partial[threadIdx.x / config::warp_size] = partial;
    }
    __syncthreads();
    if (threadIdx.x < config::warp_size) {
        storage[blockIdx.x] =
            finalize(reduce(warp, warp_partial[threadIdx.x], op));
    }
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction(std::shared_ptr<const HipExecutor> exec,
                          KernelFunction fn, ReductionOp op,
                          FinalizeOp finalize, ValueType init,
                          ValueType *result, size_type size,
                          KernelArgs &&... args)
{
    constexpr int oversubscription = 4;
    gko::hip::device_guard guard{exec->get_device_id()};
    constexpr auto block_size = default_block_size;
    const auto num_blocks = std::min<int64>(
        ceildiv(size, block_size), exec->get_num_warps() * oversubscription);
    if (num_blocks > 1) {
        Array<ValueType> partial{exec, static_cast<size_type>(num_blocks)};
        hipLaunchKernelGGL(
            generic_kernel_reduction_1d, num_blocks, block_size, 0, 0,
            static_cast<int64>(size), fn, op,
            [] __device__(auto v) { return v; }, as_hip_type(init),
            as_hip_type(partial.get_data()), map_to_device(args)...);
        hipLaunchKernelGGL(
            generic_kernel_reduction_1d, 1, block_size, 0, 0,
            static_cast<int64>(num_blocks),
            [] __device__(auto i, auto v) { return v[i]; }, op, finalize,
            as_hip_type(init), as_hip_type(result),
            as_hip_type(partial.get_const_data()));
    } else {
        hipLaunchKernelGGL(generic_kernel_reduction_1d, 1, block_size, 0, 0,
                           static_cast<int64>(size), fn, op, finalize,
                           as_hip_type(init), as_hip_type(result),
                           map_to_device(args)...);
    }
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction(std::shared_ptr<const HipExecutor> exec,
                          KernelFunction fn, ReductionOp op,
                          FinalizeOp finalize, ValueType init,
                          ValueType *result, dim<2> size, KernelArgs &&... args)
{
    constexpr int oversubscription = 4;
    gko::hip::device_guard guard{exec->get_device_id()};
    constexpr auto block_size = default_block_size;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto num_blocks =
        std::min<int64>(ceildiv(rows * cols, block_size),
                        exec->get_num_warps() * oversubscription);
    if (num_blocks > 1) {
        Array<ValueType> partial{exec, static_cast<size_type>(num_blocks)};
        hipLaunchKernelGGL(
            generic_kernel_reduction_2d, num_blocks, block_size, 0, 0, rows,
            cols, fn, op, [] __device__(auto v) { return v; },
            as_hip_type(init), as_hip_type(partial.get_data()),
            map_to_device(args)...);
        hipLaunchKernelGGL(
            generic_kernel_reduction_1d, 1, block_size, 0, 0,
            static_cast<int64>(num_blocks),
            [] __device__(auto i, auto v) { return v[i]; }, op, finalize,
            as_hip_type(init), as_hip_type(result),
            as_hip_type(partial.get_const_data()));
    } else {
        hipLaunchKernelGGL(generic_kernel_reduction_2d, 1, block_size, 0, 0,
                           rows, cols, fn, op, finalize, as_hip_type(init),
                           as_hip_type(result), map_to_device(args)...);
    }
}


template <int subwarp_size, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp, typename... KernelArgs>
__global__
    __launch_bounds__(default_block_size) void generic_kernel_row_reduction_2d(
        int64 rows, int64 cols, int64 col_parts, KernelFunction fn,
        ReductionOp op, FinalizeOp finalize, ValueType init, ValueType *result,
        int64 result_stride, KernelArgs... args)
{
    const auto idx = thread::get_subwarp_id_flat<subwarp_size, int64>();
    const auto row = idx % rows;
    const auto col_part = idx / rows;
    if (col_part >= col_parts) {
        return;
    }
    const auto cols_per_part = ceildiv(cols, col_parts);
    // TODO use boundaries divisible by subwarp_size
    const auto begin = cols_per_part * col_part;
    const auto end = min(begin + cols_per_part, cols);
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto partial = init;
    for (auto col = begin + subwarp.thread_rank(); col < end;
         col += subwarp_size) {
        partial = op(partial, fn(row, col, args...));
    }
    partial = reduce(subwarp, partial, op);
    result[(row * col_parts + col_part) * result_stride] = finalize(partial);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
__global__
    __launch_bounds__(default_block_size) void generic_kernel_col_reduction_2d(
        int64 rows, int64 cols, int64 row_parts, KernelFunction fn,
        ReductionOp op, FinalizeOp finalize, ValueType init, ValueType *result,
        KernelArgs... args)
{
    const auto idx = thread::get_thread_id_flat<int64>();
    const auto col = idx % cols;
    const auto row_part = idx / cols;
    if (row_part >= row_parts) {
        return;
    }
    const auto rows_per_part = ceildiv(rows, row_parts);
    const auto begin = rows_per_part * row_part;
    const auto end = min(begin + rows_per_part, rows);
    auto partial = init;
    for (auto row = begin; row < end; row++) {
        partial = op(partial, fn(row, col, args...));
    }
    result[col * row_parts + row_part] = finalize(partial);
}


template <int subwarp_size, typename ValueType, typename ReductionOp,
          typename FinalizeOp>
__global__
    __launch_bounds__(default_block_size) void generic_kernel_reduction_finalize_2d(
        int64 num_results, int64 num_parts, ReductionOp op, FinalizeOp finalize,
        ValueType init, const ValueType *input, int64 result_stride,
        ValueType *result)
{
    const auto idx = thread::get_subwarp_id_flat<subwarp_size, int64>();
    if (idx >= num_results) {
        return;
    }
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto partial = init;
    for (int64 part = subwarp.thread_rank(); part < num_parts;
         part += subwarp_size) {
        partial = op(partial, input[idx * num_parts + part]);
    }
    partial = reduce(subwarp, partial, op);
    if (subwarp.thread_rank() == 0) {
        result[idx * result_stride] = finalize(partial);
    }
}


namespace {


template <int subwarp_size, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp, typename... KernelArgs>
void run_generic_kernel_row_reduction(syn::value_list<int, subwarp_size>,
                                      int64 rows, int64 cols, int64 col_parts,
                                      KernelFunction fn, ReductionOp op,
                                      FinalizeOp finalize, ValueType init,
                                      ValueType *result, int64 result_stride,
                                      KernelArgs... args)
{
    constexpr auto block_size = default_block_size;
    const auto num_blocks = ceildiv(rows * cols * subwarp_size, block_size);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(generic_kernel_row_reduction_2d<subwarp_size>),
        num_blocks, block_size, 0, 0, rows, cols, col_parts, fn, op, finalize,
        as_hip_type(init), as_hip_type(result), result_stride, args...);
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_run_generic_kernel_row_reduction,
                                    run_generic_kernel_row_reduction)


template <int subwarp_size, typename ValueType, typename ReductionOp,
          typename FinalizeOp>
void run_kernel_reduction_finalize(syn::value_list<int, subwarp_size>,
                                   int64 num_results, int64 num_parts,
                                   ReductionOp op, FinalizeOp finalize,
                                   ValueType init, const ValueType *input,
                                   int64 result_stride, ValueType *result)
{
    constexpr auto block_size = default_block_size;
    const auto num_blocks = ceildiv(num_results * subwarp_size, block_size);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(generic_kernel_reduction_finalize_2d<subwarp_size>),
        num_blocks, block_size, 0, 0, num_results, num_parts, op, finalize,
        as_hip_type(init), as_hip_type(input),
        static_cast<int64>(result_stride), as_hip_type(result));
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_run_kernel_reduction_finalize,
                                    run_kernel_reduction_finalize)


}  // namespace


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_row_reduction(std::shared_ptr<const HipExecutor> exec,
                              KernelFunction fn, ReductionOp op,
                              FinalizeOp finalize, ValueType init,
                              ValueType *result, size_type result_stride,
                              dim<2> size, KernelArgs &&... args)
{
    using subwarp_sizes =
        syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;
    constexpr int oversubscription = 4;
    gko::hip::device_guard guard{exec->get_device_id()};
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto resources = exec->get_num_warps() * oversubscription;
    const auto col_parts = 1;  // TODO tune
    if (col_parts > 1) {
        Array<ValueType> partial{exec,
                                 static_cast<size_type>(col_parts * rows)};
        select_run_generic_kernel_row_reduction(
            subwarp_sizes(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size >= cols ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), rows, cols, col_parts,
            fn, op, [] __device__(auto i) { return i; }, init,
            partial.get_data(), 1, map_to_device(args)...);
        select_run_kernel_reduction_finalize(
            subwarp_sizes(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size >= col_parts ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), rows, col_parts, op,
            finalize, init, partial.get_const_data(),
            static_cast<int64>(result_stride), result);
    } else {
        select_run_generic_kernel_row_reduction(
            subwarp_sizes(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size >= cols ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), rows, cols, 1, fn, op,
            finalize, init, result, static_cast<int64>(result_stride),
            map_to_device(args)...);
    }
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_col_reduction(std::shared_ptr<const HipExecutor> exec,
                              KernelFunction fn, ReductionOp op,
                              FinalizeOp finalize, ValueType init,
                              ValueType *result, dim<2> size,
                              KernelArgs &&... args)
{
    constexpr int oversubscription = 4;
    gko::hip::device_guard guard{exec->get_device_id()};
    constexpr auto block_size = default_block_size;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto resources =
        exec->get_num_warps() * config::warp_size * oversubscription;
    const auto num_blocks = ceildiv(rows * cols, block_size);
    const auto row_parts = 1;  // TODO tune
    if (row_parts > 1) {
        Array<ValueType> partial{exec,
                                 static_cast<size_type>(row_parts * cols)};
        hipLaunchKernelGGL(
            generic_kernel_col_reduction_2d, num_blocks, block_size, 0, 0, rows,
            cols, row_parts, fn, op, [] __device__(auto i) { return i; },
            as_hip_type(init), as_hip_type(partial.get_data()),
            map_to_device(args)...);
        using subwarp_sizes =
            syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;
        select_run_kernel_reduction_finalize(
            subwarp_sizes(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size >= row_parts ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), cols, row_parts, op,
            finalize, as_hip_type(init), as_hip_type(partial.get_const_data()),
            1, as_hip_type(result));
    } else {
        hipLaunchKernelGGL(generic_kernel_col_reduction_2d, num_blocks,
                           block_size, 0, 0, rows, cols, 1, fn, op, finalize,
                           as_hip_type(init), as_hip_type(result),
                           map_to_device(args)...);
    }
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko
