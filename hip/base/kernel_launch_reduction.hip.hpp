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
        storage[blockIdx.x] =
            finalize(reduce(warp, warp_partial[threadIdx.x], op));
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
        generic_kernel_reduction_2d<<<num_blocks, block_size>>>(
            rows, cols, fn, op, [] __device__(auto v) { return v; },
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


}  // namespace hip
}  // namespace kernels
}  // namespace gko
