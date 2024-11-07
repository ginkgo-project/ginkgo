// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cuda/solver/batch_cg_launch.cuh"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/solver/batch_cg_kernels.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_cg_kernels.hpp"
#include "core/solver/batch_dispatch.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_cg {


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_num_threads_per_block(std::shared_ptr<const DefaultExecutor> exec,
                              const int num_rows)
{
    int num_warps = std::max(num_rows / 4, 2);
    constexpr int warp_sz = static_cast<int>(config::warp_size);
    const int min_block_size = 2 * warp_sz;
    const int device_max_threads =
        (std::max(num_rows, min_block_size) / warp_sz) * warp_sz;
    auto get_num_regs = [](const auto func) {
        cudaFuncAttributes funcattr;
        cudaFuncGetAttributes(&funcattr, func);
        return funcattr.numRegs;
    };
    const int num_regs_used = std::max(
        get_num_regs(
            batch_single_kernels::apply_kernel<StopType, 5, true, PrecType,
                                               LogType, BatchMatrixType,
                                               ValueType>),
        get_num_regs(
            batch_single_kernels::apply_kernel<StopType, 0, false, PrecType,
                                               LogType, BatchMatrixType,
                                               ValueType>));
    int max_regs_blk = 0;
    cudaDeviceGetAttribute(&max_regs_blk, cudaDevAttrMaxRegistersPerBlock,
                           exec->get_device_id());
    const int max_threads_regs =
        ((max_regs_blk / static_cast<int>(num_regs_used)) / warp_sz) * warp_sz;
    int max_threads = std::min(max_threads_regs, device_max_threads);
    max_threads = max_threads <= max_cg_threads ? max_threads : max_cg_threads;
    return std::max(std::min(num_warps * warp_sz, max_threads), min_block_size);
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
int get_max_dynamic_shared_memory(std::shared_ptr<const DefaultExecutor> exec)
{
    int shmem_per_sm = 0;
    cudaDeviceGetAttribute(&shmem_per_sm,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                           exec->get_device_id());
    GKO_ASSERT_NO_CUDA_ERRORS(cudaFuncSetAttribute(
        batch_single_kernels::apply_kernel<StopType, 5, true, PrecType, LogType,
                                           BatchMatrixType, ValueType>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 99 /*%*/));
    cudaFuncAttributes funcattr;
    cudaFuncGetAttributes(
        &funcattr,
        batch_single_kernels::apply_kernel<StopType, 5, true, PrecType, LogType,
                                           BatchMatrixType, ValueType>);
    return funcattr.maxDynamicSharedSizeBytes;
}


// begin
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_CG_GET_NUM_THREADS_PER_BLOCK);
// split
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_CG_GET_MAX_DYNAMIC_SHARED_MEMORY);
// end


}  // namespace batch_cg
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
