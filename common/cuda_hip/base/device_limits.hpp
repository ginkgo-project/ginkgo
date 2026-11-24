// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_DEVICE_LIMITS_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_DEVICE_LIMITS_HPP_


#include <memory>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/runtime.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/**
 * @return the most dynamic shared memory a block can use on exec's device,
 *         after allow_dynamic_shared_memory
 */
inline int max_shared_memory_per_block_optin(
    std::shared_ptr<const DefaultExecutor> exec)
{
    int bytes{};
#if defined(GKO_COMPILING_CUDA)
    GKO_ASSERT_NO_CUDA_ERRORS(
        cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               exec->get_device_id()));
#else
    GKO_ASSERT_NO_HIP_ERRORS(
        hipDeviceGetAttribute(&bytes, hipDeviceAttributeMaxSharedMemoryPerBlock,
                              exec->get_device_id()));
#endif
    return bytes;
}


/**
 * Lets `kernel` launch with up to `bytes` of dynamic shared memory; needed on
 * CUDA above the 48 KiB default.
 */
template <typename Kernel>
void allow_dynamic_shared_memory(Kernel kernel, int bytes)
{
#if defined(GKO_COMPILING_CUDA)
    GKO_ASSERT_NO_CUDA_ERRORS(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
#endif
}


/** @return the free memory on exec's device in bytes */
inline size_type free_device_memory(std::shared_ptr<const DefaultExecutor> exec)
{
    size_t free_bytes{};
    size_t total_bytes{};
    auto guard = exec->get_scoped_device_id_guard();
#if defined(GKO_COMPILING_CUDA)
    GKO_ASSERT_NO_CUDA_ERRORS(cudaMemGetInfo(&free_bytes, &total_bytes));
#else
    GKO_ASSERT_NO_HIP_ERRORS(hipMemGetInfo(&free_bytes, &total_bytes));
#endif
    return free_bytes;
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_BASE_DEVICE_LIMITS_HPP_
