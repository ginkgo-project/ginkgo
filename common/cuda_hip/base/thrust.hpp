// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_THRUST_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_THRUST_HPP_


#include <thrust/execution_policy.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/executor.hpp>


#if defined(GKO_COMPILING_CUDA) || \
    (defined(GKO_COMPILING_HIP) && !GINKGO_HIP_PLATFORM_HCC)
#include <thrust/system/cuda/detail/execution_policy.h>
#else
#include <thrust/system/hip/detail/execution_policy.h>
#endif


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


#if defined(GKO_COMPILING_CUDA)
inline auto thrust_policy(std::shared_ptr<const CudaExecutor> exec)
{
    return thrust::cuda::par.on(exec->get_stream());
}
#elif defined(GKO_COMPILING_HIP)
inline auto thrust_policy(std::shared_ptr<const HipExecutor> exec)
{
#if GINKGO_HIP_PLATFORM_HCC
    return thrust::hip::par.on(exec->get_stream());
#else
    return thrust::cuda::par.on(exec->get_stream());
#endif
}
#else
#error "Executor definition missing"
#endif


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_BASE_THRUST_HPP_
