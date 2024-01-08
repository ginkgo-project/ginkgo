// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_THRUST_CUH_
#define GKO_CUDA_BASE_THRUST_CUH_


#include <thrust/execution_policy.h>
#include <thrust/system/cuda/detail/execution_policy.h>


#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace kernels {
namespace cuda {


inline auto thrust_policy(std::shared_ptr<const CudaExecutor> exec)
{
    return thrust::cuda::par.on(exec->get_stream());
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_THRUST_CUH_
