// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_THRUST_HIP_HPP_
#define GKO_HIP_BASE_THRUST_HIP_HPP_


#include <thrust/execution_policy.h>
#include <thrust/system/hip/detail/execution_policy.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace kernels {
namespace hip {


inline auto thrust_policy(std::shared_ptr<const HipExecutor> exec)
{
    return thrust::hip::par.on(exec->get_stream());
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_THRUST_HIP_HPP_
