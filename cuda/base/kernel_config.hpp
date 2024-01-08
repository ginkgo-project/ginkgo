// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_KERNEL_CONFIG_HPP_
#define GKO_CUDA_BASE_KERNEL_CONFIG_HPP_


#include <cuda_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace cuda {
namespace detail {


template <typename ValueType>
class shared_memory_config_guard {
public:
    using value_type = ValueType;
    shared_memory_config_guard() : original_config_{}
    {
        GKO_ASSERT_NO_CUDA_ERRORS(
            cudaDeviceGetSharedMemConfig(&original_config_));

        if (sizeof(value_type) == 4) {
            GKO_ASSERT_NO_CUDA_ERRORS(
                cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
        } else if (sizeof(value_type) % 8 == 0) {
            GKO_ASSERT_NO_CUDA_ERRORS(
                cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        } else {
            GKO_ASSERT_NO_CUDA_ERRORS(
                cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault));
        }
    }


    ~shared_memory_config_guard()
    {
        // No need to exit or throw if we cant set the value back.
        cudaDeviceSetSharedMemConfig(original_config_);
    }

private:
    cudaSharedMemConfig original_config_;
};


}  // namespace detail
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_KERNEL_CONFIG_HPP_
