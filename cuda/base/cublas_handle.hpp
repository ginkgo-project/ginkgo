// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_CUBLAS_HANDLE_HPP_
#define GKO_CUDA_BASE_CUBLAS_HANDLE_HPP_


#include <cublas_v2.h>

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace cuda {
namespace cublas {


inline cublasHandle_t init(cudaStream_t stream)
{
    cublasHandle_t handle;
    GKO_ASSERT_NO_CUBLAS_ERRORS(cublasCreate(&handle));
    GKO_ASSERT_NO_CUBLAS_ERRORS(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    GKO_ASSERT_NO_CUBLAS_ERRORS(cublasSetStream(handle, stream));
    return handle;
}


inline void destroy(cublasHandle_t handle)
{
    GKO_ASSERT_NO_CUBLAS_ERRORS(cublasDestroy(handle));
}


}  // namespace cublas
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CUBLAS_HANDLE_HPP_
