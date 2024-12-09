// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_HIPBLAS_HANDLE_HPP_
#define GKO_HIP_BASE_HIPBLAS_HANDLE_HPP_


#if HIP_VERSION >= 50200000
#include <hipblas/hipblas.h>
#else
#include <hipblas.h>
#endif

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace hip {
namespace hipblas {


inline hipblasContext* init(hipStream_t stream)
{
    hipblasHandle_t handle;
    GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasCreate(&handle));
    GKO_ASSERT_NO_HIPBLAS_ERRORS(
        hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasSetStream(handle, stream));
    return reinterpret_cast<hipblasContext*>(handle);
}


inline void destroy_hipblas_handle(hipblasContext* handle)
{
    GKO_ASSERT_NO_HIPBLAS_ERRORS(
        hipblasDestroy(reinterpret_cast<hipblasHandle_t>(handle)));
}


}  // namespace hipblas
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_HIPBLAS_HANDLE_HPP_
