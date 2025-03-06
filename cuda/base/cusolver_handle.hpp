// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_CUSOLVER_HANDLE_HPP_
#define GKO_CUDA_BASE_CUSOLVER_HANDLE_HPP_

#include <cuda.h>
#include <cusolverDn.h>

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CUSOLVER namespace.
 *
 * @ingroup cusolver
 */
namespace cusolver {


inline cusolverDnHandle_t init(cudaStream_t stream)
{
    cusolverDnHandle_t handle;
    GKO_ASSERT_NO_CUSOLVER_ERRORS(cusolverDnCreate(&handle));
    GKO_ASSERT_NO_CUSOLVER_ERRORS(cusolverDnSetStream(handle, stream));
    return handle;
}


inline void destroy(cusolverDnHandle_t handle)
{
    GKO_ASSERT_NO_CUSOLVER_ERRORS(cusolverDnDestroy(handle));
}


}  // namespace cusolver
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CUSOLVER_HANDLE_HPP_
