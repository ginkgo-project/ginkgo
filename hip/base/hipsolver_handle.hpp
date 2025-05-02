// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_HIPSOLVER_HANDLE_HPP_
#define GKO_HIP_BASE_HIPSOLVER_HANDLE_HPP_

#if HIP_VERSION >= 50200000
#include <hipsolver/hipsolver.h>
#else
#include <hipsolver.h>
#endif

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The HIPSOLVER namespace.
 *
 * @ingroup hipsolver
 */
namespace hipsolver {


inline hipsolverDnContext* init(hipStream_t stream)
{
    hipsolverDnHandle_t handle;
    GKO_ASSERT_NO_HIPSOLVER_ERRORS(hipsolverDnCreate(&handle));
    GKO_ASSERT_NO_HIPSOLVER_ERRORS(hipsolverDnSetStream(handle, stream));
    return reinterpret_cast<hipsolverDnContext*>(handle);
}


inline void destroy_hipsolver_handle(hipsolverDnHandle_t handle)
{
    GKO_ASSERT_NO_HIPSOLVER_ERRORS(
        hipsolverDnDestroy(reinterpret_cast<hipsolverDnHandle_t>(handle)));
}


}  // namespace hipsolver
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_HIPSOLVER_HANDLE_HPP_
