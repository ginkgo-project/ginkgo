// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_HIPSPARSE_HANDLE_HPP_
#define GKO_HIP_BASE_HIPSPARSE_HANDLE_HPP_


#if HIP_VERSION >= 50200000
#include <hipsparse/hipsparse.h>
#else
#include <hipsparse.h>
#endif

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The HIPSPARSE namespace.
 *
 * @ingroup hipsparse
 */
namespace hipsparse {


inline hipsparseContext* init(hipStream_t stream)
{
    hipsparseHandle_t handle{};
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseCreate(&handle));
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(
        hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseSetStream(handle, stream));
    return reinterpret_cast<hipsparseContext*>(handle);
}


inline void destroy_hipsparse_handle(hipsparseContext* handle)
{
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(
        hipsparseDestroy(reinterpret_cast<hipsparseHandle_t>(handle)));
}


}  // namespace hipsparse
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_HIPSPARSE_HANDLE_HPP_
