// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_CUSPARSE_HANDLE_HPP_
#define GKO_CUDA_BASE_CUSPARSE_HANDLE_HPP_


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CUSPARSE namespace.
 *
 * @ingroup cusparse
 */
namespace cusparse {


inline cusparseHandle_t init(cudaStream_t stream)
{
    cusparseHandle_t handle{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreate(&handle));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetStream(handle, stream));
    return handle;
}


inline void destroy(cusparseHandle_t handle)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroy(handle));
}


}  // namespace cusparse
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CUSPARSE_HANDLE_HPP_
