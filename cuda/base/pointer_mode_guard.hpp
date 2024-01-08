// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_POINTER_MODE_GUARD_HPP_
#define GKO_CUDA_BASE_POINTER_MODE_GUARD_HPP_


#include <exception>


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/std_extensions.hpp>


namespace gko {
namespace kernels {
namespace cuda {
namespace cublas {


/**
 * This class defines a pointer mode guard for the cuda functions and the cuda
 * module. The guard is used to make sure that the correct pointer mode has been
 * set when using scalars for the cublas functions. The class records the
 * current handle and sets the pointer mode to host for the current scope. After
 * the scope has been exited, the destructor sets the pointer mode back to
 * device.
 */
class pointer_mode_guard {
public:
    pointer_mode_guard(cublasHandle_t& handle)
    {
        l_handle = &handle;
        GKO_ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    }

    pointer_mode_guard(pointer_mode_guard& other) = delete;

    pointer_mode_guard& operator=(const pointer_mode_guard& other) = delete;

    pointer_mode_guard(pointer_mode_guard&& other) = delete;

    pointer_mode_guard const& operator=(pointer_mode_guard&& other) = delete;

    ~pointer_mode_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (xstd::uncaught_exception()) {
            cublasSetPointerMode(*l_handle, CUBLAS_POINTER_MODE_DEVICE);
        } else {
            GKO_ASSERT_NO_CUBLAS_ERRORS(
                cublasSetPointerMode(*l_handle, CUBLAS_POINTER_MODE_DEVICE));
        }
    }

private:
    cublasHandle_t* l_handle;
};


}  // namespace cublas


namespace cusparse {


/**
 * This class defines a pointer mode guard for the cuda functions and the cuda
 * module. The guard is used to make sure that the correct pointer mode has been
 * set when using scalars for the cusparse functions. The class records the
 * current handle and sets the pointer mode to host for the current scope. After
 * the scope has been exited, the destructor sets the pointer mode back to
 * device.
 */
class pointer_mode_guard {
public:
    pointer_mode_guard(cusparseHandle_t handle)
    {
        l_handle = handle;
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
    }

    pointer_mode_guard(pointer_mode_guard& other) = delete;

    pointer_mode_guard& operator=(const pointer_mode_guard& other) = delete;

    pointer_mode_guard(pointer_mode_guard&& other) = delete;

    pointer_mode_guard const& operator=(pointer_mode_guard&& other) = delete;

    ~pointer_mode_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (xstd::uncaught_exception()) {
            cusparseSetPointerMode(l_handle, CUSPARSE_POINTER_MODE_DEVICE);
        } else {
            GKO_ASSERT_NO_CUSPARSE_ERRORS(
                cusparseSetPointerMode(l_handle, CUSPARSE_POINTER_MODE_DEVICE));
        }
    }

private:
    cusparseHandle_t l_handle;
};


}  // namespace cusparse
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_POINTER_MODE_GUARD_HPP_
