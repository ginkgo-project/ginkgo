// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_POINTER_MODE_GUARD_HIP_HPP_
#define GKO_HIP_BASE_POINTER_MODE_GUARD_HIP_HPP_


#include <exception>


#include <hip/hip_runtime.h>
#if HIP_VERSION >= 50200000
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>
#else
#include <hipblas.h>
#include <hipsparse.h>
#endif


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/std_extensions.hpp>


namespace gko {
namespace kernels {
namespace hip {
namespace hipblas {


/**
 * This class defines a pointer mode guard for the hip functions and the hip
 * module. The guard is used to make sure that the correct pointer mode has been
 * set when using scalars for the hipblas functions. The class records the
 * current handle and sets the pointer mode to host for the current scope. After
 * the scope has been exited, the destructor sets the pointer mode back to
 * device.
 */
class pointer_mode_guard {
public:
    pointer_mode_guard(hipblasContext* handle)
    {
        l_handle = handle;
        GKO_ASSERT_NO_HIPBLAS_ERRORS(
            hipblasSetPointerMode(reinterpret_cast<hipblasHandle_t>(handle),
                                  HIPBLAS_POINTER_MODE_HOST));
    }

    pointer_mode_guard(pointer_mode_guard& other) = delete;

    pointer_mode_guard& operator=(const pointer_mode_guard& other) = delete;

    pointer_mode_guard(pointer_mode_guard&& other) = delete;

    pointer_mode_guard const& operator=(pointer_mode_guard&& other) = delete;

    ~pointer_mode_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (xstd::uncaught_exception()) {
            hipblasSetPointerMode(reinterpret_cast<hipblasHandle_t>(l_handle),
                                  HIPBLAS_POINTER_MODE_DEVICE);
        } else {
            GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasSetPointerMode(
                reinterpret_cast<hipblasHandle_t>(l_handle),
                HIPBLAS_POINTER_MODE_DEVICE));
        }
    }

private:
    hipblasContext* l_handle;
};


}  // namespace hipblas


namespace hipsparse {


/**
 * This class defines a pointer mode guard for the hip functions and the hip
 * module. The guard is used to make sure that the correct pointer mode has been
 * set when using scalars for the hipsparse functions. The class records the
 * current handle and sets the pointer mode to host for the current scope. After
 * the scope has been exited, the destructor sets the pointer mode back to
 * device.
 */
class pointer_mode_guard {
public:
    pointer_mode_guard(hipsparseContext* handle)
    {
        l_handle = handle;
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(
            hipsparseSetPointerMode(reinterpret_cast<hipsparseHandle_t>(handle),
                                    HIPSPARSE_POINTER_MODE_HOST));
    }

    pointer_mode_guard(pointer_mode_guard& other) = delete;

    pointer_mode_guard& operator=(const pointer_mode_guard& other) = delete;

    pointer_mode_guard(pointer_mode_guard&& other) = delete;

    pointer_mode_guard const& operator=(pointer_mode_guard&& other) = delete;

    ~pointer_mode_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (xstd::uncaught_exception()) {
            hipsparseSetPointerMode(
                reinterpret_cast<hipsparseHandle_t>(l_handle),
                HIPSPARSE_POINTER_MODE_DEVICE);
        } else {
            GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseSetPointerMode(
                reinterpret_cast<hipsparseHandle_t>(l_handle),
                HIPSPARSE_POINTER_MODE_DEVICE));
        }
    }

private:
    hipsparseContext* l_handle;
};


}  // namespace hipsparse
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_POINTER_MODE_GUARD_HIP_HPP_
