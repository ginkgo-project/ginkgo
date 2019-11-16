/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_HIP_BASE_POINTER_MODE_GUARD_HIP_HPP_
#define GKO_HIP_BASE_POINTER_MODE_GUARD_HIP_HPP_


#include <exception>


#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <hipsparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>


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
    pointer_mode_guard(hipblasContext *handle)
    {
        l_handle = handle;
        GKO_ASSERT_NO_HIPBLAS_ERRORS(
            hipblasSetPointerMode(reinterpret_cast<hipblasHandle_t>(handle),
                                  HIPBLAS_POINTER_MODE_HOST));
    }

    pointer_mode_guard(pointer_mode_guard &other) = delete;

    pointer_mode_guard &operator=(const pointer_mode_guard &other) = delete;

    pointer_mode_guard(pointer_mode_guard &&other) = delete;

    pointer_mode_guard const &operator=(pointer_mode_guard &&other) = delete;

    ~pointer_mode_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (std::uncaught_exception()) {
            hipblasSetPointerMode(reinterpret_cast<hipblasHandle_t>(l_handle),
                                  HIPBLAS_POINTER_MODE_DEVICE);
        } else {
            GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasSetPointerMode(
                reinterpret_cast<hipblasHandle_t>(l_handle),
                HIPBLAS_POINTER_MODE_DEVICE));
        }
    }

private:
    hipblasContext *l_handle;
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
    pointer_mode_guard(hipsparseContext *handle)
    {
        l_handle = handle;
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(
            hipsparseSetPointerMode(reinterpret_cast<hipsparseHandle_t>(handle),
                                    HIPSPARSE_POINTER_MODE_HOST));
    }

    pointer_mode_guard(pointer_mode_guard &other) = delete;

    pointer_mode_guard &operator=(const pointer_mode_guard &other) = delete;

    pointer_mode_guard(pointer_mode_guard &&other) = delete;

    pointer_mode_guard const &operator=(pointer_mode_guard &&other) = delete;

    ~pointer_mode_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (std::uncaught_exception()) {
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
    hipsparseContext *l_handle;
};


}  // namespace hipsparse
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_POINTER_MODE_GUARD_HIP_HPP_
