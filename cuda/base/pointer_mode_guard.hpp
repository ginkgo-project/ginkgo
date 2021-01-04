/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CUDA_BASE_POINTER_MODE_GUARD_HPP_
#define GKO_CUDA_BASE_POINTER_MODE_GUARD_HPP_


#include <exception>


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>


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
    pointer_mode_guard(cublasHandle_t &handle)
    {
        l_handle = &handle;
        GKO_ASSERT_NO_CUBLAS_ERRORS(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    }

    pointer_mode_guard(pointer_mode_guard &other) = delete;

    pointer_mode_guard &operator=(const pointer_mode_guard &other) = delete;

    pointer_mode_guard(pointer_mode_guard &&other) = delete;

    pointer_mode_guard const &operator=(pointer_mode_guard &&other) = delete;

    ~pointer_mode_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (std::uncaught_exception()) {
            cublasSetPointerMode(*l_handle, CUBLAS_POINTER_MODE_DEVICE);
        } else {
            GKO_ASSERT_NO_CUBLAS_ERRORS(
                cublasSetPointerMode(*l_handle, CUBLAS_POINTER_MODE_DEVICE));
        }
    }

private:
    cublasHandle_t *l_handle;
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
    pointer_mode_guard(cusparseHandle_t &handle)
    {
        l_handle = &handle;
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
    }

    pointer_mode_guard(pointer_mode_guard &other) = delete;

    pointer_mode_guard &operator=(const pointer_mode_guard &other) = delete;

    pointer_mode_guard(pointer_mode_guard &&other) = delete;

    pointer_mode_guard const &operator=(pointer_mode_guard &&other) = delete;

    ~pointer_mode_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (std::uncaught_exception()) {
            cusparseSetPointerMode(*l_handle, CUSPARSE_POINTER_MODE_DEVICE);
        } else {
            GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetPointerMode(
                *l_handle, CUSPARSE_POINTER_MODE_DEVICE));
        }
    }

private:
    cusparseHandle_t *l_handle;
};


}  // namespace cusparse
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_POINTER_MODE_GUARD_HPP_
