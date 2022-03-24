/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_HIP_BASE_STREAM_GUARD_HIP_HPP_
#define GKO_HIP_BASE_STREAM_GUARD_HIP_HPP_


#include <exception>


#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <hipsparse.h>


#include <ginkgo/core/base/async_handle.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>


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
class stream_guard {
public:
    stream_guard(hipblasContext* handle, hipStream_t new_stream) : old_stream_{}
    {
        // GKO_ASSERT_NO_HIP_ERRORS(
        //     hipStreamCreateWithFlags(&old_stream_, hipStreamNonBlocking));
        handle_ = handle;
        // GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasGetStream(
        //     reinterpret_cast<hipblasHandle_t>(handle_), &old_stream_));
        GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasSetStream(
            reinterpret_cast<hipblasHandle_t>(handle_), new_stream));
    }

    stream_guard(stream_guard& other) = delete;

    stream_guard& operator=(const stream_guard& other) = delete;

    stream_guard(stream_guard&& other) = delete;

    stream_guard const& operator=(stream_guard&& other) = delete;

    ~stream_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        // if (std::uncaught_exception()) {
        //     GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasSetStream(
        //         reinterpret_cast<hipblasHandle_t>(handle_), old_stream_));
        // } else {
        //     GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasSetStream(
        //         reinterpret_cast<hipblasHandle_t>(handle_), old_stream_));
        // }
    }

private:
    hipblasContext* handle_;
    hipStream_t old_stream_;
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
class stream_guard {
public:
    stream_guard(hipsparseContext* handle, hipStream_t new_stream)
        : old_stream_{}
    {
        GKO_ASSERT_NO_HIP_ERRORS(
            hipStreamCreateWithFlags(&old_stream_, hipStreamNonBlocking));
        handle_ = handle;
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseGetStream(
            reinterpret_cast<hipsparseHandle_t>(handle_), &old_stream_));
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseSetStream(
            reinterpret_cast<hipsparseHandle_t>(handle_), new_stream));
    }

    stream_guard(stream_guard& other) = delete;

    stream_guard& operator=(const stream_guard& other) = delete;

    stream_guard(stream_guard&& other) = delete;

    stream_guard const& operator=(stream_guard&& other) = delete;

    ~stream_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (std::uncaught_exception()) {
            GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseSetStream(
                reinterpret_cast<hipsparseHandle_t>(handle_), old_stream_));
        } else {
            GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseSetStream(
                reinterpret_cast<hipsparseHandle_t>(handle_), old_stream_));
        }
    }

private:
    hipsparseContext* handle_;
    hipStream_t old_stream_;
};


}  // namespace hipsparse
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_STREAM_GUARD_HIP_HPP_
