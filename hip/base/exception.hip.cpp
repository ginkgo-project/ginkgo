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

#include <ginkgo/core/base/exception.hpp>

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <hipsparse.h>

namespace gko {


std::string HipError::get_error(int64 error_code)
{
    std::string name = hipGetErrorName(static_cast<hipError_t>(error_code));
    std::string message =
        hipGetErrorString(static_cast<hipError_t>(error_code));
    return name + ": " + message;
}


std::string HipblasError::get_error(int64 error_code)
{
#define GKO_REGISTER_HIPBLAS_ERROR(error_name)          \
    if (error_code == static_cast<int64>(error_name)) { \
        return #error_name;                             \
    }
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_SUCCESS);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_NOT_INITIALIZED);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_ALLOC_FAILED);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_INVALID_VALUE);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_ARCH_MISMATCH);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_MAPPING_ERROR);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_EXECUTION_FAILED);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_INTERNAL_ERROR);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_NOT_SUPPORTED);
    return "Unknown error";
}


std::string HipsparseError::get_error(int64 error_code)
{
#define GKO_REGISTER_HIPSPARSE_ERROR(error_name) \
    if (error_code == int64(error_name)) {       \
        return #error_name;                      \
    }
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_SUCCESS);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_NOT_INITIALIZED);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_ALLOC_FAILED);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_INVALID_VALUE);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_ARCH_MISMATCH);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_MAPPING_ERROR);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_EXECUTION_FAILED);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_INTERNAL_ERROR);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    return "Unknown error";
}


}  // namespace gko
