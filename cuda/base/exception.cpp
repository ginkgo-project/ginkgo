/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/exception.hpp>


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>


namespace gko {


std::string CudaError::get_error(int64 error_code)
{
    std::string name = cudaGetErrorName(static_cast<cudaError>(error_code));
    std::string message =
        cudaGetErrorString(static_cast<cudaError>(error_code));
    return name + ": " + message;
}


std::string CublasError::get_error(int64 error_code)
{
#define GKO_REGISTER_CUBLAS_ERROR(error_name)           \
    if (error_code == static_cast<int64>(error_name)) { \
        return #error_name;                             \
    }
    GKO_REGISTER_CUBLAS_ERROR(CUBLAS_STATUS_SUCCESS);
    GKO_REGISTER_CUBLAS_ERROR(CUBLAS_STATUS_NOT_INITIALIZED);
    GKO_REGISTER_CUBLAS_ERROR(CUBLAS_STATUS_ALLOC_FAILED);
    GKO_REGISTER_CUBLAS_ERROR(CUBLAS_STATUS_INVALID_VALUE);
    GKO_REGISTER_CUBLAS_ERROR(CUBLAS_STATUS_ARCH_MISMATCH);
    GKO_REGISTER_CUBLAS_ERROR(CUBLAS_STATUS_MAPPING_ERROR);
    GKO_REGISTER_CUBLAS_ERROR(CUBLAS_STATUS_EXECUTION_FAILED);
    GKO_REGISTER_CUBLAS_ERROR(CUBLAS_STATUS_INTERNAL_ERROR);
    GKO_REGISTER_CUBLAS_ERROR(CUBLAS_STATUS_NOT_SUPPORTED);
    GKO_REGISTER_CUBLAS_ERROR(CUBLAS_STATUS_LICENSE_ERROR);
    return "Unknown error";
}


std::string CusparseError::get_error(int64 error_code)
{
#define GKO_REGISTER_CUSPARSE_ERROR(error_name) \
    if (error_code == int64(error_name)) {      \
        return #error_name;                     \
    }
    GKO_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_SUCCESS);
    GKO_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_NOT_INITIALIZED);
    GKO_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_ALLOC_FAILED);
    GKO_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_INVALID_VALUE);
    GKO_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_ARCH_MISMATCH);
    GKO_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_MAPPING_ERROR);
    GKO_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_EXECUTION_FAILED);
    GKO_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_INTERNAL_ERROR);
    GKO_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    return "Unknown error";
}


}  // namespace gko
