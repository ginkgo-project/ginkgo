/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

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

#ifndef CUDA_BASE_CUBLAS_BINDINGS_HPP_
#define CUDA_BASE_CUBLAS_BINDINGS_HPP_


#include <cublas_v2.h>


#include "core/base/exception_helpers.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace cublas {
namespace detail {
namespace {


template <typename... Args>
inline int64 not_implemented(Args &&...)
{
    return static_cast<int64>(CUBLAS_STATUS_NOT_SUPPORTED);
}


}  // namespace
}  // namespace detail


namespace {


template <typename ValueType>
struct is_supported : std::false_type {};

template <>
struct is_supported<float> : std::true_type {};

template <>
struct is_supported<double> : std::true_type {};

template <>
struct is_supported<std::complex<float>> : std::true_type {};

template <>
struct is_supported<std::complex<double>> : std::true_type {};


#define BIND_CUBLAS_GEMM(ValueType, CublasName)                                \
    inline void gemm(cublasHandle_t handle, cublasOperation_t transa,          \
                     cublasOperation_t transb, int m, int n, int k,            \
                     const ValueType *alpha, const ValueType *a, int lda,      \
                     const ValueType *b, int ldb, const ValueType *beta,       \
                     ValueType *c, int ldc)                                    \
    {                                                                          \
        ASSERT_NO_CUBLAS_ERRORS(                                               \
            CublasName(handle, transa, transb, m, n, k, as_culibs_type(alpha), \
                       as_culibs_type(a), lda, as_culibs_type(b), ldb,         \
                       as_culibs_type(beta), as_culibs_type(c), ldc));         \
    }

BIND_CUBLAS_GEMM(float, cublasSgemm);
BIND_CUBLAS_GEMM(double, cublasDgemm);
BIND_CUBLAS_GEMM(std::complex<float>, cublasCgemm);
BIND_CUBLAS_GEMM(std::complex<double>, cublasZgemm);
template <typename ValueType>
BIND_CUBLAS_GEMM(ValueType, detail::not_implemented);

#undef BIND_CUBLAS_GEMM


#define BIND_CUBLAS_GEAM(ValueType, CublasName)                             \
    inline void geam(cublasHandle_t handle, cublasOperation_t transa,       \
                     cublasOperation_t transb, int m, int n,                \
                     const ValueType *alpha, const ValueType *a, int lda,   \
                     const ValueType *beta, const ValueType *b, int ldb,    \
                     ValueType *c, int ldc)                                 \
    {                                                                       \
        ASSERT_NO_CUBLAS_ERRORS(                                            \
            CublasName(handle, transa, transb, m, n, as_culibs_type(alpha), \
                       as_culibs_type(a), lda, as_culibs_type(beta),        \
                       as_culibs_type(b), ldb, as_culibs_type(c), ldc));    \
    }

BIND_CUBLAS_GEAM(float, cublasSgeam);
BIND_CUBLAS_GEAM(double, cublasDgeam);
BIND_CUBLAS_GEAM(std::complex<float>, cublasCgeam);
BIND_CUBLAS_GEAM(std::complex<double>, cublasZgeam);
template <typename ValueType>
BIND_CUBLAS_GEAM(ValueType, detail::not_implemented);

#undef BIND_CUBLAS_GEAM


#define BIND_CUBLAS_SCAL(ValueType, CublasName)                              \
    inline void scal(cublasHandle_t handle, int n, const ValueType *alpha,   \
                     ValueType *x, int incx)                                 \
    {                                                                        \
        ASSERT_NO_CUBLAS_ERRORS(CublasName(handle, n, as_culibs_type(alpha), \
                                           as_culibs_type(x), incx));        \
    }

BIND_CUBLAS_SCAL(float, cublasSscal);
BIND_CUBLAS_SCAL(double, cublasDscal);
BIND_CUBLAS_SCAL(std::complex<float>, cublasCscal);
BIND_CUBLAS_SCAL(std::complex<double>, cublasZscal);
template <typename ValueType>
BIND_CUBLAS_SCAL(ValueType, detail::not_implemented);

#undef BIND_CUBLAS_SCAL


#define BIND_CUBLAS_AXPY(ValueType, CublasName)                              \
    inline void axpy(cublasHandle_t handle, int n, const ValueType *alpha,   \
                     const ValueType *x, int incx, ValueType *y, int incy)   \
    {                                                                        \
        ASSERT_NO_CUBLAS_ERRORS(CublasName(handle, n, as_culibs_type(alpha), \
                                           as_culibs_type(x), incx,          \
                                           as_culibs_type(y), incy));        \
    }

BIND_CUBLAS_AXPY(float, cublasSaxpy);
BIND_CUBLAS_AXPY(double, cublasDaxpy);
BIND_CUBLAS_AXPY(std::complex<float>, cublasCaxpy);
BIND_CUBLAS_AXPY(std::complex<double>, cublasZaxpy);
template <typename ValueType>
BIND_CUBLAS_AXPY(ValueType, detail::not_implemented);

#undef BIND_CUBLAS_AXPY


#define BIND_CUBLAS_DOT(ValueType, CublasName)                                 \
    inline void dot(cublasHandle_t handle, int n, const ValueType *x,          \
                    int incx, const ValueType *y, int incy, ValueType *result) \
    {                                                                          \
        ASSERT_NO_CUBLAS_ERRORS(CublasName(handle, n, as_culibs_type(x), incx, \
                                           as_culibs_type(y), incy,            \
                                           as_culibs_type(result)));           \
    }

BIND_CUBLAS_DOT(float, cublasSdot);
BIND_CUBLAS_DOT(double, cublasDdot);
BIND_CUBLAS_DOT(std::complex<float>, cublasCdotc);
BIND_CUBLAS_DOT(std::complex<double>, cublasZdotc);
template <typename ValueType>
BIND_CUBLAS_DOT(ValueType, detail::not_implemented);

#undef BIND_CUBLAS_DOT


#define BIND_CUBLAS_NORM2(ValueType, ResultType, CublasName)                   \
    inline void norm2(cublasHandle_t handle, int n, const ValueType *x,        \
                      int incx, ResultType *result)                            \
    {                                                                          \
        ASSERT_NO_CUBLAS_ERRORS(CublasName(handle, n, as_culibs_type(x), incx, \
                                           as_culibs_type(result)));           \
    }

BIND_CUBLAS_NORM2(float, float, cublasSnrm2);
BIND_CUBLAS_NORM2(double, double, cublasDnrm2);
BIND_CUBLAS_NORM2(std::complex<float>, float, cublasScnrm2);
BIND_CUBLAS_NORM2(std::complex<double>, double, cublasDznrm2);
template <typename ValueType, typename ResultType>
BIND_CUBLAS_NORM2(ValueType, ResultType, detail::not_implemented);

#undef BIND_CUBLAS_NORM2


inline cublasHandle_t init()
{
    cublasHandle_t handle;
    ASSERT_NO_CUBLAS_ERRORS(cublasCreate(&handle));
    ASSERT_NO_CUBLAS_ERRORS(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    return handle;
}


inline void destroy(cublasHandle_t handle)
{
    ASSERT_NO_CUBLAS_ERRORS(cublasDestroy(handle));
}


}  // namespace
}  // namespace cublas
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // CUDA_BASE_CUBLAS_BINDINGS_HPP_
