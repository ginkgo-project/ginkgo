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

#ifndef GKO_CUDA_BASE_CUBLAS_BINDINGS_HPP_
#define GKO_CUDA_BASE_CUBLAS_BINDINGS_HPP_


#include <cublas_v2.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/zero_array.hpp"


namespace gko {
/**
 * @brief The device specific kernels.
 *
 * \ingroup kernels
 */
namespace kernels {
/**
 * @brief The CUDA namespace.
 *
 * \ingroup cuda
 */
namespace cuda {
/**
 * @brief The CUBLAS namespace.
 *
 * \ingroup cublas
 */
namespace cublas {
/**
 * @brief The detail namespace.
 *
 * \ingroup detail
 */
namespace detail {


template <typename... Args>
inline int64 not_implemented(Args &&...)
{
    return static_cast<int64>(CUBLAS_STATUS_NOT_SUPPORTED);
}


}  // namespace detail


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


#define GKO_BIND_CUBLAS_GEMM(ValueType, CublasName)                            \
    inline void gemm(cublasHandle_t handle, cublasOperation_t transa,          \
                     cublasOperation_t transb, int m, int n, int k,            \
                     const ValueType *alpha, const ValueType *a, int lda,      \
                     const ValueType *b, int ldb, const ValueType *beta,       \
                     ValueType *c, int ldc)                                    \
    {                                                                          \
        GKO_ASSERT_NO_CUBLAS_ERRORS(                                           \
            CublasName(handle, transa, transb, m, n, k, as_culibs_type(alpha), \
                       as_culibs_type(a), lda, as_culibs_type(b), ldb,         \
                       as_culibs_type(beta), as_culibs_type(c), ldc));         \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUBLAS_GEMM(float, cublasSgemm);
GKO_BIND_CUBLAS_GEMM(double, cublasDgemm);
GKO_BIND_CUBLAS_GEMM(std::complex<float>, cublasCgemm);
GKO_BIND_CUBLAS_GEMM(std::complex<double>, cublasZgemm);
template <typename ValueType>
GKO_BIND_CUBLAS_GEMM(ValueType, detail::not_implemented);

#undef GKO_BIND_CUBLAS_GEMM


#define GKO_BIND_CUBLAS_GEAM(ValueType, CublasName)                          \
    inline void geam(cublasHandle_t handle, cublasOperation_t transa,        \
                     cublasOperation_t transb, int m, int n,                 \
                     const ValueType *alpha, const ValueType *a, int lda,    \
                     const ValueType *beta, const ValueType *b, int ldb,     \
                     ValueType *c, int ldc)                                  \
    {                                                                        \
        GKO_ASSERT_NO_CUBLAS_ERRORS(                                         \
            CublasName(handle, transa, transb, m, n, as_culibs_type(alpha),  \
                       as_culibs_type(a), lda, as_culibs_type(beta),         \
                       as_culibs_type(b), ldb, as_culibs_type(c), ldc));     \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUBLAS_GEAM(float, cublasSgeam);
GKO_BIND_CUBLAS_GEAM(double, cublasDgeam);
GKO_BIND_CUBLAS_GEAM(std::complex<float>, cublasCgeam);
GKO_BIND_CUBLAS_GEAM(std::complex<double>, cublasZgeam);
template <typename ValueType>
GKO_BIND_CUBLAS_GEAM(ValueType, detail::not_implemented);

#undef GKO_BIND_CUBLAS_GEAM


#define GKO_BIND_CUBLAS_SCAL(ValueType, CublasName)                          \
    inline void scal(cublasHandle_t handle, int n, const ValueType *alpha,   \
                     ValueType *x, int incx)                                 \
    {                                                                        \
        GKO_ASSERT_NO_CUBLAS_ERRORS(CublasName(                              \
            handle, n, as_culibs_type(alpha), as_culibs_type(x), incx));     \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUBLAS_SCAL(float, cublasSscal);
GKO_BIND_CUBLAS_SCAL(double, cublasDscal);
GKO_BIND_CUBLAS_SCAL(std::complex<float>, cublasCscal);
GKO_BIND_CUBLAS_SCAL(std::complex<double>, cublasZscal);
template <typename ValueType>
GKO_BIND_CUBLAS_SCAL(ValueType, detail::not_implemented);

#undef GKO_BIND_CUBLAS_SCAL


#define GKO_BIND_CUBLAS_AXPY(ValueType, CublasName)                          \
    inline void axpy(cublasHandle_t handle, int n, const ValueType *alpha,   \
                     const ValueType *x, int incx, ValueType *y, int incy)   \
    {                                                                        \
        GKO_ASSERT_NO_CUBLAS_ERRORS(                                         \
            CublasName(handle, n, as_culibs_type(alpha), as_culibs_type(x),  \
                       incx, as_culibs_type(y), incy));                      \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUBLAS_AXPY(float, cublasSaxpy);
GKO_BIND_CUBLAS_AXPY(double, cublasDaxpy);
GKO_BIND_CUBLAS_AXPY(std::complex<float>, cublasCaxpy);
GKO_BIND_CUBLAS_AXPY(std::complex<double>, cublasZaxpy);
template <typename ValueType>
GKO_BIND_CUBLAS_AXPY(ValueType, detail::not_implemented);

#undef GKO_BIND_CUBLAS_AXPY


#define GKO_BIND_CUBLAS_DOT(ValueType, CublasName)                             \
    inline void dot(cublasHandle_t handle, int n, const ValueType *x,          \
                    int incx, const ValueType *y, int incy, ValueType *result) \
    {                                                                          \
        GKO_ASSERT_NO_CUBLAS_ERRORS(CublasName(handle, n, as_culibs_type(x),   \
                                               incx, as_culibs_type(y), incy,  \
                                               as_culibs_type(result)));       \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUBLAS_DOT(float, cublasSdot);
GKO_BIND_CUBLAS_DOT(double, cublasDdot);
GKO_BIND_CUBLAS_DOT(std::complex<float>, cublasCdotc);
GKO_BIND_CUBLAS_DOT(std::complex<double>, cublasZdotc);
template <typename ValueType>
GKO_BIND_CUBLAS_DOT(ValueType, detail::not_implemented);

#undef GKO_BIND_CUBLAS_DOT


#define GKO_BIND_CUBLAS_COMPLEX_NORM2(ValueType, CublasName)                 \
    inline void norm2(cublasHandle_t handle, int n, const ValueType *x,      \
                      int incx, ValueType *result)                           \
    {                                                                        \
        zero_array(n, result);                                               \
        GKO_ASSERT_NO_CUBLAS_ERRORS(                                         \
            CublasName(handle, n, as_culibs_type(x), incx,                   \
                       reinterpret_cast<remove_complex<ValueType> *>(        \
                           as_culibs_type(result))));                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define GKO_BIND_CUBLAS_NORM2(ValueType, CublasName)                           \
    inline void norm2(cublasHandle_t handle, int n, const ValueType *x,        \
                      int incx, ValueType *result)                             \
    {                                                                          \
        GKO_ASSERT_NO_CUBLAS_ERRORS(CublasName(handle, n, as_culibs_type(x),   \
                                               incx, as_culibs_type(result))); \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


GKO_BIND_CUBLAS_NORM2(float, cublasSnrm2);
GKO_BIND_CUBLAS_NORM2(double, cublasDnrm2);
GKO_BIND_CUBLAS_COMPLEX_NORM2(std::complex<float>, cublasScnrm2);
GKO_BIND_CUBLAS_COMPLEX_NORM2(std::complex<double>, cublasDznrm2);
template <typename ValueType>
GKO_BIND_CUBLAS_NORM2(ValueType, detail::not_implemented);

#undef GKO_BIND_CUBLAS_NORM2


inline cublasHandle_t init()
{
    cublasHandle_t handle;
    GKO_ASSERT_NO_CUBLAS_ERRORS(cublasCreate(&handle));
    GKO_ASSERT_NO_CUBLAS_ERRORS(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    return handle;
}


inline void destroy(cublasHandle_t handle)
{
    GKO_ASSERT_NO_CUBLAS_ERRORS(cublasDestroy(handle));
}


}  // namespace cublas
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CUBLAS_BINDINGS_HPP_
