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

#ifndef GKO_HIP_BASE_HIPBLAS_BINDINGS_HIP_HPP_
#define GKO_HIP_BASE_HIPBLAS_BINDINGS_HIP_HPP_


#include <hipblas.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace gko {
/**
 * @brief The device specific kernels namespace.
 *
 * @ingroup kernels
 */
namespace kernels {
/**
 * @brief The HIP namespace.
 *
 * @ingroup hip
 */
namespace hip {
/**
 * @brief The HIPBLAS namespace.
 *
 * @ingroup hipblas
 */
namespace hipblas {
/**
 * @brief The detail namespace.
 *
 * @ingroup detail
 */
namespace detail {


template <typename... Args>
inline int64 not_implemented(Args &&...)
{
    return static_cast<int64>(HIPBLAS_STATUS_NOT_SUPPORTED);
}


}  // namespace detail


template <typename ValueType>
struct is_supported : std::false_type {};

template <>
struct is_supported<float> : std::true_type {};

template <>
struct is_supported<double> : std::true_type {};

// hipblas supports part of complex function version is >= 0.19, but the version
// is not set now.
/* not implemented
template <>
struct is_supported<std::complex<float>> : std::true_type {};

template <>
struct is_supported<std::complex<double>> : std::true_type {};
*/


#define GKO_BIND_HIPBLAS_GEMM(ValueType, HipblasName)                        \
    inline void gemm(hipblasHandle_t handle, hipblasOperation_t transa,      \
                     hipblasOperation_t transb, int m, int n, int k,         \
                     const ValueType *alpha, const ValueType *a, int lda,    \
                     const ValueType *b, int ldb, const ValueType *beta,     \
                     ValueType *c, int ldc)                                  \
    {                                                                        \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(HipblasName(                            \
            handle, transa, transb, m, n, k, as_hiplibs_type(alpha),         \
            as_hiplibs_type(a), lda, as_hiplibs_type(b), ldb,                \
            as_hiplibs_type(beta), as_hiplibs_type(c), ldc));                \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_GEMM(float, hipblasSgemm);
GKO_BIND_HIPBLAS_GEMM(double, hipblasDgemm);
/* not implemented
GKO_BIND_HIPBLAS_GEMM(std::complex<float>, hipblasCgemm);
GKO_BIND_HIPBLAS_GEMM(std::complex<double>, hipblasZgemm);
*/
template <typename ValueType>
GKO_BIND_HIPBLAS_GEMM(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_GEMM


#define GKO_BIND_HIPBLAS_GEAM(ValueType, HipblasName)                         \
    inline void geam(hipblasHandle_t handle, hipblasOperation_t transa,       \
                     hipblasOperation_t transb, int m, int n,                 \
                     const ValueType *alpha, const ValueType *a, int lda,     \
                     const ValueType *beta, const ValueType *b, int ldb,      \
                     ValueType *c, int ldc)                                   \
    {                                                                         \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(                                         \
            HipblasName(handle, transa, transb, m, n, as_hiplibs_type(alpha), \
                        as_hiplibs_type(a), lda, as_hiplibs_type(beta),       \
                        as_hiplibs_type(b), ldb, as_hiplibs_type(c), ldc));   \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_GEAM(float, hipblasSgeam);
GKO_BIND_HIPBLAS_GEAM(double, hipblasDgeam);
// Hipblas does not provide geam complex version yet.
template <typename ValueType>
GKO_BIND_HIPBLAS_GEAM(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_GEAM


#define GKO_BIND_HIPBLAS_SCAL(ValueType, HipblasName)                        \
    inline void scal(hipblasHandle_t handle, int n, const ValueType *alpha,  \
                     ValueType *x, int incx)                                 \
    {                                                                        \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(HipblasName(                            \
            handle, n, as_hiplibs_type(alpha), as_hiplibs_type(x), incx));   \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_SCAL(float, hipblasSscal);
GKO_BIND_HIPBLAS_SCAL(double, hipblasDscal);
/* not implemented
GKO_BIND_HIPBLAS_SCAL(std::complex<float>, hipblasCscal);
GKO_BIND_HIPBLAS_SCAL(std::complex<double>, hipblasZscal);
*/
template <typename ValueType>
GKO_BIND_HIPBLAS_SCAL(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_SCAL


#define GKO_BIND_HIPBLAS_AXPY(ValueType, HipblasName)                          \
    inline void axpy(hipblasHandle_t handle, int n, const ValueType *alpha,    \
                     const ValueType *x, int incx, ValueType *y, int incy)     \
    {                                                                          \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(                                          \
            HipblasName(handle, n, as_hiplibs_type(alpha), as_hiplibs_type(x), \
                        incx, as_hiplibs_type(y), incy));                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_AXPY(float, hipblasSaxpy);
GKO_BIND_HIPBLAS_AXPY(double, hipblasDaxpy);
/* not implemented
GKO_BIND_HIPBLAS_AXPY(std::complex<float>, hipblasCaxpy);
GKO_BIND_HIPBLAS_AXPY(std::complex<double>, hipblasZaxpy);
*/
template <typename ValueType>
GKO_BIND_HIPBLAS_AXPY(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_AXPY


#define GKO_BIND_HIPBLAS_DOT(ValueType, HipblasName)                           \
    inline void dot(hipblasHandle_t handle, int n, const ValueType *x,         \
                    int incx, const ValueType *y, int incy, ValueType *result) \
    {                                                                          \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(                                          \
            HipblasName(handle, n, as_hiplibs_type(x), incx,                   \
                        as_hiplibs_type(y), incy, as_hiplibs_type(result)));   \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_DOT(float, hipblasSdot);
GKO_BIND_HIPBLAS_DOT(double, hipblasDdot);
/* not implemented
GKO_BIND_HIPBLAS_DOT(std::complex<float>, hipblasCdotc);
GKO_BIND_HIPBLAS_DOT(std::complex<double>, hipblasZdotc);
*/
template <typename ValueType>
GKO_BIND_HIPBLAS_DOT(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_DOT


#define GKO_BIND_HIPBLAS_COMPLEX_NORM2(ValueType, CublasName)                \
    inline void norm2(hipblasHandle_t handle, int n, const ValueType *x,     \
                      int incx, ValueType *result)                           \
    {                                                                        \
        hipMemset(result, 0, sizeof(ValueType));                             \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(                                        \
            CublasName(handle, n, as_hiplibs_type(x), incx,                  \
                       reinterpret_cast<remove_complex<ValueType> *>(        \
                           as_hiplibs_type(result))));                       \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_HIPBLAS_NORM2(ValueType, HipblasName)                       \
    inline void norm2(hipblasHandle_t handle, int n, const ValueType *x,     \
                      int incx, ValueType *result)                           \
    {                                                                        \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(HipblasName(                            \
            handle, n, as_hiplibs_type(x), incx, as_hiplibs_type(result)));  \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


GKO_BIND_HIPBLAS_NORM2(float, hipblasSnrm2);
GKO_BIND_HIPBLAS_NORM2(double, hipblasDnrm2);
/* not implemented
GKO_BIND_HIPBLAS_COMPLEX_NORM2(std::complex<float>, hipblasScnrm2);
GKO_BIND_HIPBLAS_COMPLEX_NORM2(std::complex<double>, hipblasDznrm2);
*/
template <typename ValueType>
GKO_BIND_HIPBLAS_NORM2(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_NORM2


inline hipblasContext *init()
{
    hipblasHandle_t handle;
    GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasCreate(&handle));
    GKO_ASSERT_NO_HIPBLAS_ERRORS(
        hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    return reinterpret_cast<hipblasContext *>(handle);
}


inline void destroy_hipblas_handle(hipblasContext *handle)
{
    GKO_ASSERT_NO_HIPBLAS_ERRORS(
        hipblasDestroy(reinterpret_cast<hipblasHandle_t>(handle)));
}


}  // namespace hipblas
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_HIPBLAS_BINDINGS_HIP_HPP_
