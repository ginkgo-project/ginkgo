// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_HIPBLAS_BINDINGS_HIP_HPP_
#define GKO_HIP_BASE_HIPBLAS_BINDINGS_HIP_HPP_


#include <hip/hip_runtime.h>
#if HIP_VERSION >= 50200000
#include <hipblas/hipblas.h>
#else
#include <hipblas.h>
#endif


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
inline int64 not_implemented(Args&&...)
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

template <>
struct is_supported<std::complex<float>> : std::true_type {};

template <>
struct is_supported<std::complex<double>> : std::true_type {};


#define GKO_BIND_HIPBLAS_GEMM(ValueType, HipblasName)                        \
    inline void gemm(hipblasHandle_t handle, hipblasOperation_t transa,      \
                     hipblasOperation_t transb, int m, int n, int k,         \
                     const ValueType* alpha, const ValueType* a, int lda,    \
                     const ValueType* b, int ldb, const ValueType* beta,     \
                     ValueType* c, int ldc)                                  \
    {                                                                        \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(HipblasName(                            \
            handle, transa, transb, m, n, k, as_hipblas_type(alpha),         \
            as_hipblas_type(a), lda, as_hipblas_type(b), ldb,                \
            as_hipblas_type(beta), as_hipblas_type(c), ldc));                \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_GEMM(float, hipblasSgemm);
GKO_BIND_HIPBLAS_GEMM(double, hipblasDgemm);
GKO_BIND_HIPBLAS_GEMM(std::complex<float>, hipblasCgemm);
GKO_BIND_HIPBLAS_GEMM(std::complex<double>, hipblasZgemm);

template <typename ValueType>
GKO_BIND_HIPBLAS_GEMM(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_GEMM


#define GKO_BIND_HIPBLAS_GEAM(ValueType, HipblasName)                         \
    inline void geam(hipblasHandle_t handle, hipblasOperation_t transa,       \
                     hipblasOperation_t transb, int m, int n,                 \
                     const ValueType* alpha, const ValueType* a, int lda,     \
                     const ValueType* beta, const ValueType* b, int ldb,      \
                     ValueType* c, int ldc)                                   \
    {                                                                         \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(                                         \
            HipblasName(handle, transa, transb, m, n, as_hipblas_type(alpha), \
                        as_hipblas_type(a), lda, as_hipblas_type(beta),       \
                        as_hipblas_type(b), ldb, as_hipblas_type(c), ldc));   \
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
    inline void scal(hipblasHandle_t handle, int n, const ValueType* alpha,  \
                     ValueType* x, int incx)                                 \
    {                                                                        \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(HipblasName(                            \
            handle, n, as_hipblas_type(alpha), as_hipblas_type(x), incx));   \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_SCAL(float, hipblasSscal);
GKO_BIND_HIPBLAS_SCAL(double, hipblasDscal);
GKO_BIND_HIPBLAS_SCAL(std::complex<float>, hipblasCscal);
GKO_BIND_HIPBLAS_SCAL(std::complex<double>, hipblasZscal);

template <typename ValueType>
GKO_BIND_HIPBLAS_SCAL(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_SCAL


#define GKO_BIND_HIPBLAS_AXPY(ValueType, HipblasName)                          \
    inline void axpy(hipblasHandle_t handle, int n, const ValueType* alpha,    \
                     const ValueType* x, int incx, ValueType* y, int incy)     \
    {                                                                          \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(                                          \
            HipblasName(handle, n, as_hipblas_type(alpha), as_hipblas_type(x), \
                        incx, as_hipblas_type(y), incy));                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_AXPY(float, hipblasSaxpy);
GKO_BIND_HIPBLAS_AXPY(double, hipblasDaxpy);
GKO_BIND_HIPBLAS_AXPY(std::complex<float>, hipblasCaxpy);
GKO_BIND_HIPBLAS_AXPY(std::complex<double>, hipblasZaxpy);

template <typename ValueType>
GKO_BIND_HIPBLAS_AXPY(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_AXPY


#define GKO_BIND_HIPBLAS_DOT(ValueType, HipblasName)                           \
    inline void dot(hipblasHandle_t handle, int n, const ValueType* x,         \
                    int incx, const ValueType* y, int incy, ValueType* result) \
    {                                                                          \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(                                          \
            HipblasName(handle, n, as_hipblas_type(x), incx,                   \
                        as_hipblas_type(y), incy, as_hipblas_type(result)));   \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_DOT(float, hipblasSdot);
GKO_BIND_HIPBLAS_DOT(double, hipblasDdot);
GKO_BIND_HIPBLAS_DOT(std::complex<float>, hipblasCdotu);
GKO_BIND_HIPBLAS_DOT(std::complex<double>, hipblasZdotu);

template <typename ValueType>
GKO_BIND_HIPBLAS_DOT(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_DOT


#define GKO_BIND_HIPBLAS_CONJ_DOT(ValueType, HipblasName)                    \
    inline void conj_dot(hipblasHandle_t handle, int n, const ValueType* x,  \
                         int incx, const ValueType* y, int incy,             \
                         ValueType* result)                                  \
    {                                                                        \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(                                        \
            HipblasName(handle, n, as_hipblas_type(x), incx,                 \
                        as_hipblas_type(y), incy, as_hipblas_type(result))); \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_CONJ_DOT(float, hipblasSdot);
GKO_BIND_HIPBLAS_CONJ_DOT(double, hipblasDdot);
GKO_BIND_HIPBLAS_CONJ_DOT(std::complex<float>, hipblasCdotc);
GKO_BIND_HIPBLAS_CONJ_DOT(std::complex<double>, hipblasZdotc);

template <typename ValueType>
GKO_BIND_HIPBLAS_CONJ_DOT(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_CONJ_DOT


#define GKO_BIND_HIPBLAS_NORM2(ValueType, HipblasName)                       \
    inline void norm2(hipblasHandle_t handle, int n, const ValueType* x,     \
                      int incx, remove_complex<ValueType>* result)           \
    {                                                                        \
        GKO_ASSERT_NO_HIPBLAS_ERRORS(HipblasName(                            \
            handle, n, as_hipblas_type(x), incx, as_hipblas_type(result)));  \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPBLAS_NORM2(float, hipblasSnrm2);
GKO_BIND_HIPBLAS_NORM2(double, hipblasDnrm2);
GKO_BIND_HIPBLAS_NORM2(std::complex<float>, hipblasScnrm2);
GKO_BIND_HIPBLAS_NORM2(std::complex<double>, hipblasDznrm2);

template <typename ValueType>
GKO_BIND_HIPBLAS_NORM2(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPBLAS_NORM2


inline hipblasContext* init(hipStream_t stream)
{
    hipblasHandle_t handle;
    GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasCreate(&handle));
    GKO_ASSERT_NO_HIPBLAS_ERRORS(
        hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    GKO_ASSERT_NO_HIPBLAS_ERRORS(hipblasSetStream(handle, stream));
    return reinterpret_cast<hipblasContext*>(handle);
}


inline void destroy_hipblas_handle(hipblasContext* handle)
{
    GKO_ASSERT_NO_HIPBLAS_ERRORS(
        hipblasDestroy(reinterpret_cast<hipblasHandle_t>(handle)));
}


}  // namespace hipblas
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_HIPBLAS_BINDINGS_HIP_HPP_
