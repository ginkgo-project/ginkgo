/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_DPCPP_BASE_ONEMKL_BINDINGS_HPP_
#define GKO_DPCPP_BASE_ONEMKL_BINDINGS_HPP_


#include <functional>
#include <type_traits>

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
/**
 * @brief The device specific kernels namespace.
 *
 * @ingroup kernels
 */
namespace kernels {
/**
 * @brief The DPCPP namespace.
 *
 * @ingroup dpcpp
 */
namespace dpcpp {
/**
 * @brief The ONEMKL namespace.
 *
 * @ingroup onemkl
 */
namespace onemkl {
/**
 * @brief The detail namespace.
 *
 * @ingroup detail
 */
namespace detail {


template <typename... Args>
inline void not_implemented(Args&&...) GKO_NOT_IMPLEMENTED;


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


#define GKO_BIND_DOT(ValueType, Name, Func)                                    \
    inline void Name(sycl::queue& exec_queue, std::int64_t n,                  \
                     const ValueType* x, std::int64_t incx,                    \
                     const ValueType* y, std::int64_t incy, ValueType* result) \
    {                                                                          \
        Func(exec_queue, n, x, incx, y, incy, result);                         \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

// Bind the dot for x^T * y
GKO_BIND_DOT(float, dot, oneapi::mkl::blas::row_major::dot);
GKO_BIND_DOT(double, dot, oneapi::mkl::blas::row_major::dot);
GKO_BIND_DOT(std::complex<float>, dot, oneapi::mkl::blas::row_major::dotu);
GKO_BIND_DOT(std::complex<double>, dot, oneapi::mkl::blas::row_major::dotu);
template <typename ValueType>
GKO_BIND_DOT(ValueType, dot, detail::not_implemented);

// Bind the conj_dot for x' * y
GKO_BIND_DOT(float, conj_dot, oneapi::mkl::blas::row_major::dot);
GKO_BIND_DOT(double, conj_dot, oneapi::mkl::blas::row_major::dot);
GKO_BIND_DOT(std::complex<float>, conj_dot, oneapi::mkl::blas::row_major::dotc);
GKO_BIND_DOT(std::complex<double>, conj_dot,
             oneapi::mkl::blas::row_major::dotc);
template <typename ValueType>
GKO_BIND_DOT(ValueType, conj_dot, detail::not_implemented);

#undef GKO_BIND_DOT


using oneapi::mkl::lapack::getrf_batch_scratchpad_size;
using oneapi::mkl::lapack::getrs_batch_scratchpad_size;
using oneapi::mkl::transpose::nontrans;

// TODO: is there any reasons that we need these macro bindings insteads of
// using "using" keyword to put those oneMKL rountine into our dpcpp::onemkl::
// namespace and use it directly? * Check the two lines below and
// batch_direct_kernels.dp.cpp
using oneapi::mkl::lapack::getrf_batch;
using oneapi::mkl::lapack::getrs_batch;


#define GKO_BIND_ONEMKL_BATCH_GETRF(T, FuncName)                               \
    inline void batch_getrf(sycl::queue& queue, std::int64_t m,                \
                            std::int64_t n, T* a, std::int64_t lda,            \
                            std::int64_t stride_a, std::int64_t* ipiv,         \
                            std::int64_t stride_ipiv, std::int64_t batch_size, \
                            T* scratchpad, std::int64_t scratchpad_size)       \
    {                                                                          \
        FuncName(queue, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, \
                 scratchpad, scratchpad_size);                                 \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_ONEMKL_BATCH_GETRF(float, oneapi::mkl::lapack::getrf_batch);
GKO_BIND_ONEMKL_BATCH_GETRF(double, oneapi::mkl::lapack::getrf_batch);
GKO_BIND_ONEMKL_BATCH_GETRF(std::complex<float>,
                            oneapi::mkl::lapack::getrf_batch);
GKO_BIND_ONEMKL_BATCH_GETRF(std::complex<double>,
                            oneapi::mkl::lapack::getrf_batch);
template <typename ValueType>
GKO_BIND_ONEMKL_BATCH_GETRF(ValueType, detail::not_implemented);

#undef GKO_BIND_ONEMKL_BATCH_GETRF


#define GKO_BIND_ONEMKL_BATCH_GETRS(T, FuncName)                              \
    inline void batch_getrs(sycl::queue& queue, std::int64_t m,               \
                            std::int64_t nrhs, T* a, std::int64_t lda,        \
                            std::int64_t stride_a, std::int64_t* ipiv,        \
                            std::int64_t stride_ipiv, T* b, std::int64_t ldb, \
                            std::int64_t stride_b, std::int64_t batch_size,   \
                            T* scratchpad, std::int64_t scratchpad_size)      \
    {                                                                         \
        FuncName(queue, nontrans, m, nrhs, a, lda, stride_a, ipiv,            \
                 stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,       \
                 scratchpad_size);                                            \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_ONEMKL_BATCH_GETRS(float, oneapi::mkl::lapack::getrs_batch);
GKO_BIND_ONEMKL_BATCH_GETRS(double, oneapi::mkl::lapack::getrs_batch);
GKO_BIND_ONEMKL_BATCH_GETRS(std::complex<float>,
                            oneapi::mkl::lapack::getrs_batch);
GKO_BIND_ONEMKL_BATCH_GETRS(std::complex<double>,
                            oneapi::mkl::lapack::getrs_batch);
template <typename ValueType>
GKO_BIND_ONEMKL_BATCH_GETRS(ValueType, detail::not_implemented);

#undef GKO_BIND_ONEMKL_BATCH_GETRS


}  // namespace onemkl
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_ONEMKL_BINDINGS_HPP_
