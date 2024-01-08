// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_ONEMKL_BINDINGS_HPP_
#define GKO_DPCPP_BASE_ONEMKL_BINDINGS_HPP_


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

}  // namespace onemkl
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_ONEMKL_BINDINGS_HPP_
