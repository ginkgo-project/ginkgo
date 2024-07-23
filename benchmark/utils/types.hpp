// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_TYPES_HPP_
#define GKO_BENCHMARK_UTILS_TYPES_HPP_


#include <complex>


#include <ginkgo/core/base/math.hpp>


using itype = gko::int32;


#if defined(GKO_BENCHMARK_USE_DOUBLE_PRECISION) ||         \
    defined(GKO_BENCHMARK_USE_SINGLE_PRECISION) ||         \
    defined(GKO_BENCHMARK_USE_DOUBLE_COMPLEX_PRECISION) || \
    defined(GKO_BENCHMARK_USE_SINGLE_COMPLEX_PRECISION) || \
    defined(GKO_BENCHMARK_USE_HALF_PRECISION)
// separate ifdefs to catch duplicate definitions
#ifdef GKO_BENCHMARK_USE_DOUBLE_PRECISION
using etype = double;
#endif
#ifdef GKO_BENCHMARK_USE_SINGLE_PRECISION
using etype = float;
#endif
#ifdef GKO_BENCHMARK_USE_DOUBLE_COMPLEX_PRECISION
using etype = std::complex<double>;
#endif
#ifdef GKO_BENCHMARK_USE_SINGLE_COMPLEX_PRECISION
using etype = std::complex<float>;
#endif
#ifdef GKO_BENCHMARK_USE_HALF_PRECISION
#include <ginkgo/core/base/half.hpp>
using etype = gko::half;
#endif
#else  // default to double precision
using etype = double;
#endif

using rc_etype = gko::remove_complex<etype>;


namespace detail {


// singly linked list of all our supported precisions
template <typename T>
struct next_precision_impl {};

template <>
struct next_precision_impl<float> {
    using type = double;
};

template <>
struct next_precision_impl<double> {
    using type = float;
};


template <typename T>
struct next_precision_impl<std::complex<T>> {
    using type = std::complex<typename next_precision_impl<T>::type>;
};


}  // namespace detail

template <typename T>
using next_precision = typename detail::next_precision_impl<T>::type;

#endif  // GKO_BENCHMARK_UTILS_TYPES_HPP_
