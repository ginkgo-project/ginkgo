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

#ifndef GKO_BENCHMARK_UTILS_TYPES_HPP_
#define GKO_BENCHMARK_UTILS_TYPES_HPP_


#include <complex>


#include <ginkgo/core/base/math.hpp>


using itype = gko::int32;


#if defined(GKO_BENCHMARK_USE_DOUBLE_PRECISION) ||         \
    defined(GKO_BENCHMARK_USE_SINGLE_PRECISION) ||         \
    defined(GKO_BENCHMARK_USE_DOUBLE_COMPLEX_PRECISION) || \
    defined(GKO_BENCHMARK_USE_SINGLE_COMPLEX_PRECISION)
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
