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

#ifndef GKO_CORE_TEST_UTILS_HPP_
#define GKO_CORE_TEST_UTILS_HPP_


#include <complex>
#include <initializer_list>
#include <type_traits>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/test/utils/assertions.hpp"
#include "core/test/utils/matrix_generator.hpp"


namespace gko {
namespace test {


using ValueTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;


using ComplexValueTypes =
    ::testing::Types<std::complex<float>, std::complex<double>>;


using IndexTypes = ::testing::Types<gko::int32, gko::int64>;


using ValueAndIndexTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>,
                     gko::int32, gko::int64, gko::size_type>;


using ValueIndexTypes = ::testing::Types<
    std::tuple<float, gko::int32>, std::tuple<double, gko::int32>,
    std::tuple<std::complex<float>, gko::int32>,
    std::tuple<std::complex<double>, gko::int32>, std::tuple<float, gko::int64>,
    std::tuple<double, gko::int64>, std::tuple<std::complex<float>, gko::int64>,
    std::tuple<std::complex<double>, gko::int64>>;


using RealValueIndexTypes = ::testing::Types<
    std::tuple<float, gko::int32>, std::tuple<double, gko::int32>,
    std::tuple<float, gko::int64>, std::tuple<double, gko::int64>>;


using ComplexValueIndexTypes =
    ::testing::Types<std::tuple<std::complex<float>, gko::int32>,
                     std::tuple<std::complex<double>, gko::int32>,
                     std::tuple<std::complex<float>, gko::int64>,
                     std::tuple<std::complex<double>, gko::int64>>;


template <typename T>
struct reduction_factor {
    static constexpr gko::remove_complex<T> value =
        std::is_same<gko::remove_complex<T>, float>::value ? 1.0e-7 : 1.0e-14;
};


template <typename T>
constexpr gko::remove_complex<T> reduction_factor<T>::value;


}  // namespace test
}  // namespace gko


template <typename T>
using r = typename gko::test::reduction_factor<T>;


template <typename T>
using I = std::initializer_list<T>;


#endif  // GKO_CORE_TEST_UTILS_HPP_
