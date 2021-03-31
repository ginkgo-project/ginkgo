/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
#include <limits>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/extended_float.hpp"
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

using RealValueAndIndexTypes =
    ::testing::Types<float, double, gko::int32, gko::int64, gko::size_type>;

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


template <typename Precision, typename OutputType>
struct reduction_factor {
    using nc_output = remove_complex<OutputType>;
    using nc_precision = remove_complex<Precision>;
    static constexpr nc_output value{
        std::numeric_limits<nc_precision>::epsilon() * nc_output{10}};
};


template <typename Precision, typename OutputType>
constexpr remove_complex<OutputType>
    reduction_factor<Precision, OutputType>::value;


}  // namespace test
}  // namespace gko


template <typename Precision, typename OutputType = Precision>
using r = typename gko::test::reduction_factor<Precision, OutputType>;


template <typename Precision1, typename Precision2>
constexpr double r_mixed()
{
    return std::max<double>(r<Precision1>::value, r<Precision2>::value);
}


template <typename T>
using I = std::initializer_list<T>;


#endif  // GKO_CORE_TEST_UTILS_HPP_
