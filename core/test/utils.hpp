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

#ifndef GKO_CORE_TEST_UTILS_HPP_
#define GKO_CORE_TEST_UTILS_HPP_


#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils_helper.hpp>


#include "core/base/extended_float.hpp"
#include "core/test/utils/array_generator.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/value_generator.hpp"


namespace gko {
namespace test {


using ValueTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<float, std::complex<float>>;
#else
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
#endif

using ComplexValueTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<std::complex<float>>;
#else
    ::testing::Types<std::complex<float>, std::complex<double>>;
#endif

using RealValueTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<float>;
#else
    ::testing::Types<float, double>;
#endif


using IndexTypes = ::testing::Types<gko::int32, gko::int64>;


using LocalGlobalIndexTypes =
    ::testing::Types<std::tuple<int32, int32>, std::tuple<int32, int64>,
                     std::tuple<int64, int64>>;


using PODTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<float, gko::int32, gko::int64>;
#else
    ::testing::Types<float, double, gko::int32, gko::int64>;
#endif


using ValueAndIndexTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<float, std::complex<float>, gko::int32, gko::int64,
                     gko::size_type>;
#else
    ::testing::Types<float, double, std::complex<float>, std::complex<double>,
                     gko::int32, gko::int64, gko::size_type>;
#endif


using RealValueAndIndexTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<float, gko::int32, gko::int64, gko::size_type>;
#else
    ::testing::Types<float, double, gko::int32, gko::int64, gko::size_type>;
#endif


using ValueIndexTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<std::tuple<float, gko::int32>,
                     std::tuple<std::complex<float>, gko::int32>,
                     std::tuple<float, gko::int64>,
                     std::tuple<std::complex<float>, gko::int64>>;
#else
    ::testing::Types<
        std::tuple<float, gko::int32>, std::tuple<double, gko::int32>,
        std::tuple<std::complex<float>, gko::int32>,
        std::tuple<std::complex<double>, gko::int32>,
        std::tuple<float, gko::int64>, std::tuple<double, gko::int64>,
        std::tuple<std::complex<float>, gko::int64>,
        std::tuple<std::complex<double>, gko::int64>>;
#endif


using RealValueIndexTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<std::tuple<float, gko::int32>,
                     std::tuple<float, gko::int64>>;
#else
    ::testing::Types<
        std::tuple<float, gko::int32>, std::tuple<double, gko::int32>,
        std::tuple<float, gko::int64>, std::tuple<double, gko::int64>>;
#endif


using ComplexValueIndexTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<std::tuple<std::complex<float>, gko::int32>,
                     std::tuple<std::complex<float>, gko::int64>>;
#else
    ::testing::Types<std::tuple<std::complex<float>, gko::int32>,
                     std::tuple<std::complex<double>, gko::int32>,
                     std::tuple<std::complex<float>, gko::int64>,
                     std::tuple<std::complex<double>, gko::int64>>;
#endif


using TwoValueIndexType =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<
        std::tuple<float, float, gko::int32>,
        std::tuple<std::complex<float>, std::complex<float>, gko::int32>,
        std::tuple<float, float, gko::int64>,
        std::tuple<std::complex<float>, std::complex<float>, gko::int64>>;
#else
    ::testing::Types<
        std::tuple<float, float, gko::int32>,
        std::tuple<float, double, gko::int32>,
        std::tuple<double, double, gko::int32>,
        std::tuple<double, float, gko::int32>,
        std::tuple<std::complex<float>, std::complex<float>, gko::int32>,
        std::tuple<std::complex<float>, std::complex<double>, gko::int32>,
        std::tuple<std::complex<double>, std::complex<double>, gko::int32>,
        std::tuple<std::complex<double>, std::complex<float>, gko::int32>,
        std::tuple<float, float, gko::int64>,
        std::tuple<float, double, gko::int64>,
        std::tuple<double, double, gko::int64>,
        std::tuple<double, float, gko::int64>,
        std::tuple<std::complex<float>, std::complex<float>, gko::int64>,
        std::tuple<std::complex<float>, std::complex<double>, gko::int64>,
        std::tuple<std::complex<double>, std::complex<double>, gko::int64>,
        std::tuple<std::complex<double>, std::complex<float>, gko::int64>>;
#endif


using ValueLocalGlobalIndexTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<std::tuple<float, gko::int32, int32>,
                     std::tuple<float, gko::int32, int64>,
                     std::tuple<float, gko::int64, int64>,
                     std::tuple<std::complex<float>, gko::int32, int32>,
                     std::tuple<std::complex<float>, gko::int32, int64>,
                     std::tuple<std::complex<float>, gko::int64, int64>>;
#else
    ::testing::Types<std::tuple<float, gko::int32, int32>,
                     std::tuple<float, gko::int32, int64>,
                     std::tuple<float, gko::int64, int64>,
                     std::tuple<double, gko::int32, int32>,
                     std::tuple<double, gko::int32, int64>,
                     std::tuple<double, gko::int64, int64>,
                     std::tuple<std::complex<float>, gko::int32, int32>,
                     std::tuple<std::complex<float>, gko::int32, int64>,
                     std::tuple<std::complex<float>, gko::int64, int64>,
                     std::tuple<std::complex<double>, gko::int32, int32>,
                     std::tuple<std::complex<double>, gko::int32, int64>,
                     std::tuple<std::complex<double>, gko::int64, int64>>;
#endif


template <typename Precision, typename OutputType>
struct reduction_factor {
    using nc_output = remove_complex<OutputType>;
    using nc_precision = remove_complex<Precision>;
    static nc_output value;
};


template <typename Precision, typename OutputType>
remove_complex<OutputType> reduction_factor<Precision, OutputType>::value =
    std::numeric_limits<nc_precision>::epsilon() * nc_output{10} *
    (gko::is_complex<Precision>() ? nc_output{1.4142} : one<nc_output>());


}  // namespace test
}  // namespace gko


template <typename Precision, typename OutputType = Precision>
using r = typename gko::test::reduction_factor<Precision, OutputType>;


template <typename Precision1, typename Precision2>
constexpr double r_mixed()
{
    return std::max<double>(r<Precision1>::value, r<Precision2>::value);
}


template <typename PtrType>
gko::remove_complex<typename gko::detail::pointee<PtrType>::value_type>
inf_norm(PtrType&& mat, size_t col = 0)
{
    using T = typename gko::detail::pointee<PtrType>::value_type;
    using std::abs;
    using no_cpx_t = gko::remove_complex<T>;
    no_cpx_t norm = 0.0;
    for (std::size_t i = 0; i < mat->get_size()[0]; ++i) {
        no_cpx_t absEntry = abs(mat->at(i, col));
        if (norm < absEntry) norm = absEntry;
    }
    return norm;
}


template <typename T>
using I = std::initializer_list<T>;


struct TypenameNameGenerator {
    template <typename T>
    static std::string GetName(int i)
    {
        return gko::name_demangling::get_type_name(typeid(T));
    }
};


struct PairTypenameNameGenerator {
    template <typename T>
    static std::string GetName(int i)
    {
        static_assert(std::tuple_size<T>::value == 2, "expected a pair");
        return "<" +
               gko::name_demangling::get_type_name(
                   typeid(typename std::tuple_element<0, T>::type)) +
               ", " +
               gko::name_demangling::get_type_name(
                   typeid(typename std::tuple_element<1, T>::type)) +
               ">";
    }
};


struct TupleTypenameNameGenerator {
    template <typename T>
    static std::string GetName(int i)
    {
        static_assert(std::tuple_size<T>::value == 3, "expected a tuple");
        return "<" +
               gko::name_demangling::get_type_name(
                   typeid(typename std::tuple_element<0, T>::type)) +
               ", " +
               gko::name_demangling::get_type_name(
                   typeid(typename std::tuple_element<1, T>::type)) +
               ", " +
               gko::name_demangling::get_type_name(
                   typeid(typename std::tuple_element<2, T>::type)) +
               ">";
    }
};


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


#endif  // GKO_CORE_TEST_UTILS_HPP_
