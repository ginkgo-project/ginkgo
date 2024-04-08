// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
namespace detail {


template <typename LeftList, typename RightList, typename... Result>
struct cartesian_type_product {};

template <template <typename...> class OuterWrapper, typename FirstLeft,
          typename... RemainingLeftArgs, typename... RightArgs,
          typename... Result>
struct cartesian_type_product<OuterWrapper<FirstLeft, RemainingLeftArgs...>,
                              OuterWrapper<RightArgs...>, Result...>
    : cartesian_type_product<OuterWrapper<RemainingLeftArgs...>,
                             OuterWrapper<RightArgs...>, Result...,
                             std::tuple<FirstLeft, RightArgs>...> {};

template <template <typename...> class OuterWrapper, typename... RightArgs,
          typename... Result>
struct cartesian_type_product<OuterWrapper<>, OuterWrapper<RightArgs...>,
                              Result...> {
    using type = OuterWrapper<Result...>;
};

template <typename ExistingCombinationList, typename NewElementList,
          typename... Result>
struct add_to_cartesian_type_product {};

template <template <typename...> class OuterWrapper,
          typename... CurrentCombinationArgs,
          typename... RemainingOldCombinations, typename... NewElementArgs,
          typename... Result>
struct add_to_cartesian_type_product<
    OuterWrapper<std::tuple<CurrentCombinationArgs...>,
                 RemainingOldCombinations...>,
    OuterWrapper<NewElementArgs...>, Result...>
    : add_to_cartesian_type_product<
          OuterWrapper<RemainingOldCombinations...>,
          OuterWrapper<NewElementArgs...>, Result...,
          std::tuple<CurrentCombinationArgs..., NewElementArgs>...> {};

template <template <typename...> class OuterWrapper, typename... NewElementArgs,
          typename... Result>
struct add_to_cartesian_type_product<
    OuterWrapper<>, OuterWrapper<NewElementArgs...>, Result...> {
    using type = OuterWrapper<Result...>;
};

template <typename NewElementList, typename ExistingCombinationList,
          typename... Result>
struct add_to_cartesian_type_product_left {};

template <template <typename...> class OuterWrapper, typename... NewElementArgs,
          typename... CurrentCombinationArgs,
          typename... RemainingOldCombinations, typename... Result>
struct add_to_cartesian_type_product_left<
    OuterWrapper<NewElementArgs...>,
    OuterWrapper<std::tuple<CurrentCombinationArgs...>,
                 RemainingOldCombinations...>,
    Result...>
    : add_to_cartesian_type_product_left<
          OuterWrapper<NewElementArgs...>,
          OuterWrapper<RemainingOldCombinations...>, Result...,
          std::tuple<NewElementArgs, CurrentCombinationArgs...>...> {};

template <template <typename...> class OuterWrapper, typename... NewElementArgs,
          typename... Result>
struct add_to_cartesian_type_product_left<OuterWrapper<NewElementArgs...>,
                                          OuterWrapper<>, Result...> {
    using type = OuterWrapper<Result...>;
};

template <typename FirstList, typename SecondList>
struct merge_type_list {};

template <template <typename...> class OuterWrapper, typename... Args1,
          typename... Args2>
struct merge_type_list<OuterWrapper<Args1...>, OuterWrapper<Args2...>> {
    using type = OuterWrapper<Args1..., Args2...>;
};


template <template <typename...> class NewOuterWrapper,
          typename OldOuterWrapper>
struct change_outer_wrapper {};

template <template <typename...> class NewOuterWrapper,
          template <typename...> class OldOuterWrapper, typename... Args>
struct change_outer_wrapper<NewOuterWrapper, OldOuterWrapper<Args...>> {
    using type = NewOuterWrapper<Args...>;
};


template <template <typename...> class NewInnerWrapper, typename ListType>
struct add_internal_wrapper {};

template <template <typename...> class NewInnerWrapper,
          template <typename...> class OuterWrapper, typename... Args>
struct add_internal_wrapper<NewInnerWrapper, OuterWrapper<Args...>> {
    using type = OuterWrapper<NewInnerWrapper<Args>...>;
};


}  // namespace detail


template <typename LeftList, typename RightList>
using cartesian_type_product_t =
    typename detail::cartesian_type_product<LeftList, RightList>::type;

template <typename ExistingCombinationList, typename NewElementList>
using add_to_cartesian_type_product_t =
    typename detail::add_to_cartesian_type_product<ExistingCombinationList,
                                                   NewElementList>::type;

template <typename NewElementList, typename ExistingCombinationList>
using add_to_cartesian_type_product_left_t =
    typename detail::add_to_cartesian_type_product_left<
        NewElementList, ExistingCombinationList>::type;

template <typename FirstList, typename SecondList>
using merge_type_list_t =
    typename detail::merge_type_list<FirstList, SecondList>::type;

template <template <typename...> class NewInnerWrapper, typename ListType>
using add_internal_wrapper_t =
    typename detail::add_internal_wrapper<NewInnerWrapper, ListType>::type;

template <template <typename...> class NewOuterWrapper, typename ListType>
using change_outer_wrapper_t =
    typename detail::change_outer_wrapper<NewOuterWrapper, ListType>::type;


using RealValueTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<float>;
#else
    ::testing::Types<float, double>;
#endif

using ComplexValueTypes = add_internal_wrapper_t<std::complex, RealValueTypes>;

using ValueTypes = merge_type_list_t<RealValueTypes, ComplexValueTypes>;

using IndexTypes = ::testing::Types<int32, int64>;

using LocalGlobalIndexTypes =
    ::testing::Types<std::tuple<int32, int32>, std::tuple<int32, int64>,
                     std::tuple<int64, int64>>;

using PODTypes =
    merge_type_list_t<merge_type_list_t<RealValueTypes, IndexTypes>,
                      ::testing::Types<size_type>>;

using ComplexAndPODTypes = merge_type_list_t<ComplexValueTypes, PODTypes>;

using ValueIndexTypes = cartesian_type_product_t<ValueTypes, IndexTypes>;

using RealValueIndexTypes =
    cartesian_type_product_t<RealValueTypes, IndexTypes>;

using ComplexValueIndexTypes =
    cartesian_type_product_t<ComplexValueTypes, IndexTypes>;

using TwoValueIndexType = add_to_cartesian_type_product_t<
    merge_type_list_t<
        cartesian_type_product_t<RealValueTypes, RealValueTypes>,
        cartesian_type_product_t<ComplexValueTypes, ComplexValueTypes>>,
    IndexTypes>;

using ValueLocalGlobalIndexTypes =
    add_to_cartesian_type_product_left_t<ValueTypes, LocalGlobalIndexTypes>;


template <typename Precision, typename OutputType>
struct reduction_factor {
    using nc_output = remove_complex<OutputType>;
    using nc_precision = remove_complex<Precision>;
    static constexpr nc_output value{
        std::numeric_limits<nc_precision>::epsilon() * nc_output{10} *
        (gko::is_complex<Precision>() ? nc_output{1.4142} : one<nc_output>())};
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


#endif  // GKO_CORE_TEST_UTILS_HPP_
