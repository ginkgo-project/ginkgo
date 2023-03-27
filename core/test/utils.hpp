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

#include <ginkgo/core/base/half.hpp>
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


/**
 * @see cartesian_type_product_t for details.
 *
 * The wrapper / list type needs to be a structure that can take an arbitrary
 * amount of types. An example for this wrapper is std::tuple, but a simple
 * `template<typename... Args> struct wrapper {};` also works.
 * Both the left and right list need to use the same wrapper, otherwise, the
 * type specialization fails.
 *
 * This structure uses partial specialization to:
 * - remove the OuterWrapper of both the left and right list;
 * - extracts a single element of the left list and combines it with all
 *   elements of the right list;
 * - after no elements remain in the left list, put all generated combinations
 *   together and wrap them in an OuterWrapper again
 *
 * This structure uses inheritance to store the combinations it creates in the
 * parameter pack Result, which will be wrapped in the original OuterWrapper
 * after all parameters from the LeftList have been processed (in the last
 * specialization).
 */
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


/**
 * @see add_to_cartesian_type_product_t for details.
 *
 * Uses a similar technique to cartesian_type_product.
 * It also uses the parameter pack Result to store the interim results, which
 * will be put in the OuterWrapper after all inputs have been processed.
 */
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


/**
 * @see add_to_cartesian_type_product_left_t for details.
 */
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


/**
 * @see merge_type_lists_t for details.
 */
template <typename FirstList, typename SecondList>
struct merge_type_list {};

template <template <typename...> class OuterWrapper, typename... Args1,
          typename... Args2>
struct merge_type_list<OuterWrapper<Args1...>, OuterWrapper<Args2...>> {
    using type = OuterWrapper<Args1..., Args2...>;
};


/**
 * @see change_outer_wrapper_t for details.
 */
template <template <typename...> class NewOuterWrapper,
          typename OldOuterWrapper>
struct change_outer_wrapper {};

template <template <typename...> class NewOuterWrapper,
          template <typename...> class OldOuterWrapper, typename... Args>
struct change_outer_wrapper<NewOuterWrapper, OldOuterWrapper<Args...>> {
    using type = NewOuterWrapper<Args...>;
};


/**
 * @see add_inner_wrapper_t for details.
 */
template <template <typename...> class NewInnerWrapper, typename ListType>
struct add_inner_wrapper {};

template <template <typename...> class NewInnerWrapper,
          template <typename...> class OuterWrapper, typename... Args>
struct add_inner_wrapper<NewInnerWrapper, OuterWrapper<Args...>> {
    using type = OuterWrapper<NewInnerWrapper<Args>...>;
};


}  // namespace detail


/**
 * This type alias creates a cartesian product of the types in the left list
 * with the types of the right list and stores the combination in a std::tuple.
 * The resulting type is a list (it will be the same type wrapper as the left
 * and right list) of `std::tuple`.
 * Example:
 * ```
 * // Here, we use std::tuple as the outer type wrapper.
 * using left_list = std::tuple<a1, a2, a3>;
 * using right_list = std::tuple<b1, b2>;
 * using result = cartesian_type_product_t<left_list, right_list>;
 * // result = std::tuple<std::tuple<a1, b1>, std::tuple<a1, b2>,
 * //                     std::tuple<a2, b1>, std::tuple<a2, b2>,
 * //                     std::tuple<a3, b1>, std::tuple<a3, b2>>;
 * ```
 *
 * @tparam LeftList  A wrapper type (like std::tuple) containing the list of
 *                   types that you want to create the cartesian product with.
 *                   The parameters of this list will be the left type in the
 *                   resulting `std::tuple`
 * @tparam RightList  Similar to the LeftList. Must use the same outer wrapper
 *                    as the LeftList.
 */
template <typename LeftList, typename RightList>
using cartesian_type_product_t =
    typename detail::cartesian_type_product<LeftList, RightList>::type;


/**
 * This type alias is intended to be used with cartesian_type_product_t in order
 * to create a more than two dimensional cartesian product by adding one element
 * to the result per call.
 * This structure expects the left list to have all elements of the type
 * std::tuple (as it is returned from cartesian_type_product_t) and the right
 * list of elements you want to add to those tuples.
 * It creates a new list where it adds all combinations of the std::tuple with
 * the new element list as a new member of the std::tuple to the right side.
 * Example:
 * ```
 * template<typename... Args>
 * using t = std::tuple<Args>;  // use this alias to increase readability
 * using left_combinations = t<t<a1, b1>, t<a1, b2>>;
 * using right_new = t<n1, n2>;
 * using new_list =
 *     add_to_cartesian_type_product_t<left_combinations, right_new>;
 * // new_list = t<t<a1, b1, n1>, t<a1, b1, n2>, t<a1, b2, n1>, t<a1, b2, n2>>;
 * ```
 *
 * @tparam ExistingCombinationList  An outer type wrapper containing different
 *                                  std::tuples that you want to add elements to
 * @tparam NewElementList  The list of new elements (using the same outer
 *                         wrapper as ExistingCombinationList) you want to
 *                         create all possible combinations with. These elements
 *                         will be added to the right of each std::tuple
 */
template <typename ExistingCombinationList, typename NewElementList>
using add_to_cartesian_type_product_t =
    typename detail::add_to_cartesian_type_product<ExistingCombinationList,
                                                   NewElementList>::type;


/**
 * This type alias is very similar to add_to_cartesian_type_product_t. It only
 * differs in where the new element is added to the `std::tuple`, which is to
 * the left here, and the order of the parameter.
 * Example:
 * ```
 * template<typename... Args> using t = std::tuple<Args>;
 * using right_combinations = t<t<a1, b1>, t<a1, b2>>;
 * using left_new = t<n1, n2>;
 * using new_list =
 *     add_to_cartesian_type_product_left_t<left_new, right_combinations>;
 * // new_list = t<t<n1, a1, b1>, t<n2, a1, b1>, t<n1, a1, b2>, t<n2, a1, b2>>;
 * ```
 *
 * @tparam NewElementList  The list of new elements (using the same outer
 *                         wrapper as ExistingCombinationList) you want to
 *                         create all possible combinations with. These elements
 *                         will be added to the left of each std::tuple
 * @tparam ExistingCombinationList  An outer type wrapper containing different
 *                                  std::tuples that you want to add elements to
 */
template <typename NewElementList, typename ExistingCombinationList>
using add_to_cartesian_type_product_left_t =
    typename detail::add_to_cartesian_type_product_left<
        NewElementList, ExistingCombinationList>::type;


/**
 * Merges two lists into a single list.
 * The left and right list need to use the same type wrapper, which will also be
 * the resulting wrapper containing elements of both lists. The order of the
 * left and right list are preserved. The resulting list will have all elements
 * of the left list, followed by all elements of the right list.
 *
 * @tparam FirstList  The first list of types
 * @tparam SecondList  The second list of types. The type wrapper needs to be
 *                     the same as for FirstList.
 */
template <typename FirstList, typename SecondList>
using merge_type_list_t =
    typename detail::merge_type_list<FirstList, SecondList>::type;


/**
 * This type alias can change the outer type wrapper to the new, given one.
 * Example:
 * ```
 * template <typename... Args>
 * struct type_wrapper {};
 * using old_list = std::tuple<int, double, short>;
 * using new_list = change_outer_wrapper_t<type_wrapper, old_list>;
 * // new_list = type_wrapper<int, double, short>;
 * ```
 *
 * @tparam NewOuterWrapper  the new wrapper you want to use as the new outer
 *                          wrapper
 * @tparam ListType  The list of types where you want to replace the outer
 *                   wrapper.
 */
template <template <typename...> class NewOuterWrapper, typename ListType>
using change_outer_wrapper_t =
    typename detail::change_outer_wrapper<NewOuterWrapper, ListType>::type;


/**
 * Creates a type list (the outer wrapper stays the same) where each original
 * type is wrapped into the given NewInnerWrapper.
 * Example:
 * ```
 * using new_type =
 *     add_inner_wrapper<std::complex, std::tuple<float, double>>;
 * // new_type = std::tuple<std::complex<float>, std::complex<double>>;
 * ```
 *
 * @tparam NewInnerWrapper  the new wrapper you want to use to wrap each type
 *                          in the list
 * @tparam ListType  The list of types where you want to add a wrapper to each
 */
template <template <typename...> class NewInnerWrapper, typename ListType>
using add_inner_wrapper_t =
    typename detail::add_inner_wrapper<NewInnerWrapper, ListType>::type;


using RealValueTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<gko::half, float>;
#else
    ::testing::Types<gko::half, float, double>;
#endif

using RealValueTypesNoHalf =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<float>;
#else
    ::testing::Types<float, double>;
#endif

using ComplexValueTypes = add_inner_wrapper_t<std::complex, RealValueTypes>;

using ComplexValueTypesNoHalf =
    add_inner_wrapper_t<std::complex, RealValueTypesNoHalf>;

using ValueTypes = merge_type_list_t<RealValueTypes, ComplexValueTypes>;

using IndexTypes = ::testing::Types<int32, int64>;

using IntegerTypes = merge_type_list_t<IndexTypes, ::testing::Types<size_type>>;

using LocalGlobalIndexTypes =
    ::testing::Types<std::tuple<int32, int32>, std::tuple<int32, int64>,
                     std::tuple<int64, int64>>;

using PODTypes = merge_type_list_t<RealValueTypes, IntegerTypes>;

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
    add_to_cartesian_type_product_left_t<ValueTypesNoHalf,
                                         LocalGlobalIndexTypes>;


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
struct next_precision_impl<gko::half> {
    using type = gko::half;
};

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
