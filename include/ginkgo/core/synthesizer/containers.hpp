// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SYNTHESIZER_CONTAINERS_HPP_
#define GKO_PUBLIC_CORE_SYNTHESIZER_CONTAINERS_HPP_


#include <array>
#include <type_traits>


namespace gko {
/**
 * @brief The Synthesizer namespace.
 *
 * @ingroup syn
 */
namespace syn {


/**
 * value_list records several values with the same type in template.
 *
 * @tparam T  the value type of the list
 * @tparam Values  the values in the list
 */
template <typename T, T... Values>
struct value_list {};


/**
 * type_list records several types in template
 *
 * @tparam Types  the types in the list
 */
template <typename... Types>
struct type_list {};


/**
 * range records start, end, step in template
 *
 * @tparam Start  start of range
 * @tparam End  end of range
 * @tparam Step  step of range. default is 1
 */
template <int Start, int End, int Step = 1>
struct range {};


namespace detail {


/**
 * concatenate_impl base type
 *
 * @tparam List1  the first List
 * @tparam List2  the second List
 */
template <typename List1, typename List2>
struct concatenate_impl;

/**
 * concatenate_impl specializes for two value_list with the same value type.
 *
 * @tparam T  the value type of two value_list
 * @tparam Values  the values of the first list
 * @tparam Values  the values of the second list
 */
template <typename T, T... Values1, T... Values2>
struct concatenate_impl<value_list<T, Values1...>, value_list<T, Values2...>> {
    using type = value_list<T, Values1..., Values2...>;
};


}  // namespace detail


/**
 * concatenate combines two value_list into one value_list.
 *
 * @tparam List1  the first list
 * @tparam List2  the second list
 */
template <typename List1, typename List2>
using concatenate = typename detail::concatenate_impl<List1, List2>::type;


namespace detail {


/**
 * as_list_impl base type
 *
 * @tparam T  the input template
 */
template <typename T, typename = void>
struct as_list_impl;

/**
 * as_list_impl specializes for the value_list
 *
 * @tparam T  the value_list type
 * @tparam Values  the values of value_list
 */
template <typename T, T... Values>
struct as_list_impl<value_list<T, Values...>> {
    using type = value_list<T, Values...>;
};

/**
 * as_list_impl specializes for the type_list
 *
 * @tparam ...Types  the types of type_list
 */
template <typename... Types>
struct as_list_impl<type_list<Types...>> {
    using type = type_list<Types...>;
};

/**
 * as_list_impl specializes for the range. This is the recursive case. It will
 * concatenate Start and range<Start + Step, End, Step>.
 *
 * @tparam int  the start of range
 * @tparam int  the end of range
 * @tparam int  the step of range
 */
template <int Start, int End, int Step>
struct as_list_impl<range<Start, End, Step>, std::enable_if_t<(Start < End)>> {
    using type = concatenate<
        value_list<int, Start>,
        typename as_list_impl<range<Start + Step, End, Step>>::type>;
};

/**
 * as_list_impl specializes for the range. This is the end case.
 *
 * @tparam int  the start of range
 * @tparam int  the end of range
 * @tparam int  the step of range
 */
template <int Start, int End, int Step>
struct as_list_impl<range<Start, End, Step>, std::enable_if_t<(Start >= End)>> {
    using type = value_list<int>;
};


}  // namespace detail


/**
 * as_list<T> gives the alias type of as_list_impl<T>::type. It gives a list
 * (itself) if input is already a list, or generates list type from range input.
 *
 * @tparam T  list or range
 */
template <typename T>
using as_list = typename detail::as_list_impl<T>::type;


/**
 * as_array<T> returns the array from value_list. It will be helpful if using
 * for in runtime on the array.
 *
 * @tparam T  the type of value_list
 * @tparam Value  the values of value_list
 *
 * @param value_list  the input value_list
 *
 * @return std::array  the std::array contains the values of value_list
 */
template <typename T, T... Value>
constexpr std::array<T, sizeof...(Value)> as_array(value_list<T, Value...> vl)
{
    return std::array<T, sizeof...(Value)>{Value...};
}


}  // namespace syn
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SYNTHESIZER_CONTAINERS_HPP_
