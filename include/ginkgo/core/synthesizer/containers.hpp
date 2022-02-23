/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_SYNTHESIZER_CONTAINERS_HPP_
#define GKO_PUBLIC_CORE_SYNTHESIZER_CONTAINERS_HPP_


#include <array>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>


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
using value_list = std::integer_sequence<T, Values...>;


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
 * @tparam Lists  a list of std::integer_sequence
 */
template <typename... Lists>
struct concatenate_impl;

/**
 * concatenate_impl specialization for a single std::integer_sequence
 *
 * @tparam T  the value type of the std::integer_sequence
 * @tparam Values  the values of the list
 */
template <typename T, T... Values>
struct concatenate_impl<std::integer_sequence<T, Values...>> {
    using type = std::integer_sequence<T, Values...>;
};

/**
 * concatenate_impl specialization for multiple std::integer_sequence with the
 * same value type.
 *
 * @tparam T  the value type of two std::integer_sequence
 * @tparam Values1  the values of the first list
 * @tparam Values2  the values of the second list
 * @tparam Tail  the lists which have not been concatenated yet
 */
template <typename T, T... Values1, T... Values2, typename... Tail>
struct concatenate_impl<std::integer_sequence<T, Values1...>,
                        std::integer_sequence<T, Values2...>, Tail...> {
    using type = typename concatenate_impl<
        std::integer_sequence<T, Values1..., Values2...>, Tail...>::type;
};


}  // namespace detail


/**
 * concatenate an arbitrary number of std::integer_sequence with the same base
 * type into one
 *
 * @tparam Lists  a list of std::integer_sequence
 */
template <typename... Lists>
using concatenate = typename detail::concatenate_impl<Lists...>::type;


namespace detail {


/**
 * as_list_impl base type
 *
 * @tparam T  the input template
 */
template <typename T, typename = void>
struct as_list_impl;

/**
 * as_list_impl specializes for the std::integer_sequence
 *
 * @tparam T  the std::integer_sequence type
 * @tparam Values  the values of std::integer_sequence
 */
template <typename T, T... Values>
struct as_list_impl<std::integer_sequence<T, Values...>> {
    using type = std::integer_sequence<T, Values...>;
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
        std::integer_sequence<int, Start>,
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
    using type = std::integer_sequence<int>;
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
 * as_array<T> returns the array from std::integer_sequence. It will be helpful
 * if using for in runtime on the array.
 *
 * @tparam T  the type of std::integer_sequence
 * @tparam Value  the values of std::integer_sequence
 *
 * @param vl  the input std::integer_sequence
 *
 * @return std::array  the std::array contains the values of vl
 */
template <typename T, T... Value>
constexpr auto as_array(std::integer_sequence<T, Value...> vl)
{
    std::ignore = vl;
    return std::array<T, sizeof...(Value)>{Value...};
}

/**
 * as_value<T> returns the (first) value contained within an
 * std::integer_sequence. The empty case is made to fail on purpose.
 *
 * @tparam T  the type of std::integer_sequence
 * @tparam Value  the values of std::integer_sequence
 *
 * @param vl  the input std::integer_sequence
 *
 * @return the first value within vl
 */
template <typename T, T... Value>
constexpr auto as_value(std::integer_sequence<T, Value...> vl)
{
    static_assert(sizeof...(Value) > 0,
                  "Do not call as_value on an empty set!");
    return as_array(vl)[0];
}


namespace detail {


/**
 * This is the base type of a helper for sorting. It partitions the values
 * within an std::integer_sequence into three parts based on a pivot: values
 * lower than required, values equal, values above the requirement.
 *
 * The results will depend on the ascending or descending order, which reverses
 * above and lower.
 *
 * For example, when considering
 * ```
 *    using idxs = std::integer_sequence<int, 7614, 453, 16, 9, 16, 0, 0>;
 *    using parts = partition_impl<16, idxs>; // the pivot is 16
 *    // Then effectively:
 *    // parts::lower = std::integer_sequence<int, 9, 0, 0>
 *    // parts::equal = std::integer_sequence<int, 16, 16>
 *    // parts::above = std::integer_sequence<int, 7614, 453>
 * ```
 *
 * @tparam i  the pivot value
 * @tparam ascending  whether to group in ascending or descending order
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <int i, bool ascending, typename T, T... Values>
struct partition_impl;

/**
 * The default case for partitioning an empty std::integer_sequence. All parts
 * are the empty set.
 *
 * @tparam i  the pivot value
 * @tparam ascending  whether to group in ascending or descending order
 * @tparam T  the type of the values
 */
template <int i, bool ascending, typename T>
struct partition_impl<i, ascending, std::integer_sequence<T>> {
    using lower = std::integer_sequence<T>;
    using equal = std::integer_sequence<T>;
    using above = std::integer_sequence<T>;
};

/**
 * The recursive case for partitioning an std::integer_sequence. The value v1 is
 * put in the matching part.
 *
 * @tparam i  the pivot value
 * @tparam ascending  whether to group in ascending or descending order
 * @tparam T  the type of the values
 * @tparam v1  the current value being processed
 * @tparam Values  the values left to process
 */
template <int i, bool ascending, typename T, T v1, T... Values>
struct partition_impl<i, ascending, std::integer_sequence<T, v1, Values...>> {
    using this_elt = std::integer_sequence<T, v1>;
    using empty = std::integer_sequence<T>;
    using recurse =
        partition_impl<i, ascending, std::integer_sequence<T, Values...>>;

    using lower = concatenate<
        std::conditional_t<(ascending ? v1 < i : v1 > i), this_elt, empty>,
        typename recurse::lower>;
    using equal = concatenate<std::conditional_t<(v1 == i), this_elt, empty>,
                              typename recurse::equal>;
    using above = concatenate<
        std::conditional_t<(ascending ? v1 > i : v1 < i), this_elt, empty>,
        typename recurse::above>;
};


/**
 * This is the base type of the sorting structure. It sorts the values within
 * an std::integer_sequence by using partition_t as a helper.
 *
 * The sorting will depend on the ascending or descending order and whether
 * duplicates are kept or not.
 *
 * For example, when considering
 * ```
 *    using idxs = std::integer_sequence<int, 7614, 453, 16, 9, 16, 0, 0>;
 *    using asc_dups = typename sort_impl<true, true, idxs>::type;
 *    using asc_nodups = typename sort_impl<true, false, idxs>::type;
 *    using desc_dups = typename sort_impl<false, true, idxs>::type;
 *    // Then effectively:
 *    // asc_dups = std::integer_sequence<int, 0, 0, 9, 16, 16, 453, 7614>
 *    // asc_nodups = std::integer_sequence<int, 0, 9, 16, 453, 7614>
 *    // desc_dups = std::integer_sequence<int, 7614, 453, 16, 16, 9, 0, 0>
 * ```
 *
 * @tparam ascending  whether to sort in ascending or descending order
 * @tparam keep_dups  whether to keep duplicates or not
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <bool ascending, bool keep_dups, typename T, T... Values>
struct sort_impl;

/**
 * The default case for sorting an empty std::integer_sequence. The result is an
 * empty set.
 *
 * @tparam ascending  whether to group in ascending or descending order
 * @tparam keep_dups  whether to keep duplicates or not
 * @tparam T  the type of the values
 */
template <bool ascending, bool keep_dups, typename T>
struct sort_impl<ascending, keep_dups, std::integer_sequence<T>> {
    using type = std::integer_sequence<T>;
};

/**
 * The recursive case for sorting an std::integer_sequence. The value v1 becomes
 * the partition_impl pivot. The obtained groups are concatenated in order.
 * Duplicates are removed by only populating `this_elt` instead of `this_elt +
 * parts::equal`.
 *
 * @tparam ascending  whether to group in ascending or descending order
 * @tparam keep_dups  whether to keep duplicates or not
 * @tparam T  the type of the values
 * @tparam v1  the current value and pivot for partition_impl
 * @tparam Values  the values to sort
 */
template <bool ascending, bool keep_dups, typename T, T v1, T... Values>
struct sort_impl<ascending, keep_dups,
                 std::integer_sequence<T, v1, Values...>> {
    using this_elt = std::integer_sequence<T, v1>;
    using empty = std::integer_sequence<T>;
    using parts =
        partition_impl<v1, ascending, std::integer_sequence<T, Values...>>;
    using sorted_inf =
        typename sort_impl<ascending, keep_dups, typename parts::lower>::type;
    using sorted_eq =
        concatenate<this_elt, std::conditional_t<keep_dups == true,
                                                 typename parts::equal, empty>>;
    using sorted_up =
        typename sort_impl<ascending, keep_dups, typename parts::above>::type;

    using type = concatenate<sorted_inf, sorted_eq, sorted_up>;
};


}  // namespace detail

/**
 * This is a helper interface for sorting an std::integer_sequence. It always
 * removes duplicate values.
 *
 * @see detail::sort_impl<bool, bool, typename T, T... Values>
 *
 * @tparam ascending  whether to group in ascending or descending order
 * @tparam T  the type of the values
 * @tparam Values  the values to sort
 */
template <bool ascending, typename T, T... Values>
using sort = typename detail::sort_impl<ascending, false, T, Values...>::type;

/**
 * This is sorting variant which keeps duplicates
 *
 * @see detail::sort_impl<bool, bool, typename T, T... Values>
 *
 * @tparam ascending  whether to group in ascending or descending order
 * @tparam T  the type of the values
 * @tparam Values  the values to sort
 */
template <bool ascending, typename T, T... Values>
using sort_keep =
    typename detail::sort_impl<ascending, true, T, Values...>::type;


namespace detail {


/**
 * This is the base type of the accessing an element of an std::integer_sequence
 * at a given index.
 *
 * For example, when considering
 * ```
 *    using idxs = std::integer_sequence<int, 7614, 453, 16, 9, 16, 0, 0>;
 *    using num1 = typename at_index_impl<0, idxs>::type;
 *    using num2 = typename at_index_impl<2, idxs>::type;
 *    using num5 = typename at_index_impl<5, idxs>::type;
 *    // Then effectively:
 *    // num1 = std::integer_sequence<int, 7614>
 *    // num2 = std::integer_sequence<int, 16>
 *    // num5 = std::integer_sequence<int, 0>
 * ```
 *
 * @tparam idx  the index to find
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <std::size_t idx, typename T, T... Values>
struct at_index_impl;

/**
 * The default case for accessing an element of an empty std::integer_sequence
 * at a given index. The result is an empty set.
 *
 * @tparam idx  the index to find
 * @tparam T  the type of the values
 */
template <std::size_t idx, typename T>
struct at_index_impl<idx, std::integer_sequence<T>> {
    using type = std::integer_sequence<T>;
};

/**
 * The recursive case for accessing an element of an std::integer_sequence at a
 * given index. Idx is counted down until 0, where v1 is the requested element.
 *
 * @tparam idx  the distance to the index to find
 * @tparam T  the type of the values
 * @tparam v1  the value being processed
 * @tparam Values  the values
 */
template <std::size_t idx, typename T, T v1, T... Values>
struct at_index_impl<idx, std::integer_sequence<T, v1, Values...>> {
    using recurse =
        typename at_index_impl<idx - 1,
                               std::integer_sequence<T, Values...>>::type;
    using type =
        std::conditional_t<(idx <= 0), std::integer_sequence<T, v1>, recurse>;
};


}  // namespace detail


/**
 * This is a helper interface for accessing an std::integer_sequence at a given
 * index.
 *
 * @see detail::at_index_impl<std::size_t, typename T, T... Values>
 *
 * @tparam idx  the index of the element to find
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <std::size_t idx, typename T, T... Values>
using at_index = typename detail::at_index_impl<idx, T, Values...>::type;


namespace detail {


/**
 * Access the element at the back of an std::integer_sequence. This is the base
 * type.
 *
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <typename T, T... Values>
struct back_impl;

/**
 * Access the element at the back of an std::integer_sequence. This is the
 * specialization for std::integer_sequence. We simply reuse at_index. We need
 * to unpack the std::integer_sequence in order to have the proper size of the
 * parameter pack `sizeof...(Values)`, otherwise it is always one.
 *
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <typename T, T... Values>
struct back_impl<std::integer_sequence<T, Values...>> {
    using type =
        at_index<sizeof...(Values) - 1, std::integer_sequence<T, Values...>>;
};


/**
 * Access the median element of an std::integer_sequence. This is the base type.
 *
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <typename T, T... Values>
struct median_impl;

/**
 * Access the median element of an std::integer_sequence. This is the
 * specialization for std::integer_sequence. We simply reuse at_index of the
 * middle element after calling sort. We need to unpack the
 * std::integer_sequence in order to have the proper size of the parameter pack
 * `sizeof...(Values)`, otherwise it is always one.
 *
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <typename T, T... Values>
struct median_impl<std::integer_sequence<T, Values...>> {
    using type =
        at_index<sizeof...(Values) / 2, std::integer_sequence<T, Values...>>;
};


}  // namespace detail


/**
 * This is a helper interface for accessing the front of an
 * std::integer_sequence.
 *
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <typename T, T... Values>
using front = at_index<0, T, Values...>;

/**
 * This is a helper interface for accessing the back of an
 * std::integer_sequence.
 *
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <typename T, T... Values>
using back = typename detail::back_impl<T, Values...>::type;

/**
 * This is a helper interface for accessing the median element of an
 * std::integer_sequence.
 *
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <typename T, T... Values>
using median = typename detail::median_impl<sort<true, T, Values...>>::type;


/**
 * This is a helper interface for accessing the minimum element of an
 * std::integer_sequence.
 *
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <typename T, T... Values>
using min = front<sort<true, T, Values...>>;


/**
 * This is a helper interface for accessing the minimum element of an
 * std::integer_sequence.
 *
 * @tparam T  the type of the values
 * @tparam Values  the values
 */
template <typename T, T... Values>
using max = front<sort<false, T, Values...>>;


namespace detail {

/**
 * This is the base type of a helper to merge_impl. It merges two lists of
 * std::integer_sequence by applying EncodingType::encode() on every pair of
 * values. In this case, the first list is a single element. This helps
 * logically split the double recursion needed for merging two general lists.
 *
 * For example, when considering
 * ```
 *    struct int_encoder {
 *        using can_encode = std::true_type;
 *        static constexpr auto encode(int v1, int v2) { return v1*v2; }
 *    };
 *    using idx1 = std::integer_sequence<int, 2>;
 *    using idx2 = std::integer_sequence<int, 4, 8>;
 *    using merged = typename merge_one_impl<int_encoder, idx1, idx2>::type;
 *    // Then effectively:
 *    // merged = std::integer_sequence<int, 8, 16>
 * ```
 *
 * @tparam EncodingType  the type used to encode values. It must at least look
 *                       like the encoder in the example above. ConfigSet is a
 *                       Ginkgo type which can encode.
 * @tparam Lists  the lists to merge and encode
 */
template <typename EncodingType, typename... Lists>
struct merge_one_impl;

/**
 * This is the base case for merge_one_impl's recursion. The second list is
 * empty.
 *
 * @tparam EncodingType  the type used to encode values
 * @tparam T  the type of the values
 * @tparam T  v1 the value from the first list
 */
template <typename EncodingType, typename T, T v1>
struct merge_one_impl<EncodingType, std::integer_sequence<T, v1>,
                      std::integer_sequence<T>> {
    using type = std::integer_sequence<T>;
};

/**
 * This is the recursive case for merge_one_impl. We merge and encode v1
 * with every subsequent v2 from the second std::integer_sequence.
 *
 * @tparam EncodingType  the type used to encode values
 * @tparam T  the type of the values
 * @tparam T  v1 the value from the first list being merged
 * @tparam T  v2 the value from the second list being merged
 * @tparam Values  the values not yet processed from the second list
 */
template <typename EncodingType, typename T, T v1, T v2, T... Values>
struct merge_one_impl<EncodingType, std::integer_sequence<T, v1>,
                      std::integer_sequence<T, v2, Values...>> {
    static_assert(
        std::is_same<typename EncodingType::can_encode, std::true_type>::value,
        "EncodingType must have encoding functionality.");

    using v1_as_seq = std::integer_sequence<T, v1>;
    using values_as_seq = std::integer_sequence<T, Values...>;
    using recurse = merge_one_impl<EncodingType, v1_as_seq, values_as_seq>;
    using type =
        concatenate<std::integer_sequence<T, EncodingType::encode(v1, v2)>,
                    typename recurse::type>;
};


/**
 * This is the base type of merge_impl. It merges two
 * std::integer_sequence and calls EncodingType::encode() on every pair of
 * values.
 *
 * For example, when considering
 * ```
 *    struct int_encoder {
 *        using can_encode = std::true_type;
 *        static constexpr auto encode(int v1, int v2) { return v1*v2; }
 *    };
 *    using idx1 = std::integer_sequence<int, 2, 3>;
 *    using idx2 = std::integer_sequence<int, 4, 8>;
 *    using merged = typename merge_impl<int_encoder, idx1, idx2>::type;
 *    // Then effectively:
 *    // merged = std::integer_sequence<int, 8, 16, 12, 24>
 * ```
 *
 * @see detail::merge_one_impl<typename EncodingType, typename ...Lists>
 *
 * @tparam EncodingType  the type used to encode values. It must at least look
 *                       like the encoder in the example above. ConfigSet is a
 *                       Ginkgo type which can encode.
 * @tparam Lists  the lists to merge and encode
 */
template <typename EncodingType, typename... Lists>
struct merge_impl;

/**
 * This is the base case for merge_impl's recursion. The first list has been
 * completely consumed.
 *
 * @tparam EncodingType  the type used to encode values
 * @tparam T  the type of the values
 * @tparam T  Values2 the values of the second list
 */
template <typename EncodingType, typename T, T... Values2>
struct merge_impl<EncodingType, std::integer_sequence<T>,
                  std::integer_sequence<T, Values2...>> {
    using type = std::integer_sequence<T>;
};

/**
 * This is the first recursive case for merge_impl. In this case, the second
 * list is empty. We only encode v1 one after the other.
 *
 * @tparam EncodingType  the type used to encode values
 * @tparam T  the type of the values
 * @tparam T  v1 the value from the first list being merged
 * @tparam Values1  the values left to consume from the first list
 * @tparam Values2  the values of the second list
 */
template <typename EncodingType, typename T, T v1, T... Values1>
struct merge_impl<EncodingType, std::integer_sequence<T, v1, Values1...>,
                  std::integer_sequence<T>> {
    using v1_as_seq = std::integer_sequence<T, v1>;
    using empty = std::integer_sequence<T>;
    using val1_as_seq = std::integer_sequence<T, Values1...>;
    using processed_v1 = std::integer_sequence<T, EncodingType::encode(v1)>;
    // move to the next v1
    using recurse = merge_impl<EncodingType, val1_as_seq, empty>;
    using type = concatenate<processed_v1, typename recurse::type>;
};

/**
 * This is the recursive case for merge_impl with a non empty Values2 list. We
 * call merge_one_impl for every v1 and element of Values2 until Values1 is
 * completely consumed.
 *
 * @tparam EncodingType  the type used to encode values
 * @tparam T  the type of the values
 * @tparam T  v1 the value from the first list being merged
 * @tparam Values1  the values left to consume from the first list
 * @tparam Values2  the values of the second list
 */
template <typename EncodingType, typename T, T v1, T... Values1, T... Values2>
struct merge_impl<EncodingType, std::integer_sequence<T, v1, Values1...>,
                  std::integer_sequence<T, Values2...>> {
    using v1_as_seq = std::integer_sequence<T, v1>;
    using val1_as_seq = std::integer_sequence<T, Values1...>;
    using val2_as_seq = std::integer_sequence<T, Values2...>;
    using process_v1 = merge_one_impl<EncodingType, v1_as_seq, val2_as_seq>;
    // move to the next v1
    using recurse = merge_impl<EncodingType, val1_as_seq, val2_as_seq>;
    using type = concatenate<typename process_v1::type, typename recurse::type>;
};


}  // namespace detail


/**
 * This is a helper interface for merging two lists of std::integer_sequence
 * using EncodingType as an encoder. It only merges two lists at once.
 * EncodingType must look like ConfigSet or the following:
 *
 * ```
 *    struct int_encoder {
 *        using can_encode = std::true_type;
 *        static constexpr auto encode(int v1, int v2) { return v1*v2; }
 *    };
 * ```
 *
 * @see detail::merge_impl<typename EncodingType, typename ...Lists>
 *
 * @tparam EncodingType  the type used to encode values. It must at least look
 *                       like the encoder in the example above. ConfigSet is a
 *                       Ginkgo type which can encode.
 * @tparam Lists  the lists to merge and encode
 */
template <typename EncodingType, typename... Lists>
using merge = typename detail::merge_impl<EncodingType, Lists...>::type;


}  // namespace syn
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SYNTHESIZER_CONTAINERS_HPP_
