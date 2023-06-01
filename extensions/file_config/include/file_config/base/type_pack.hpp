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

#ifndef GKO_PUBLIC_EXT_FILE_CONFIG_BASE_TYPE_PACK_HPP_
#define GKO_PUBLIC_EXT_FILE_CONFIG_BASE_TYPE_PACK_HPP_


#include <type_traits>


#include <ginkgo/core/synthesizer/containers.hpp>


namespace gko {
namespace extensions {
namespace file_config {


template <typename... Types>
using type_list = ::gko::syn::type_list<Types...>;


/**
 * tt_list has the same purpose as type_list, but we use separate type to
 * distinguish them in span
 *
 * @tparam ...Types  the types
 */
template <typename... Types>
struct tt_list {};


/**
 * concatenate is to concatenate tt_list and others type to tt_list.
 *
 * @tparam K  the type
 * @tparam T  the type
 */
template <typename K, typename T>
struct concatenate {
    using type = tt_list<K, T>;
};

template <typename... Types, typename T>
struct concatenate<tt_list<Types...>, T> {
    using type = tt_list<Types..., T>;
};

template <typename T, typename... Types>
struct concatenate<T, tt_list<Types...>> {
    using type = tt_list<T, Types...>;
};

template <typename... Types1, typename... Types2>
struct concatenate<tt_list<Types1...>, tt_list<Types2...>> {
    using type = tt_list<Types1..., Types2...>;
};


/**
 * concat is to concatenate type_list and others type to type_list.
 *
 * @tparam K  the type
 * @tparam T  the type
 */
template <typename K, typename T>
struct concat {
    using type = type_list<K, T>;
};
template <typename... Types, typename T>
struct concat<type_list<Types...>, T> {
    using type = type_list<Types..., T>;
};

template <typename T, typename... Types>
struct concat<T, type_list<Types...>> {
    using type = type_list<T, Types...>;
};

template <typename... Types1, typename... Types2>
struct concat<type_list<Types1...>, type_list<Types2...>> {
    using type = type_list<Types1..., Types2...>;
};


/**
 * is_tt_list returns true if the type is tt_list.
 *
 * @tparam T  the type
 */
template <typename T>
struct is_tt_list : public std::integral_constant<bool, false> {};

template <typename... T>
struct is_tt_list<tt_list<T...>> : public std::integral_constant<bool, true> {};


/**
 * span_type (kronecker product) is to build a tt_list from each combination of
 * two tt_list (or pure type). The result from span_type<tt_list<T1, T2, ...>,
 * tt_list<K1, K2, ...>> is tt_list<type_list<T1, K1>, type_list<T1, K2>, ...,
 * type_list<T2, K1>, type_list<T2, K2>, ...>
 *
 * @tparam K  the type or tt_list
 * @tparam T  the type or tt_list
 */
template <typename K, typename T, typename = void>
struct span_type {
    using type = tt_list<typename concat<K, T>::type>;
};

// handle pure type x tt_list end case
template <typename K, typename T>
struct span_type<K, tt_list<T>,
                 typename std::enable_if<!is_tt_list<K>::value>::type> {
    using type = tt_list<typename concat<K, T>::type>;
};

// handle pure type x tt_list
template <typename K, typename T, typename... TT>
struct span_type<K, tt_list<T, TT...>,
                 typename std::enable_if<!is_tt_list<K>::value>::type> {
    using type =
        typename concatenate<typename span_type<K, T>::type,
                             typename span_type<K, tt_list<TT...>>::type>::type;
};

// handle end case of tt_list x any type
template <typename K, typename T>
struct span_type<tt_list<K>, T> {
    using type = typename span_type<K, T>::type;
};

// handle tt_list x any type
template <typename K, typename... K1, typename T>
struct span_type<tt_list<K, K1...>, T> {
    using type =
        typename concatenate<typename span_type<K, T>::type,
                             typename span_type<tt_list<K1...>, T>::type>::type;
};


/**
 * span_list is a extension for span_type. It can span varadic template
 * parameters. The result has the same rules as span_type, which expand the last
 * argument first.
 *
 * @tparam K  the type or tt_list
 * @tparam ...T  the rest types or tt_lists
 */
template <typename K, typename... T>
struct span_list {};

// handle two lists through span_type
template <typename K, typename T>
struct span_list<K, T> {
    using type = typename span_type<K, T>::type;
};

// handle more than two lists, we get the span_type of the first two and then
// continue to next span operation.
template <typename K, typename T, typename... S>
struct span_list<K, T, S...> {
    using type = typename span_list<typename span_type<K, T>::type, S...>::type;
};

template <typename... K>
struct span_list<tt_list<K...>> {
    using type = tt_list<K...>;
};


}  // namespace file_config
}  // namespace extensions
}  // namespace gko

#endif  // GKO_PUBLIC_EXT_FILE_CONFIG_BASE_TYPE_PACK_HPP_
