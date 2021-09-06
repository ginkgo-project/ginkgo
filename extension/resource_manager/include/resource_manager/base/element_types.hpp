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

#ifndef GKOEXT_RESOURCE_MANAGER_BASE_ELEMENT_TYPES_HPP_
#define GKOEXT_RESOURCE_MANAGER_BASE_ELEMENT_TYPES_HPP_

#include <ginkgo/ginkgo.hpp>
#include <type_traits>
// #include <resource_manager/base/macro_helper.hpp>


namespace gko {
namespace extension {
namespace resource_manager {


template <typename... Types>
using type_list = ::gko::syn::type_list<Types...>;

/**
 * GET_STRING_PARTIAL is to generate a specialization of get_string, which gives
 * `get_string<_type> = _str`.
 *
 * @param _type  the type
 * @param _str  the corresponding string
 */
#define GET_STRING_PARTIAL(_type, _str)                                      \
    template <>                                                              \
    std::string get_string<_type>()                                          \
    {                                                                        \
        return _str;                                                         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

/**
 * get_string returns the string identifier of type
 *
 * @tparam T  the type
 *
 * @return the identifier string
 *
 * @note the identifier string must be identical among this system
 */
template <typename T>
std::string get_string();

GET_STRING_PARTIAL(double, "double");
GET_STRING_PARTIAL(float, "float");
GET_STRING_PARTIAL(gko::int32, "int");
GET_STRING_PARTIAL(gko::int64, "int64");
GET_STRING_PARTIAL(gko::uint32, "uint32");
using isai_lower =
    std::integral_constant<gko::preconditioner::isai_type,
                           gko::preconditioner::isai_type::lower>;
using isai_upper =
    std::integral_constant<gko::preconditioner::isai_type,
                           gko::preconditioner::isai_type::upper>;
using isai_general =
    std::integral_constant<gko::preconditioner::isai_type,
                           gko::preconditioner::isai_type::general>;
using isai_spd = std::integral_constant<gko::preconditioner::isai_type,
                                        gko::preconditioner::isai_type::spd>;
GET_STRING_PARTIAL(isai_lower, "isai_lower");
GET_STRING_PARTIAL(isai_upper, "isai_upper");
GET_STRING_PARTIAL(isai_general, "isai_general");
GET_STRING_PARTIAL(isai_spd, "isai_spd");


/**
 * @copydoc get_string<T>()
 *
 * @param T the type input
 *
 * @note this is another version such that allow arg input.
 */
template <typename T>
std::string get_string(T)
{
    return get_string<T>();
}

/**
 * get_string for the type_list ending case. it will return the item's
 * identifier.
 *
 * @tparam K  the type in the type_list of input
 *
 * @param type_list<K>  the type_list input
 */
template <typename K>
std::string get_string(type_list<K>)
{
    return get_string<K>();
}

/**
 * get_string for the type_list general case. it will return the first item's
 * identifier + the Rest items' identidier with `+` separator.
 *
 * @tparam K  the first type in the type_list of input
 * @tparam ...Rest  the rest types in the type_list of input
 *
 * @param type_list<K, Rest...>  the type_list input
 */
template <typename K, typename... Rest>
typename std::enable_if<(sizeof...(Rest) > 0), std::string>::type get_string(
    type_list<K, Rest...>)
{
    return get_string<K>() + "+" + get_string(type_list<Rest...>());
}

/**
 * get_the_type is to apply the type T into base.
 * get_the_type<base, T>::type = base<T>;
 *
 * @tparam base  the templated class
 * @tparam T  the arg of template
 */
template <template <typename...> class base, typename T>
struct get_the_type {
    using type = base<T>;
};

/**
 * get_the_type is to apply the type_list into base, which requires several
 * template parametrs. get_the_type<base, type_list<T...>>::type = base<T...>;
 *
 * @tparam base  the templated class
 * @tparam ...Rest  the args from type_list
 */
template <template <typename...> class base, typename... Rest>
struct get_the_type<base, type_list<Rest...>> {
    using type = base<Rest...>;
};

/**
 * get_the_factory_type is to get the Factory class from base<T>.
 * get_the_factory_type<base, T>::type = base<T>::Factory;
 *
 * @tparam base  the templated class
 * @tparam T  the arg of template
 */
template <template <typename...> class base, typename T>
struct get_the_factory_type {
    using type = typename get_the_type<base, T>::type::Factory;
};


/**
 * acutul_type is a struct to give the flexibilty to control some special cases.
 * For example, gives a integral_constant but use the value in practice.
 *
 * @tparam T  the type
 *
 * @note It is required because selection uses type_list as template input to
 * support several template. Thus, we use integral_constant to store value and
 * then transfer them by this class. For example,
 * ```
 * template <int i, typename VT>
 * struct actual_class{};
 * ```
 * We only pass the type_list as template input, so it only supports types.
 * We can also use type_list as base type or some specific class name to avoid
 * several type with the same template signature. In this case, we will call
 * `get_the_type<type_list, type_list<integral_constant<int i>, VT>>` and we
 * will get the same type `type_list<integral_constant<int i>, VT>>` as `Type`
 * and we specialize actual_type as the following
 * ```
 * template <int i, typename VT>
 * struct actual_type<type_list<integral_constant<int, i>, VT>> {
 *     using type = actual_class<i, VT>;
 * }
 * ```
 * Thus, from `actual_type<Type>::type`, we can get the actual_class with
 * desired templated parameters.
 */
template <typename T>
struct actual_type {
    using type = T;
};

template <gko::preconditioner::isai_type isai_value, typename... Rest>
struct actual_type<type_list<
    std::integral_constant<gko::preconditioner::isai_type, isai_value>,
    Rest...>> {
    using type = gko::preconditioner::Isai<isai_value, Rest...>;
};

/**
 * get_actual_type uses `actual_type<get_the_type<base, T>::type>::type` to
 * handle those classes with value template.
 *
 * @tparam base  the templated class
 * @tparam T  the templated parameters.
 */
template <template <typename...> class base, typename T>
struct get_actual_type {
    using type =
        typename actual_type<typename get_the_type<base, T>::type>::type;
};

/**
 * get_actual_factory_type uses `actual_type<get_the_type<base,
 * T>::type>::type::Factory` to does the same thing as `get_actual_type` but get
 * the Factory class.
 *
 * @tparam base  the templated class
 * @tparam T  the templated parameters.
 */
template <template <typename...> class base, typename T>
struct get_actual_factory_type {
    using type = typename get_actual_type<base, T>::type::Factory;
};


/**
 * tt_list has the same purpose as type_list, but we use separate type to
 * distinguish them in span
 *
 * @tparam ...Types  the types
 */
template <typename... Types>
struct tt_list {};


/**
 * ENABLE_SELECTION is to build a template selection on the given tt_list. It
 * will take each item (single type or type_list) of tt_list and the
 * corresponding identifier string. If the string is accepted by the Predicate,
 * it will launch the function with the accepted type.
 *
 * @param _name  the selection function name
 * @param _callable  the function to launch
 * @param _return  the return type of the function (pointer)
 * @param _get_type  the method to get the type (get_actual_type or
 *                   get_actual_factory_type)
 */
#define ENABLE_SELECTION(_name, _callable, _return, _get_type)                 \
    template <template <typename...> class Base, typename Predicate,           \
              typename... InferredArgs>                                        \
    _return _name(tt_list<>, Predicate is_eligible, rapidjson::Value &item,    \
                  InferredArgs... args)                                        \
    {                                                                          \
        GKO_KERNEL_NOT_FOUND;                                                  \
        return nullptr;                                                        \
    }                                                                          \
                                                                               \
    template <template <typename...> class Base, typename K, typename... Rest, \
              typename Predicate, typename... InferredArgs>                    \
    _return _name(tt_list<K, Rest...>, Predicate is_eligible,                  \
                  rapidjson::Value &item, InferredArgs... args)                \
    {                                                                          \
        auto key = get_string(K{});                                            \
        if (is_eligible(key)) {                                                \
            return _callable<typename _get_type<Base, K>::type>(               \
                item, std::forward<InferredArgs>(args)...);                    \
        } else {                                                               \
            return _name<Base>(tt_list<Rest...>(), is_eligible, item,          \
                               std::forward<InferredArgs>(args)...);           \
        }                                                                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


const std::string default_valuetype{get_string<gko::default_precision>()};
const std::string default_indextype{get_string<gko::int32>()};


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
 * span is to build a tt_list from each combination of each tt_list (or one
 * type). The result from span<tt_list<T1, T2, ...>, tt_list<K1, K2, ...>> is
 * tt_list<type_list<T1, K1>, type_list<T1, K2>, ..., type_list<T2, K1>,
 * type_list<T2, K2>, ...>
 *
 * @tparam K  the type or tt_list
 * @tparam T  the type or tt_list
 */
template <typename K, typename T, typename = void>
struct span {
    using type = tt_list<typename concat<K, T>::type>;
};

template <typename K, typename T>
struct span<K, tt_list<T>,
            typename std::enable_if<!is_tt_list<K>::value>::type> {
    using type = tt_list<typename concat<K, T>::type>;
};

template <typename K, typename T, typename... TT>
struct span<K, tt_list<T, TT...>,
            typename std::enable_if<!is_tt_list<K>::value>::type> {
    using type =
        typename concatenate<typename span<K, T>::type,
                             typename span<K, tt_list<TT...>>::type>::type;
};

template <typename K, typename T>
struct span<tt_list<K>, T> {
    using type = typename span<K, T>::type;
};

template <typename K, typename... K1, typename T>
struct span<tt_list<K, K1...>, T> {
    using type =
        typename concatenate<typename span<K, T>::type,
                             typename span<tt_list<K1...>, T>::type>::type;
};


/**
 * span_list is a extension for span. It can span varadic template parameters.
 * The result has the same rules as span, which expand the last argument first.
 *
 * @tparam K  the type or tt_list
 * @tparam ...T  the rest types or tt_lists
 */
template <typename K, typename... T>
struct span_list {};

template <typename K, typename T>
struct span_list<K, T> {
    using type = typename span<K, T>::type;
};

template <typename K, typename T, typename... S>
struct span_list<K, T, S...> {
    using type = typename span_list<typename span<K, T>::type, S...>::type;
};


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko

#endif  // GKOEXT_RESOURCE_MANAGER_BASE_ELEMENT_TYPES_HPP_
