// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_STD_EXTENSIONS_HPP_
#define GKO_PUBLIC_CORE_BASE_STD_EXTENSIONS_HPP_


#include <exception>
#include <functional>
#include <memory>
#include <type_traits>


// This header provides implementations of useful utilities introduced into the
// C++ standard after C++14 (e.g. C++17 and C++20).
// For documentation about these utilities refer to the newer version of the
// standard.


namespace gko {
/**
 * @brief The namespace for functionalities after C++14 standard.
 * @internal
 * @ingroup xstd
 */
namespace xstd {
namespace detail {


template <typename... Ts>
struct make_void {
    using type = void;
};


}  // namespace detail


// Added in C++17
template <typename... Ts>
using void_t = typename detail::make_void<Ts...>::type;


// Disable deprecation warnings when using standard > C++14
inline bool uncaught_exception() noexcept
{
// MSVC uses _MSVC_LANG as __cplusplus
#if (defined(_MSVC_LANG) && _MSVC_LANG > 201402L) || __cplusplus > 201402L
    return std::uncaught_exceptions() > 0;
#else
    return std::uncaught_exception();
#endif
}


// Kept for backward compatibility.
template <bool B, typename T = void>
using enable_if_t = std::enable_if_t<B, T>;


// Kept for backward compatibility.
template <bool B, typename T, typename F>
using conditional_t = std::conditional_t<B, T, F>;


// Kept for backward compatibility.
template <typename T>
using decay_t = std::decay_t<T>;


// Kept for backward compatibility.
template <typename T>
constexpr bool greater(const T&& lhs, const T&& rhs)
{
    return std::greater<void>()(lhs, rhs);
}


// Kept for backward compatibility.
template <typename T>
constexpr bool greater_equal(const T&& lhs, const T&& rhs)
{
    return std::greater_equal<void>()(lhs, rhs);
}


// Kept for backward compatibility.
template <typename T>
constexpr bool less(const T&& lhs, const T&& rhs)
{
    return std::less<void>()(lhs, rhs);
}


// Kept for backward compatibility.
template <typename T>
constexpr bool less_equal(const T&& lhs, const T&& rhs)
{
    return std::less_equal<void>()(lhs, rhs);
}


// available in <type_traits> with C++17
template <class...>
struct conjunction : std::true_type {};
template <class B1>
struct conjunction<B1> : B1 {};
template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};


}  // namespace xstd
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_STD_EXTENSIONS_HPP_
