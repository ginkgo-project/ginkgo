// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_STD_EXTENSIONS_HPP_
#define GKO_PUBLIC_CORE_BASE_STD_EXTENSIONS_HPP_


#include <exception>
#include <functional>
#include <memory>
#include <type_traits>

#include "ginkgo/core/base/types.hpp"


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


/**
 * Use the custom implementation, since the std::void_t used in
 * is_matrix_type_builder seems to trigger a compiler bug in GCC 7.5.
 */
template <typename... Ts>
using void_t = typename detail::make_void<Ts...>::type;


GKO_DEPRECATED("use std::uncaught_exceptions")
inline bool uncaught_exception() noexcept
{
    return std::uncaught_exceptions() > 0;
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


template <class... Ts>
using conjunction = std::conjunction<Ts...>;


// Provide the type_identity from C++20
template <typename T>
struct type_identity {
    using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;


}  // namespace xstd
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_STD_EXTENSIONS_HPP_
