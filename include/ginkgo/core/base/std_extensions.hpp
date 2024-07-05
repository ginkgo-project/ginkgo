// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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
template <typename... Ts>
using void_t = std::void_t<Ts...>;


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


}  // namespace xstd
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_STD_EXTENSIONS_HPP_
