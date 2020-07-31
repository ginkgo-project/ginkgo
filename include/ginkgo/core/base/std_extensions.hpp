/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_STD_EXTENSIONS_HPP_
#define GKO_CORE_BASE_STD_EXTENSIONS_HPP_


#include <functional>
#include <memory>
#include <type_traits>


// This header provides implementations of useful utilities introduced into the
// C++ standard after C++14 (e.g. C++17 and C++20).
// For documentation about these utilities refer to the newer version of the
// standard.


namespace gko {
/**
 * @brief The namespace for functionalities after C++11 standard.
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
constexpr bool greater(const T &&lhs, const T &&rhs)
{
    return std::greater<void>()(lhs, rhs);
}


// Kept for backward compatibility.
template <typename T>
constexpr bool greater_equal(const T &&lhs, const T &&rhs)
{
    return std::greater_equal<void>()(lhs, rhs);
}


// Kept for backward compatibility.
template <typename T>
constexpr bool less(const T &&lhs, const T &&rhs)
{
    return std::less<void>()(lhs, rhs);
}


// Kept for backward compatibility.
template <typename T>
constexpr bool less_equal(const T &&lhs, const T &&rhs)
{
    return std::less_equal<void>()(lhs, rhs);
}


}  // namespace xstd
}  // namespace gko


#endif  // GKO_CORE_BASE_STD_EXTENSIONS_HPP_
