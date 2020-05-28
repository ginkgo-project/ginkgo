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

#ifndef GKO_CORE_SYNTHESIZER_CONTAINERS_HPP_
#define GKO_CORE_SYNTHESIZER_CONTAINERS_HPP_


#include <ginkgo/core/base/std_extensions.hpp>


namespace gko {
/**
 * @brief The Synthesizer namespace.
 *
 * @ingroup syn
 */
namespace syn {


template <typename T, T... Values>
struct value_list {};


template <typename... Types>
struct type_list {};


template <int Start, int End, int Step = 1>
struct range {};


namespace detail {


template <typename List1, typename List2>
struct concatenate_impl;

template <typename T, T... Values1, T... Values2>
struct concatenate_impl<value_list<T, Values1...>, value_list<T, Values2...>> {
    using type = value_list<T, Values1..., Values2...>;
};


}  // namespace detail


template <typename List1, typename List2>
using concatenate = typename detail::concatenate_impl<List1, List2>::type;


namespace detail {


template <typename T, typename = void>
struct as_list_impl;

template <typename T, T... Values>
struct as_list_impl<value_list<T, Values...>> {
    using type = value_list<T, Values...>;
};

template <typename... Types>
struct as_list_impl<type_list<Types...>> {
    using type = type_list<Types...>;
};

template <int Start, int End, int Step>
struct as_list_impl<range<Start, End, Step>, xstd::enable_if_t<(Start < End)>> {
    using type = concatenate<
        value_list<int, Start>,
        typename as_list_impl<range<Start + Step, End, Step>>::type>;
};

template <int Start, int End, int Step>
struct as_list_impl<range<Start, End, Step>,
                    xstd::enable_if_t<(Start >= End)>> {
    using type = value_list<int>;
};


}  // namespace detail


template <typename T>
using as_list = typename detail::as_list_impl<T>::type;


}  // namespace syn
}  // namespace gko


#endif  // GKO_CORE_SYNTHESIZER_CONTAINERS_HPP_
