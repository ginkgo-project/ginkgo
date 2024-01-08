// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SYNTHESIZER_IMPLEMENTATION_SELECTION_HPP_
#define GKO_CORE_SYNTHESIZER_IMPLEMENTATION_SELECTION_HPP_


#include <utility>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


namespace gko {
namespace syn {


#define GKO_ENABLE_IMPLEMENTATION_SELECTION(_name, _callable)                \
    template <typename Predicate, int... IntArgs, typename... TArgs,         \
              typename... InferredArgs>                                      \
    inline void _name(::gko::syn::value_list<int>, Predicate,                \
                      ::gko::syn::value_list<int, IntArgs...>,               \
                      ::gko::syn::type_list<TArgs...>, InferredArgs&&...)    \
        GKO_KERNEL_NOT_FOUND;                                                \
                                                                             \
    template <int K, int... Rest, typename Predicate, int... IntArgs,        \
              typename... TArgs, typename... InferredArgs>                   \
    inline void _name(                                                       \
        ::gko::syn::value_list<int, K, Rest...>, Predicate is_eligible,      \
        ::gko::syn::value_list<int, IntArgs...> int_args,                    \
        ::gko::syn::type_list<TArgs...> type_args, InferredArgs&&... args)   \
    {                                                                        \
        if (is_eligible(K)) {                                                \
            _callable<IntArgs..., TArgs...>(                                 \
                ::gko::syn::value_list<int, K>(),                            \
                std::forward<InferredArgs>(args)...);                        \
        } else {                                                             \
            _name(::gko::syn::value_list<int, Rest...>(), is_eligible,       \
                  int_args, type_args, std::forward<InferredArgs>(args)...); \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


}  // namespace syn
}  // namespace gko


#endif  // GKO_CORE_SYNTHESIZER_IMPLEMENTATION_SELECTION_HPP_
