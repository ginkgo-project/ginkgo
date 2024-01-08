// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_SYNTHESIZER_IMPLEMENTATION_SELECTION_HPP_
#define GKO_DPCPP_SYNTHESIZER_IMPLEMENTATION_SELECTION_HPP_


#include <utility>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


#include "dpcpp/base/config.hpp"


namespace gko {
namespace syn {


#define GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(_name, _callable)         \
    template <typename Predicate, bool... BoolArgs, int... IntArgs,          \
              gko::size_type... SizeTArgs, typename... TArgs,                \
              typename... InferredArgs>                                      \
    inline void _name(::gko::syn::value_list<std::uint32_t>, Predicate,      \
                      ::gko::syn::value_list<bool, BoolArgs...>,             \
                      ::gko::syn::value_list<int, IntArgs...>,               \
                      ::gko::syn::value_list<gko::size_type, SizeTArgs...>,  \
                      ::gko::syn::type_list<TArgs...>, InferredArgs&&...)    \
        GKO_KERNEL_NOT_FOUND;                                                \
                                                                             \
    template <std::uint32_t K, std::uint32_t... Rest, typename Predicate,    \
              bool... BoolArgs, int... IntArgs, gko::size_type... SizeTArgs, \
              typename... TArgs, typename... InferredArgs>                   \
    inline void _name(                                                       \
        ::gko::syn::value_list<std::uint32_t, K, Rest...>,                   \
        Predicate is_eligible,                                               \
        ::gko::syn::value_list<bool, BoolArgs...> bool_args,                 \
        ::gko::syn::value_list<int, IntArgs...> int_args,                    \
        ::gko::syn::value_list<gko::size_type, SizeTArgs...> size_args,      \
        ::gko::syn::type_list<TArgs...> type_args, InferredArgs&&... args)   \
    {                                                                        \
        if (is_eligible(K)) {                                                \
            _callable<BoolArgs..., IntArgs..., SizeTArgs..., TArgs..., K>(   \
                std::forward<InferredArgs>(args)...);                        \
        } else {                                                             \
            _name(::gko::syn::value_list<std::uint32_t, Rest...>(),          \
                  is_eligible, bool_args, int_args, size_args, type_args,    \
                  std::forward<InferredArgs>(args)...);                      \
        }                                                                    \
    }


#define GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TYPE(_name, _callable)    \
    template <typename Predicate, bool... BoolArgs, int... IntArgs,          \
              gko::size_type... SizeTArgs, typename... TArgs,                \
              typename... InferredArgs>                                      \
    inline void _name(::gko::syn::type_list<>, Predicate,                    \
                      ::gko::syn::value_list<bool, BoolArgs...>,             \
                      ::gko::syn::value_list<int, IntArgs...>,               \
                      ::gko::syn::value_list<gko::size_type, SizeTArgs...>,  \
                      ::gko::syn::type_list<TArgs...>, InferredArgs&&...)    \
        GKO_KERNEL_NOT_FOUND;                                                \
                                                                             \
    template <typename K, typename... Rest, typename Predicate,              \
              bool... BoolArgs, int... IntArgs, gko::size_type... SizeTArgs, \
              typename... TArgs, typename... InferredArgs>                   \
    inline void _name(                                                       \
        ::gko::syn::type_list<K, Rest...>, Predicate is_eligible,            \
        ::gko::syn::value_list<bool, BoolArgs...> bool_args,                 \
        ::gko::syn::value_list<int, IntArgs...> int_args,                    \
        ::gko::syn::value_list<gko::size_type, SizeTArgs...> size_args,      \
        ::gko::syn::type_list<TArgs...> type_args, InferredArgs&&... args)   \
    {                                                                        \
        if (is_eligible(K())) {                                              \
            _callable<BoolArgs..., IntArgs..., SizeTArgs..., TArgs..., K>(   \
                std::forward<InferredArgs>(args)...);                        \
        } else {                                                             \
            _name(::gko::syn::type_list<Rest...>(), is_eligible, bool_args,  \
                  int_args, size_args, type_args,                            \
                  std::forward<InferredArgs>(args)...);                      \
        }                                                                    \
    }


#define GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(_name, _callable,  \
                                                          _dcfg)             \
    template <typename Predicate, bool... BoolArgs, int... IntArgs,          \
              gko::size_type... SizeTArgs, typename... TArgs,                \
              typename... InferredArgs>                                      \
    inline void _name(::gko::syn::value_list<std::uint32_t>, Predicate,      \
                      ::gko::syn::value_list<bool, BoolArgs...>,             \
                      ::gko::syn::value_list<int, IntArgs...>,               \
                      ::gko::syn::value_list<gko::size_type, SizeTArgs...>,  \
                      ::gko::syn::type_list<TArgs...>, InferredArgs&&...)    \
        GKO_KERNEL_NOT_FOUND;                                                \
                                                                             \
    template <std::uint32_t K, std::uint32_t... Rest, typename Predicate,    \
              bool... BoolArgs, int... IntArgs, gko::size_type... SizeTArgs, \
              typename... TArgs, typename... InferredArgs>                   \
    inline void _name(                                                       \
        ::gko::syn::value_list<std::uint32_t, K, Rest...>,                   \
        Predicate is_eligible,                                               \
        ::gko::syn::value_list<bool, BoolArgs...> bool_args,                 \
        ::gko::syn::value_list<int, IntArgs...> int_args,                    \
        ::gko::syn::value_list<gko::size_type, SizeTArgs...> size_args,      \
        ::gko::syn::type_list<TArgs...> type_args, InferredArgs&&... args)   \
    {                                                                        \
        if (is_eligible(K)) {                                                \
            _callable<BoolArgs..., IntArgs..., SizeTArgs..., TArgs...,       \
                      ::gko::kernels::dpcpp::device_config<                  \
                          _dcfg::decode<0>(K), _dcfg::decode<1>(K)>>(        \
                std::forward<InferredArgs>(args)...);                        \
        } else {                                                             \
            _name(::gko::syn::value_list<std::uint32_t, Rest...>(),          \
                  is_eligible, bool_args, int_args, size_args, type_args,    \
                  std::forward<InferredArgs>(args)...);                      \
        }                                                                    \
    }


#define GKO_ENABLE_IMPLEMENTATION_TWO_SELECTION_KERNEL(_name, _callable)      \
    template <typename Predicate, bool... BoolArgs, int... IntArgs,           \
              gko::size_type... SizeTArgs, typename... TArgs,                 \
              typename DeviceConfig, typename... InferredArgs>                \
    inline void _name(::gko::syn::value_list<int>, Predicate, DeviceConfig,   \
                      ::gko::syn::value_list<bool, BoolArgs...>,              \
                      ::gko::syn::value_list<int, IntArgs...>,                \
                      ::gko::syn::value_list<gko::size_type, SizeTArgs...>,   \
                      ::gko::syn::type_list<TArgs...>, InferredArgs&&...)     \
        GKO_KERNEL_NOT_FOUND;                                                 \
                                                                              \
    template <int K, int... Rest, typename Predicate, bool... BoolArgs,       \
              int... IntArgs, gko::size_type... SizeTArgs, typename... TArgs, \
              typename DeviceConfig, typename... InferredArgs>                \
    inline void _name(                                                        \
        ::gko::syn::value_list<int, K, Rest...>, Predicate is_eligible,       \
        DeviceConfig device_args,                                             \
        ::gko::syn::value_list<bool, BoolArgs...> bool_args,                  \
        ::gko::syn::value_list<int, IntArgs...> int_args,                     \
        ::gko::syn::value_list<gko::size_type, SizeTArgs...> size_args,       \
        ::gko::syn::type_list<TArgs...> type_args, InferredArgs&&... args)    \
    {                                                                         \
        if (is_eligible(K)) {                                                 \
            _callable<BoolArgs..., IntArgs..., SizeTArgs..., TArgs..., K,     \
                      DeviceConfig>(::gko::syn::value_list<int, K>(),         \
                                    std::forward<InferredArgs>(args)...);     \
        } else {                                                              \
            _name(::gko::syn::value_list<int, Rest...>(), is_eligible,        \
                  device_args, bool_args, int_args, size_args, type_args,     \
                  std::forward<InferredArgs>(args)...);                       \
        }                                                                     \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_ENABLE_IMPLEMENTATION_TWO_SELECTION_CONFIG(_name, _callable)       \
    template <typename Predicate, int... KernelArgs, typename KernelPredicate, \
              bool... BoolArgs, int... IntArgs, gko::size_type... SizeTArgs,   \
              typename... TArgs, typename... InferredArgs>                     \
    inline void _name(::gko::syn::value_list<std::uint32_t>, Predicate,        \
                      ::gko::syn::value_list<int, KernelArgs...>,              \
                      KernelPredicate,                                         \
                      ::gko::syn::value_list<bool, BoolArgs...>,               \
                      ::gko::syn::value_list<int, IntArgs...>,                 \
                      ::gko::syn::value_list<gko::size_type, SizeTArgs...>,    \
                      ::gko::syn::type_list<TArgs...>, InferredArgs&&...)      \
        GKO_KERNEL_NOT_FOUND;                                                  \
                                                                               \
    template <std::uint32_t K, std::uint32_t... Rest, typename Predicate,      \
              int... KernelArgs, typename KernelPredicate, bool... BoolArgs,   \
              int... IntArgs, gko::size_type... SizeTArgs, typename... TArgs,  \
              typename... InferredArgs>                                        \
    inline void _name(                                                         \
        ::gko::syn::value_list<std::uint32_t, K, Rest...>,                     \
        Predicate is_eligible,                                                 \
        ::gko::syn::value_list<int, KernelArgs...> kernel_args,                \
        KernelPredicate kernel_is_eligible,                                    \
        ::gko::syn::value_list<bool, BoolArgs...> bool_args,                   \
        ::gko::syn::value_list<int, IntArgs...> int_args,                      \
        ::gko::syn::value_list<gko::size_type, SizeTArgs...> size_args,        \
        ::gko::syn::type_list<TArgs...> type_args, InferredArgs&&... args)     \
    {                                                                          \
        if (is_eligible(K)) {                                                  \
            _callable(kernel_args, kernel_is_eligible,                         \
                      ::gko::kernels::dpcpp::device_config<K, 32>(),           \
                      bool_args, int_args, size_args, type_args,               \
                      std::forward<InferredArgs>(args)...);                    \
        } else {                                                               \
            _name(::gko::syn::value_list<std::uint32_t, Rest...>(),            \
                  is_eligible, kernel_args, kernel_is_eligible, bool_args,     \
                  int_args, size_args, type_args,                              \
                  std::forward<InferredArgs>(args)...);                        \
        }                                                                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_ENABLE_IMPLEMENTATION_TWO_SELECTION_CONFIG_TYPE(_name, _callable)  \
    template <typename Predicate, int... KernelArgs, typename KernelPredicate, \
              bool... BoolArgs, int... IntArgs, gko::size_type... SizeTArgs,   \
              typename... TArgs, typename... InferredArgs>                     \
    inline void _name(::gko::syn::type_list<>, Predicate,                      \
                      ::gko::syn::value_list<int, KernelArgs...>,              \
                      KernelPredicate,                                         \
                      ::gko::syn::value_list<bool, BoolArgs...>,               \
                      ::gko::syn::value_list<int, IntArgs...>,                 \
                      ::gko::syn::value_list<gko::size_type, SizeTArgs...>,    \
                      ::gko::syn::type_list<TArgs...>, InferredArgs&&...)      \
        GKO_KERNEL_NOT_FOUND;                                                  \
                                                                               \
    template <typename K, typename... Rest, typename Predicate,                \
              int... KernelArgs, typename KernelPredicate, bool... BoolArgs,   \
              int... IntArgs, gko::size_type... SizeTArgs, typename... TArgs,  \
              typename... InferredArgs>                                        \
    inline void _name(                                                         \
        ::gko::syn::type_list<K, Rest...>, Predicate is_eligible,              \
        ::gko::syn::value_list<int, KernelArgs...> kernel_args,                \
        KernelPredicate kernel_is_eligible,                                    \
        ::gko::syn::value_list<bool, BoolArgs...> bool_args,                   \
        ::gko::syn::value_list<int, IntArgs...> int_args,                      \
        ::gko::syn::value_list<gko::size_type, SizeTArgs...> size_args,        \
        ::gko::syn::type_list<TArgs...> type_args, InferredArgs&&... args)     \
    {                                                                          \
        if (is_eligible(K())) {                                                \
            _callable(kernel_args, kernel_is_eligible, K(), bool_args,         \
                      int_args, size_args, type_args,                          \
                      std::forward<InferredArgs>(args)...);                    \
        } else {                                                               \
            _name(::gko::syn::type_list<Rest...>(), is_eligible, kernel_args,  \
                  kernel_is_eligible, bool_args, int_args, size_args,          \
                  type_args, std::forward<InferredArgs>(args)...);             \
        }                                                                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_ENABLE_IMPLEMENTATION_TWO_SELECTION(_name, _callable)           \
    GKO_ENABLE_IMPLEMENTATION_TWO_SELECTION_KERNEL(_name##_tmp, _callable); \
    GKO_ENABLE_IMPLEMENTATION_TWO_SELECTION_CONFIG_TYPE(_name, _name##_tmp)


}  // namespace syn
}  // namespace gko

#endif  // GKO_DPCPP_SYNTHESIZER_IMPLEMENTATION_SELECTION_HPP_
