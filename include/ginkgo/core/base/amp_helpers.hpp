// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_AMP_HELPERS_HPP_
#define GKO_PUBLIC_CORE_BASE_AMP_HELPERS_HPP_

#include <tuple>

#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace amp {


// static constexpr int num_precisions
//         = std::tuple_size_v<gko::detail::supported_precisions>;


#if GINKGO_ENABLE_BFLOAT16 || GINKGO_ENABLE_HALF
#define GINKGO_HAVE_AMP_HALF 1
#if GINKGO_ENABLE_HALF
using amphalf = half;
#else
using amphalf = bfloat16;
#endif
using supported_precisions = std::tuple<double, float, amphalf>;
#else
using supported_precisions = std::tuple<double, float>;
#endif

constexpr int num_amp_precisions = std::tuple_size<supported_precisions>::value;


namespace detail {


template <bool cond, int val>
struct enable_if_v {};

template <int val>
struct enable_if_v<true, val> {
    static constexpr int value = val;
};


template <int i>
using type_at_idx = typename std::tuple_element<i, supported_precisions>::type;

template <typename ValueType, int i, typename Enable = void>
struct prec_idx_helper {
    using type = typename prec_idx_helper<ValueType, i - 1>::type;
    static constexpr int index = prec_idx_helper<ValueType, i - 1>::index;
};

template <typename ValueType, int i>
struct prec_idx_helper<
    ValueType, i,
    std::enable_if_t<std::is_same<ValueType, type_at_idx<i>>::value>> {
    using type = ValueType;
    static constexpr int index = i;
};

template <typename ValueType, int i>
struct prec_idx_helper<ValueType, i, std::enable_if_t<(i < 0)>> {};


}  // namespace detail


template <typename ValueType>
struct precision_index {
    static constexpr int index =
        detail::prec_idx_helper<ValueType, num_amp_precisions - 1>::index;
};

// template<>
// struct precision_index<double> {
//     static constexpr int position = 0;
// };

// template<>
// struct precision_index<float> {
//     static constexpr int position = 1;
// };

// #if GINKGO_HAVE_AMP_HALF

// template<>
// struct precision_index<amphalf> {
//     static constexpr int position = 2;
// };

// #endif


}  // namespace amp
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_AMP_HELPERS_HPP_
