// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_AMP_TYPES_HPP_
#define GKO_PUBLIC_CORE_BASE_AMP_TYPES_HPP_

#include <tuple>

#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace amp {


#if GINKGO_ENABLE_BFLOAT16 || GINKGO_ENABLE_HALF
#define GINKGO_HAVE_AMP_HALF 1
#if GINKGO_ENABLE_HALF
using half = gko::half;
#else
using half = gko::bfloat16;
#endif
using supported_precisions = std::tuple<double, float, half>;
#else
using supported_precisions = std::tuple<double, float>;
#endif

template <typename ValueType>
struct supported_types {
    using type = supported_precisions;
};

template <typename ValueType>
struct supported_types<std::complex<ValueType>> {
    using type = gko::to_complex<supported_precisions>;
};

constexpr int num_amp_precisions = std::tuple_size<supported_precisions>::value;

template <int i>
using type_at_idx = typename std::tuple_element<i, supported_precisions>::type;


namespace detail {


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


}  // namespace amp
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_AMP_TYPES_HPP_
