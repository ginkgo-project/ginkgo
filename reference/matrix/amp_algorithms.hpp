// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_MATRIX_AMP_ALGORITHMS_H_
#define GKO_REFERENCE_MATRIX_AMP_ALGORITHMS_H_

#include <ginkgo/core/base/amp_types.hpp>

#include "core/base/utils.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace amp {


template <typename T, typename U>
using precision_array = gko::amp::precision_array<T, U>;


template <typename RealType, int k>
inline void bins_precision_lower_bounds_impl(
    const RealType row_norm, const float tolerance,
    precision_array<float, RealType>& lbs)
{
    using narrow_types = typename gko::amp::narrow_types<RealType>::type;
    constexpr int q = gko::amp::narrow_types<RealType>::num_types;
    if constexpr (k > q - 1) {
        return;
    }
    if constexpr (k == q - 1) {
        lbs[k] = static_cast<float>(row_norm) * tolerance;
    } else {
        using next_type =
            typename std::tuple_element<k + 1, narrow_types>::type;
        lbs[k] = static_cast<float>(tolerance * row_norm /
                                    std::numeric_limits<next_type>::epsilon());
        bins_precision_lower_bounds_impl<RealType, k + 1>(row_norm, tolerance,
                                                          lbs);
    }
}

/**
 * Get the lower bounds for all available precision bins based only on the
 * machine epsilon.
 *
 * @tparam RealType  Real scalar type that's the highest type being considered.
 *
 * @param row_norm  The sum of absolute values in a row of a matrix.
 * @param tolerance  Tolerance for SpMV.
 *                   `float` should be enough for this since we normally only
 *                   need an order of magnitude like 1e-14, and the exponent
 *                   range down to 1e-38 should also be sufficient.
 */
template <typename RealType>
inline auto get_bins_precision_lower_bounds(const RealType row_norm,
                                            const float tolerance)
{
    constexpr int q = gko::amp::narrow_types<RealType>::num_types;
    std::array<float, q> lbs;
    bins_precision_lower_bounds_impl<RealType, 0>(row_norm, tolerance, lbs);
    return lbs;
}

/**
 * Get the minimum representable values for all bins.
 *
 * @tparam RealType  The highest precision real type to be considered.
 */
template <typename RealType>
inline auto get_bins_min_representable()
{
    using narrow_types = typename gko::amp::narrow_types<RealType>::type;
    constexpr int q = gko::amp::narrow_types<RealType>::num_types;
    std::array<RealType, q> mins = {};
    // get_bins_min_representable_impl<RealType, q, 0>(mins);
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using bin_type = typename std::tuple_element<k, narrow_types>::type;
        mins[k] = static_cast<RealType>(std::numeric_limits<bin_type>::min());
    });
    return mins;
}

/**
 * Given the absolute value of a number, determines which precision bin it
 * should go to.
 *
 * @return The precision bin index that a number should go to.
 *         Returns -1 if the number should be dropped.
 */
template <typename RealType>
inline int get_precision_bin(
    const precision_array<float, RealType>& lower_bounds,
    const RealType abs_number, const int k)
{
    constexpr int q = gko::amp::narrow_types<RealType>::num_types;
    if (k >= q) {
        return -1;
    }
    if (abs_number > static_cast<RealType>(lower_bounds[k])) {
        return k;
    } else {
        return get_precision_bin<RealType>(lower_bounds, abs_number, k + 1);
    }
}

/**
 * Adjust bin assignment to avoid underflow.
 * If a value cannot be represented in its initially assigned bin
 * (below min representable), move to a higher precision bin.
 *
 * @param min_representable  The smallest value that can represented by the
 *                           different supported real scalar types without
 *                           underflow.
 * @param abs_number  Absolute value of the number to be binned.
 * @param ibin  The initial bin assigned to the number
 *              by @ref get_precision_bin.
 */
template <typename RealType>
inline int adjust_bin_for_underflow(
    const precision_array<RealType, RealType>& min_representable,
    const RealType abs_number, int ibin)
{
    constexpr int q = gko::amp::narrow_types<RealType>::num_types;
    if (ibin < 0) {
        return ibin;  // Already dropped
    }
    // Check if value can be represented in the assigned bin
    while (ibin > 0 && abs_number < min_representable[ibin]) {
        ibin--;  // Move to higher precision bin
    }
    return ibin;
}

/**
 * Get the appropriate precision bin for the given absolute value,
 * considering both precision and underflow.
 *
 * @tparam RealType  Highest precision real type.
 *
 * @param lower_bounds  Lower bound of each precision bin,
 *                      @see get_bins_precision_lower_bounds.
 * @param min_representable  Minimum value that can be represented in each
 *                           precision bin. @see get_bins_min_representable.
 * @param abs_number  Absolute value of the number to be classified into a bin.
 */
template <typename RealType>
inline int get_adjusted_bin(
    const precision_array<float, RealType>& lower_bounds,
    const precision_array<RealType, RealType>& min_representable,
    const RealType abs_number)
{
    int ibin = get_precision_bin<RealType>(lower_bounds, abs_number, 0);
    return adjust_bin_for_underflow<RealType>(min_representable, abs_number,
                                              ibin);
}

/**
 * Assigns a given value to the given index in a tuple.
 *
 * @tparam k  Position in the tuple to check against the runtime index.
 * @tparam ValueType  scalar type to assign to the tuple position.
 * @tparam Args  Types that make up the tuple.
 *
 * @param t  The tuple to be modified.
 * @param value  The value to be assigned.
 * @param idx  The runtime position of the tuple that should be assigned to.
 */
template <int k, typename ValueType, typename... Args>
void assign_value_to_tuple(std::tuple<Args...>& t, const ValueType& value,
                           const int idx)
{
    constexpr int len = sizeof...(Args);
    if constexpr (k < 0 || k >= len) {
        return;
    } else if constexpr (k == len - 1) {
        if (k == idx) {
            std::get<k>(t) = value;
        }
        return;
    } else {
        if (k == idx) {
            std::get<k>(t) = value;
        } else {
            assign_value_to_tuple<k + 1>(t, value, idx);
        }
    }
}

/**
 * Assigns a given value to the given location of the given index
 * in a tuple of arrays.
 *
 * @tparam k  The index of the tuple-element to check. Starts at 0.
 *
 * @param t  The tuple whose element is to be to assign to.
 * @param value  The value to be assigned.
 * @param t_idx  The index of the tuple-element to be modified.
 * @param loc  The offset at which the value should be placed.
 */
template <int k, typename ValueType, typename... Args>
inline void assign_value_to_array_tuple(const std::tuple<Args...>& t,
                                        const ValueType& value, const int t_idx,
                                        const int loc)
{
    constexpr int len = sizeof...(Args);
    if constexpr (k < 0 || k >= len) {
        return;
    } else if constexpr (k == len - 1) {
        if (k == t_idx) {
            std::get<k>(t)[loc] = value;
        }
        return;
    } else {
        if (k == t_idx) {
            std::get<k>(t)[loc] = value;
        } else {
            assign_value_to_array_tuple<k + 1>(t, value, t_idx, loc);
        }
    }
}


}  // namespace amp
}  // namespace reference
}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_MATRIX_AMP_ALGORITHMS_H_
