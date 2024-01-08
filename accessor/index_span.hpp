// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_INDEX_SPAN_HPP_
#define GKO_ACCESSOR_INDEX_SPAN_HPP_

#include "utils.hpp"


namespace gko {
namespace acc {


/**
 * An index_span is a lightweight structure used to describe a contiguous span
 * of indices of one dimension.
 *
 * The main purpose of the index_span is to create sub-ranges from other ranges.
 * An index_span `is` represents a contiguous set of indexes in one dimension of
 * the range, starting from `is.begin` (inclusive) and ending at index `is.end`
 * (exclusive). An index_span is only valid if its end is larger than its
 * beginning.
 *
 * index_spans can be compared using `==` and `!=` operators. Two spans are
 * equal iff both their `begin` and `end` values are identical.
 *
 * index_sets also have two distinct partial orders defined:
 * 1. `x < y` (`y > x`) iff `x.end < y.begin`
 * 2. `x <= y` (`y >= x`) iff `x.end <= y.begin`
 * Note: `x < y || x == y` is not equivalent to `x <= y`.
 */
struct index_span {
    /**
     * Creates an index_span.
     *
     * @param begin  the beginning (inclusive) of the index_span
     * @param end  the end (exclusive) of the index_span
     *
     */
    GKO_ACC_ATTRIBUTES constexpr index_span(size_type begin,
                                            size_type end) noexcept
        : begin{begin}, end{end}
    {}

    /**
     * Creates an index_span representing the point `point`.
     *
     * The begin is set to `point`, and the end to `point + 1`
     *
     * @param point  the point which the index_span represents
     */
    GKO_ACC_ATTRIBUTES constexpr index_span(size_type point) noexcept
        : index_span{point, point + 1}
    {}

    /**
     * Checks if an index_span is valid.
     *
     * @returns true iff `this->begin < this->end`
     */
    GKO_ACC_ATTRIBUTES constexpr bool is_valid() const { return begin < end; }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator<(const index_span& first,
                                                       const index_span& second)
    {
        return first.end < second.begin;
    }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator<=(
        const index_span& first, const index_span& second)
    {
        return first.end <= second.begin;
    }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator>(const index_span& first,
                                                       const index_span& second)
    {
        return second < first;
    }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator>=(
        const index_span& first, const index_span& second)
    {
        return second <= first;
    }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator==(
        const index_span& first, const index_span& second)
    {
        return first.begin == second.begin && first.end == second.end;
    }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator!=(
        const index_span& first, const index_span& second)
    {
        return !(first == second);
    }

    const size_type begin;
    const size_type end;
};


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_INDEX_SPAN_HPP_
