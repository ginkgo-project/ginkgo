/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

    friend GKO_ACC_ATTRIBUTES constexpr bool operator<(const index_span &first,
                                                       const index_span &second)
    {
        return first.end < second.begin;
    }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator<=(
        const index_span &first, const index_span &second)
    {
        return first.end <= second.begin;
    }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator>(const index_span &first,
                                                       const index_span &second)
    {
        return second < first;
    }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator>=(
        const index_span &first, const index_span &second)
    {
        return second <= first;
    }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator==(
        const index_span &first, const index_span &second)
    {
        return first.begin == second.begin && first.end == second.end;
    }

    friend GKO_ACC_ATTRIBUTES constexpr bool operator!=(
        const index_span &first, const index_span &second)
    {
        return !(first == second);
    }

    const size_type begin;
    const size_type end;
};


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_INDEX_SPAN_HPP_
