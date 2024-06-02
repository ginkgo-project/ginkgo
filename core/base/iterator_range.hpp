// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_ITERATOR_RANGE_HPP_
#define GKO_CORE_BASE_ITERATOR_RANGE_HPP_


namespace gko {


/**
 * An iterator_range represents a pair of iterators to be used in a range-for
 * loop.
 *
 * @tparam Iterator  the iterator type
 */
template <typename Iterator>
class iterator_range {
public:
    using iterator = Iterator;

    /**
     * Constructs a range `[begin, end)` from its begin and end iterator.
     *
     * @param begin  points to the first element
     * @param end  points past the last element
     */
    constexpr explicit iterator_range(iterator begin, iterator end)
        : begin_{begin}, end_{end}
    {}

    /** @return the iterator pointing to the first element. */
    constexpr iterator begin() const { return begin_; }

    /** @return the iterator pointing past the last element. */
    constexpr iterator end() const { return end_; }

private:
    iterator begin_;
    iterator end_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_ITERATOR_RANGE_HPP_
