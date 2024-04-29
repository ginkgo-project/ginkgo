// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_INDEX_RANGE_HPP_
#define GKO_CORE_BASE_INDEX_RANGE_HPP_


#include <cassert>
#include <iterator>
#include <type_traits>


#include "core/base/iterator_range.hpp"


namespace gko {


/**
 * An index_iterator represents an iteration through an integer
 * range. Each increment of the iterator increments the integer it represents.
 *
 * @tparam IndexType  the type of the index, it must be a signed integer type.
 */
template <typename IndexType>
class index_iterator {
    static_assert(std::is_signed<IndexType>::value &&
                      std::is_integral<IndexType>::value,
                  "Can only use index_iterator with signed integral types!");

public:
    using index_type = IndexType;

    // iterator_traits requirements
    using iterator_category = std::random_access_iterator_tag;
    using value_type = index_type;
    using difference_type = index_type;
    using pointer = const index_type*;
    using reference = index_type;  // we shouldn't hand out references to
                                   // the index for lifetime safety

    /** Initializes the iterator to the given index. */
    constexpr explicit index_iterator(index_type i) : idx_{i} {}

    constexpr index_iterator() : index_iterator{0} {}

    // RandomAccessIterator requirements
    /** Iterator advance. */
    constexpr index_iterator& operator+=(difference_type n)
    {
        idx_ += n;
        return *this;
    }

    /** Dereference. */
    constexpr reference operator*() const { return idx_; }

    /** Iterator difference. */
    constexpr friend difference_type operator-(index_iterator a,
                                               index_iterator b)
    {
        return *a - *b;
    }

    // InputIterator requirements
    /** Equality. */
    constexpr friend bool operator==(index_iterator a, index_iterator b)
    {
        return a - b == 0;
    }

    /** non-equality */
    constexpr friend bool operator!=(index_iterator a, index_iterator b)
    {
        return !(a == b);
    }

    // BidirectionalIterator requirements
    /** Pre-increment. */
    constexpr index_iterator& operator++() { return *this += 1; }

    /** Post-increment. */
    constexpr index_iterator operator++(int)
    {
        auto tmp{*this};
        operator++();
        return tmp;
    }

    /** Pre-decrement. */
    constexpr index_iterator& operator--() { return *this -= 1; }

    /** Post-decrement. */
    constexpr index_iterator operator--(int)
    {
        auto tmp{*this};
        operator--();
        return tmp;
    }

    // RandomAccessIterator requirements
    /** Iterator advance. */
    constexpr friend index_iterator operator+(index_iterator a,
                                              difference_type n)
    {
        return a += n;
    }

    /** reverse advance */
    constexpr friend index_iterator operator+(difference_type n,
                                              index_iterator a)
    {
        return a + n;
    }

    /** Iterator backwards advance. */
    constexpr index_iterator& operator-=(difference_type n)
    {
        return *this += -n;
    }

    /** Iterator backwards advance. */
    constexpr friend index_iterator operator-(index_iterator a,
                                              difference_type n)
    {
        return a + -n;
    }

    /** Subscript. */
    constexpr reference operator[](difference_type n) const
    {
        return *(*this + n);
    }

    /** less than */
    constexpr friend bool operator<(index_iterator a, index_iterator b)
    {
        return b - a > 0;
    }

    /** greater than */
    constexpr friend bool operator>(index_iterator a, index_iterator b)
    {
        return b < a;
    }

    /** greater equal */
    constexpr friend bool operator>=(index_iterator a, index_iterator b)
    {
        return !(a < b);
    }

    /** less equal */
    constexpr friend bool operator<=(index_iterator a, index_iterator b)
    {
        return !(a > b);
    }

private:
    index_type idx_;
};


/**
 * Represents an index range that will be iterated through in order.
 * The following two loops are equivalent (assuming begin <= end):
 * ```cpp
 * for (int i = begin; i < end; i++);
 * for (auto i : irange<int>{begin, end});
 * ```
 *
 * @tparam IndexType  the type of the indices.
 */
template <typename IndexType>
class irange : public iterator_range<index_iterator<IndexType>> {
public:
    using index_type = IndexType;
    using iterator = index_iterator<index_type>;

    /**
     * Constructs an index range consisting of the values in [begin, end).
     *
     * @param begin  the first index in the range
     * @param end  one past the last index in the range. It must not be smaller
     *             than `begin`.
     */
    constexpr explicit irange(index_type begin, index_type end)
        : iterator_range<iterator>{iterator{begin}, iterator{end}}
    {
        assert(begin <= end);
    }

    /**
     * Constructs an index range consisting of the values in [0, end).
     *
     * @param end  one past the last index in the range. It must not be a
     *             negative number.
     */
    constexpr explicit irange(index_type end) : irange{0, end} {}

    /** @return the first index in this range. */
    constexpr index_type begin_index() const { return *this->begin(); }

    /** @return one past the last index in this range. */
    constexpr index_type end_index() const { return *this->end(); }

    /**
     * Compares two ranges for equality.
     *
     * @param lhs  the first range
     * @param rhs  the second range
     * @return true iff both ranges have the same begin and end index.
     */
    constexpr friend bool operator==(irange lhs, irange rhs)
    {
        return lhs.begin() == rhs.begin() && lhs.end() == rhs.end();
    }

    /**
     * Compares two ranges for inequality.
     *
     * @param lhs  the first range
     * @param rhs  the second range
     * @return false iff both ranges have the same begin and end index.
     */
    constexpr friend bool operator!=(irange lhs, irange rhs)
    {
        return !(lhs == rhs);
    }
};


}  // namespace gko


#endif  // GKO_CORE_BASE_INDEX_RANGE_HPP_
