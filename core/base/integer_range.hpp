// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_INTEGER_RANGE_HPP_
#define GKO_CORE_BASE_INTEGER_RANGE_HPP_


#include <cassert>
#include <iterator>
#include <type_traits>


namespace gko {


template <typename IndexType>
class irange {
    static_assert(std::is_signed<IndexType>::value &&
                      std::is_integral<IndexType>::value,
                  "Can only use irange with signed integral types!");

public:
    using index_type = IndexType;
    using value_type = index_type;

    class iterator {
    public:
        using index_type = IndexType;

        // iterator_traits requirements
        using iterator_category = std::random_access_iterator_tag;
        using value_type = index_type;
        using difference_type = index_type;
        using pointer = index_type*;
        using reference = index_type;  // we shouldn't hand out references to
                                       // the index for lifetime safety

        /** Initializes the iterator to the given index. */
        constexpr explicit iterator(IndexType i) : idx_{i} {}

        // Iterator requirements
        /** Initializes the iterator to 0. */
        constexpr iterator() : idx_{} {}

        // RandomAccessIterator requirements
        /** Iterator advance. */
        constexpr iterator& operator+=(difference_type n)
        {
            idx_ += n;
            return *this;
        }

        /** Iterator advance. */
        constexpr friend iterator operator+(iterator a, difference_type n)
        {
            return iterator{*a + n};
        }

        /** Pre-increment. */
        constexpr iterator& operator++() { return *this += 1; }

        /** Post-increment. */
        constexpr iterator operator++(int)
        {
            auto tmp{*this};
            operator++();
            return tmp;
        }

        // InputIterator requirements
        /** Equality. */
        constexpr friend bool operator==(iterator a, iterator b)
        {
            return (*a == *b);
        }

        /** Dereference. */
        constexpr reference operator*() const { return idx_; }

        // BidirectionalIterator requirements
        /** Pre-decrement. */
        constexpr iterator& operator--() { return *this -= 1; }

        /** Post-decrement. */
        constexpr iterator operator--(int)
        {
            auto tmp{*this};
            operator--();
            return tmp;
        }

        /** Iterator backwards advance. */
        constexpr iterator& operator-=(difference_type n)
        {
            idx_ += -n;
            return *this;
        }

        /** Iterator backwards advance. */
        constexpr friend iterator operator-(iterator a, difference_type n)
        {
            return a + -n;
        }

        /** Iterator difference. */
        constexpr friend difference_type operator-(iterator a, iterator b)
        {
            return *a - *b;
        }

        /** Subscript. */
        constexpr reference operator[](difference_type n) const
        {
            return *(*this + n);
        }

        // Boilerplate comparison operators
        /** non-equality */
        constexpr friend bool operator!=(iterator a, iterator b)
        {
            return !(a == b);
        }

        /** reverse advance */
        constexpr friend iterator operator+(difference_type n, iterator a)
        {
            return a + n;
        }

        /** less than */
        constexpr friend bool operator<(iterator a, iterator b)
        {
            return b - a > 0;
        }

        /** greater than */
        constexpr friend bool operator>(iterator a, iterator b)
        {
            return b < a;
        }

        /** greater equal */
        constexpr friend bool operator>=(iterator a, iterator b)
        {
            return !(a < b);
        }

        /** less equal */
        constexpr friend bool operator<=(iterator a, iterator b)
        {
            return !(a > b);
        }

    private:
        index_type idx_;
    };

    explicit irange(index_type begin, index_type end) : begin_{begin}, end_{end}
    {
        assert(end >= begin);
    }

    explicit irange(index_type end) : irange{0, end} {}

    constexpr index_type begin_index() const { return begin_; }

    constexpr index_type mid_index() const
    {
        return begin_index() + size() / 2;
    }

    constexpr index_type end_index() const { return end_; }

    constexpr index_type size() const { return end_index() - begin_index(); }

    constexpr bool empty() const { return size() == 0; }

    constexpr iterator begin() const { return iterator{begin_index()}; }

    constexpr iterator mid() const { return iterator{mid_index()}; }

    constexpr iterator end() const { return iterator{end_index()}; }

    constexpr irange lower_half() const { return irange{*begin(), *mid()}; }

    constexpr irange upper_half() const { return irange{*mid(), *end()}; }

    constexpr friend bool operator==(irange lhs, irange rhs)
    {
        return lhs.begin() == rhs.begin() && lhs.end() == rhs.end();
    }

    constexpr friend bool operator!=(irange lhs, irange rhs)
    {
        return !(lhs == rhs);
    }

private:
    index_type begin_;
    index_type end_;
};


template <typename IndexType>
class irange_strided {
public:
    using index_type = IndexType;

    // tag type to facilitate the i < end comparison
    struct iterator_sentinel {
        index_type end;
    };

    class iterator {
    public:
        // iterator_traits requirements
        using iterator_category = std::forward_iterator_tag;
        using value_type = index_type;
        using difference_type = index_type;
        using pointer = index_type*;
        using reference = index_type;  // we shouldn't hand out references to
                                       // the index for lifetime safety

        constexpr explicit iterator(index_type i, index_type stride)
            : idx_{i}, stride_{stride}
        {}

        constexpr iterator& operator++()
        {
            idx_ += stride_;
            return *this;
        }

        constexpr friend bool operator!=(iterator lhs, iterator_sentinel rhs)
        {
            return lhs.idx_ < rhs.end;
        }

        constexpr friend bool operator!=(iterator_sentinel lhs, iterator rhs)
        {
            return rhs != lhs;
        }

        constexpr friend bool operator==(iterator lhs, iterator_sentinel rhs)
        {
            return !(lhs != rhs);
        }

        constexpr friend bool operator==(iterator_sentinel lhs, iterator rhs)
        {
            return !(lhs != rhs);
        }

        constexpr const IndexType& operator*() const { return idx_; }

    private:
        index_type idx_;
        index_type stride_;
    };

    explicit constexpr irange_strided(index_type begin, index_type end,
                                      index_type stride)
        : begin_{begin}, end_{end}, stride_{stride}
    {
        assert(end >= begin);
        assert(stride > 0);
    }

    constexpr index_type begin_index() const { return begin_; }

    constexpr index_type end_index() const { return end_; }

    constexpr index_type stride() const { return stride_; }

    constexpr iterator begin() const { return iterator{begin_, stride_}; }

    constexpr iterator_sentinel end() const { return iterator_sentinel{end_}; }

private:
    index_type begin_;
    index_type end_;
    index_type stride_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_INTEGER_RANGE_HPP_
