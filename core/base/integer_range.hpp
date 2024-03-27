// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_INTEGER_RANGE_HPP_
#define GKO_CORE_BASE_INTEGER_RANGE_HPP_


#include <cassert>
#include <iterator>
#include <type_traits>


namespace gko {


namespace detail {


template <typename Group>
struct group_traits {};


struct sycl_group_traits {
    template <typename Group>
    constexpr static std::size_t get_size(Group g)
    {
        return g.get_local_linear_range();
    }

    template <typename Group>
    constexpr static std::size_t get_count(Group g)
    {
        return g.get_group_linear_range();
    }

    template <typename Group>
    constexpr static std::size_t get_local_id(Group g)
    {
        return g.get_local_linear_id();
    }

    template <typename Group>
    constexpr static std::size_t get_group_id(Group g)
    {
        return g.get_group_linear_id();
    }
};


/*template <int dimension>
struct group_traits<sycl::h_item<dimension>> : sycl_group_traits {};


template <int dimension>
struct group_traits<sycl::subgroup<dimension>> : sycl_group_traits {};


template <int dimension>
struct group_traits<sycl::group<dimension>> : sycl_group_traits {};


template <int subwarp_size>
struct group_traits<cooperative_groups::tiled_partition<subwarp_size>>
    : cuda_group_traits {};*/


}  // namespace detail


template <typename IndexType>
class integer_iterator {
    static_assert(std::is_signed<IndexType>::value &&
                      std::is_integral<IndexType>::value,
                  "Can only use integer_iterator with signed integral types!");

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
    constexpr explicit integer_iterator(index_type i, index_type stride = 1)
        : idx_{i}, stride_{stride}
    {
        assert(stride > 0);
    }

    // Iterator requirements
    /** Initializes the iterator to 0. */
    constexpr integer_iterator() : idx_{} {}

    // RandomAccessIterator requirements
    /** Iterator advance. */
    constexpr integer_iterator& operator+=(difference_type n)
    {
        idx_ += n * stride_;
        return *this;
    }

    /** Iterator advance. */
    constexpr friend integer_iterator operator+(integer_iterator a,
                                                difference_type n)
    {
        return a += n;
    }

    /** Pre-increment. */
    constexpr integer_iterator& operator++() { return *this += 1; }

    /** Post-increment. */
    constexpr integer_iterator operator++(int)
    {
        auto tmp{*this};
        operator++();
        return tmp;
    }

    // InputIterator requirements
    /** Equality. */
    constexpr friend bool operator==(integer_iterator a, integer_iterator b)
    {
        return (*a == *b);
    }

    /** Dereference. */
    constexpr reference operator*() const { return idx_; }

    // BidirectionalIterator requirements
    /** Pre-decrement. */
    constexpr integer_iterator& operator--() { return *this -= 1; }

    /** Post-decrement. */
    constexpr integer_iterator operator--(int)
    {
        auto tmp{*this};
        operator--();
        return tmp;
    }

    /** Iterator backwards advance. */
    constexpr integer_iterator& operator-=(difference_type n)
    {
        idx_ += -n;
        return *this;
    }

    /** Iterator backwards advance. */
    constexpr friend integer_iterator operator-(integer_iterator a,
                                                difference_type n)
    {
        return a + -n;
    }

    /** Iterator difference. */
    constexpr friend difference_type operator-(integer_iterator a,
                                               integer_iterator b)
    {
        assert(a.stride_ == b.stride_);
        assert((*a - *b) % a.stride_ == 0);
        return (*a - *b) / a.stride_;
    }

    /** Subscript. */
    constexpr reference operator[](difference_type n) const
    {
        return *(*this + n);
    }

    // Boilerplate comparison operators
    /** non-equality */
    constexpr friend bool operator!=(integer_iterator a, integer_iterator b)
    {
        return !(a == b);
    }

    /** reverse advance */
    constexpr friend integer_iterator operator+(difference_type n,
                                                integer_iterator a)
    {
        return a + n;
    }

    /** less than */
    constexpr friend bool operator<(integer_iterator a, integer_iterator b)
    {
        return b - a > 0;
    }

    /** greater than */
    constexpr friend bool operator>(integer_iterator a, integer_iterator b)
    {
        return b < a;
    }

    /** greater equal */
    constexpr friend bool operator>=(integer_iterator a, integer_iterator b)
    {
        return !(a < b);
    }

    /** less equal */
    constexpr friend bool operator<=(integer_iterator a, integer_iterator b)
    {
        return !(a > b);
    }

private:
    index_type idx_;
    index_type stride_;
};


template <typename IndexType>
class irange_strided {
    static_assert(std::is_signed<IndexType>::value &&
                      std::is_integral<IndexType>::value,
                  "Can only use irange with signed integral types!");

public:
    using index_type = IndexType;
    using value_type = index_type;
    using iterator = integer_iterator<index_type>;

    explicit irange_strided(index_type begin, index_type end, index_type stride)
        : begin_{begin},
          end_{begin + (end - begin) / stride * stride +
               ((end - begin) % stride != 0 ? stride : 0)},
          stride_{stride}
    {
        assert(end >= begin);
        assert(stride > 0);
    }

    constexpr index_type begin_index() const { return begin_; }

    constexpr index_type end_index() const { return end_; }

    constexpr index_type stride() const { return stride_; }

    constexpr index_type size() const
    {
        return (end_index() - begin_index()) / stride();
    }

    constexpr bool empty() const { return size() == 0; }

    constexpr iterator begin() const
    {
        return iterator{begin_index(), stride()};
    }

    constexpr iterator end() const { return iterator{end_index(), stride()}; }

    constexpr friend bool operator==(irange_strided lhs, irange_strided rhs)
    {
        return lhs.begin() == rhs.begin() && lhs.end() == rhs.end() &&
               lhs.stride() == rhs.stride();
    }

    constexpr friend bool operator!=(irange_strided lhs, irange_strided rhs)
    {
        return !(lhs == rhs);
    }

private:
    index_type begin_;
    index_type end_;
    index_type stride_;
};


template <typename IndexType>
class irange {
    static_assert(std::is_signed<IndexType>::value &&
                      std::is_integral<IndexType>::value,
                  "Can only use irange with signed integral types!");

public:
    using index_type = IndexType;
    using value_type = index_type;
    using iterator = integer_iterator<index_type>;

    explicit irange(index_type begin, index_type end) : begin_{begin}, end_{end}
    {
        assert(end >= begin);
    }

    explicit irange(index_type end) : irange{0, end} {}

    constexpr index_type begin_index() const { return begin_; }

    constexpr index_type end_index() const { return end_; }

    constexpr index_type size() const { return end_index() - begin_index(); }

    constexpr bool empty() const { return size() == 0; }

    constexpr iterator begin() const { return iterator{begin_index()}; }

    constexpr iterator end() const { return iterator{end_index()}; }

    template <typename Group>
    constexpr irange_strided<index_type> striped(Group g) const
    {
        return striped(detail::group_traits<Group>::get_local_id(g),
                       detail::group_traits<Group>::get_size(g));
    }

    constexpr irange_strided<index_type> striped(index_type local_index,
                                                 index_type group_size) const
    {
        assert(local_index >= 0);
        assert(local_index < group_size);
        return irange_strided<index_type>{begin_ + local_index, end_,
                                          group_size};
    }

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


}  // namespace gko


#endif  // GKO_CORE_BASE_INTEGER_RANGE_HPP_
