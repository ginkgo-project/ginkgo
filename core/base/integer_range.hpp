// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_INTEGER_RANGE_HPP_
#define GKO_CORE_BASE_INTEGER_RANGE_HPP_


#include <cassert>
#include <iterator>
#include <type_traits>


#include "core/base/iterator_boilerplate.hpp"


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

    /** Dereference. */
    constexpr reference operator*() const { return idx_; }


    /** Iterator difference. */
    constexpr friend difference_type operator-(integer_iterator a,
                                               integer_iterator b)
    {
        assert(a.stride_ == b.stride_);
        assert((*a - *b) % a.stride_ == 0);
        return (*a - *b) / a.stride_;
    }

    /** Returns the stride of the iterator. */
    constexpr index_type stride() const { return stride_; }

    GKO_RANDOM_ACCESS_ITERATOR_BOILERPLATE(integer_iterator);

private:
    index_type idx_;
    index_type stride_;
};


template <typename IteratorType>
class random_access_range {
    static_assert(
        std::is_same<
            typename std::iterator_traits<IteratorType>::iterator_category,
            std::random_access_iterator_tag>::value,
        "IteratorType needs to be a random access iterator");

public:
    using iterator = IteratorType;
    using value_type = typename std::iterator_traits<IteratorType>::value_type;
    using difference_type =
        typename std::iterator_traits<IteratorType>::difference_type;
    using reference = typename std::iterator_traits<IteratorType>::reference;

    constexpr explicit random_access_range(iterator begin, iterator end)
        : begin_{begin}, end_{end}
    {}

    constexpr iterator begin() const { return begin_; }

    constexpr iterator end() const { return end_; }

    constexpr difference_type size() const { return end() - begin(); }

    constexpr bool empty() const { return size() == 0; }

    constexpr reference operator[](difference_type i) const
    {
        return begin()[i];
    }

private:
    iterator begin_;
    iterator end_;
};


template <typename IndexType>
class irange_strided : public random_access_range<integer_iterator<IndexType>> {
    using base = random_access_range<integer_iterator<IndexType>>;

public:
    using index_type = IndexType;
    using iterator = typename base::iterator;
    using value_type = typename base::value_type;
    using reference = typename base::reference;
    using difference_type = typename base::difference_type;

    constexpr explicit irange_strided(index_type begin, index_type end,
                                      index_type stride)
        : random_access_range<integer_iterator<IndexType>>{
              iterator{begin, stride},
              iterator{begin + (end - begin) / stride * stride +
                           ((end - begin) % stride != 0 ? stride : 0),
                       stride}}
    {
        assert(end >= begin);
        assert(stride > 0);
    }

    constexpr friend bool operator==(irange_strided lhs, irange_strided rhs)
    {
        return lhs.stride() == rhs.stride() && *lhs.begin() == *rhs.begin() &&
               *lhs.end() == *rhs.end();
    }

    constexpr friend bool operator!=(irange_strided lhs, irange_strided rhs)
    {
        return !(lhs == rhs);
    }

    constexpr index_type begin_index() const { return *this->begin(); }

    constexpr index_type end_index() const { return *this->end(); }

    constexpr index_type stride() const { return this->begin().stride(); }
};


template <typename IndexType>
class irange : public random_access_range<integer_iterator<IndexType>> {
    using base = random_access_range<integer_iterator<IndexType>>;

public:
    using index_type = IndexType;
    using iterator = typename base::iterator;
    using value_type = typename base::value_type;
    using reference = typename base::reference;
    using difference_type = typename base::difference_type;

    constexpr explicit irange(index_type begin, index_type end)
        : random_access_range<integer_iterator<IndexType>>{iterator{begin, 1},
                                                           iterator{end, 1}}
    {
        assert(end >= begin);
    }

    constexpr explicit irange(index_type end) : irange{0, end} {}

    constexpr index_type begin_index() const { return *this->begin(); }

    constexpr index_type end_index() const { return *this->end(); }

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
        return irange_strided<index_type>{begin_index() + local_index,
                                          end_index(), group_size};
    }

    constexpr friend bool operator==(irange lhs, irange rhs)
    {
        return lhs.begin() == rhs.begin() && lhs.end() == rhs.end();
    }

    constexpr friend bool operator!=(irange lhs, irange rhs)
    {
        return !(lhs == rhs);
    }
};


}  // namespace gko


#endif  // GKO_CORE_BASE_INTEGER_RANGE_HPP_
