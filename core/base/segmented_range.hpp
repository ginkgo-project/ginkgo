// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_SEGMENTED_RANGE_HPP_
#define GKO_CORE_BASE_SEGMENTED_RANGE_HPP_


#include <iterator>
#include <type_traits>

#include "core/base/index_range.hpp"
#include "core/base/iterator_factory.hpp"


namespace gko {


/**
 * Represents a range of indices that is segmented into contiguous segments.
 * Each segment has the shape `[begin, end)`, i.e. it is a half-open interval.
 *
 * @tparam IndexType  the type of indices used to represent the segments.
 */
template <typename IndexType>
class segmented_range {
public:
    using index_type = IndexType;
    using index_iterator_type = index_iterator<index_type>;
    using segment_type = irange<index_type>;

    /**
     * An iterator pointing to (or past) a single segment in the range.
     */
    class iterator {
    public:
        constexpr explicit iterator(segmented_range range, index_type segment)
            : range_{range}, segment_{segment}
        {}

        struct enumerated_segment {
            index_type index;
            segment_type segment;
        };

        constexpr enumerated_segment operator*() const
        {
            assert(segment_ >= 0);
            assert(segment_ < range_.num_segments());
            return enumerated_segment{segment_,
                                      segment_type{range_.begin_index(segment_),
                                                   range_.end_index(segment_)}};
        }

        constexpr iterator& operator++()
        {
            ++segment_;
            return *this;
        }

        constexpr friend bool operator==(iterator lhs, iterator rhs)
        {
            assert(lhs.range_ == rhs.range_);
            return lhs.segment_ == rhs.segment_;
        }

        constexpr friend bool operator!=(iterator lhs, iterator rhs)
        {
            return !(lhs == rhs);
        }

    private:
        segmented_range range_;
        index_type segment_;
    };

    /**
     * Constructs a segmented range from separate begin and end pointers.
     * The `i`th range is given by `[begins[i], ends[i])`.
     *
     * @param begins  a pointer to the array of beginning indices
     * @param ends  a pointer to the array of end indices
     * @param num_segments  the number of segments, i.e. the size of the
     *                      beginning and end index arrays.
     */
    constexpr explicit segmented_range(const index_type* begins,
                                       const index_type* ends,
                                       index_type num_segments)
        : begins_{begins}, ends_{ends}, num_segments_{num_segments}
    {
        assert(num_segments_ >= 0);
    }

    /**
     * Constructs a segmented range from combined begin and end pointers.
     * The `i`th range is given by `[ptrs[i], ptrs[i + 1])`.
     *
     * @param ptrs  a pointer to the array of beginning and end indices
     * @param num_segments  the number of segments, i.e. the size of the
     *                      ptrs index arrays.
     */
    constexpr explicit segmented_range(const index_type* ptrs,
                                       index_type num_segments)
        : segmented_range{ptrs, ptrs + 1, num_segments}
    {}

    /**
     * Returns the segment at a given index.
     *
     * @param segment  the index to access. It must be in `[0, num_segments())`.
     * @return  the segment at this index.
     */
    constexpr segment_type operator[](index_type segment) const
    {
        assert(segment >= 0);
        assert(segment < num_segments());
        return (*iterator{*this, segment}).segment;
    }

    /** @return the number of segments in this range. */
    constexpr index_type num_segments() const { return num_segments_; }

    /** @return an index range representing all segment indices. */
    constexpr irange<index_type> segment_indices() const
    {
        return irange<index_type>{num_segments()};
    }

    /** @return iterator pointing to the first segment. */
    constexpr iterator begin() const { return iterator{*this, 0}; }

    /** @return iterator pointing one past the last segment. */
    constexpr iterator end() const { return iterator{*this, num_segments()}; }

    /** @return iterator pointing to the first segment. */
    constexpr const index_type* begin_indices() const { return begins_; }

    /** @return iterator pointing one past the last segment. */
    constexpr const index_type* end_indices() const { return ends_; }

    /** @return the beginning index of the given segment. */
    constexpr index_type begin_index(index_type segment) const
    {
        assert(segment >= 0);
        assert(segment < num_segments());
        return begin_indices()[segment];
    }

    /** @return the end index of the given segment. */
    constexpr index_type end_index(index_type segment) const
    {
        assert(segment >= 0);
        assert(segment < num_segments());
        return end_indices()[segment];
    }

    /** Compares two ranges for equality. */
    constexpr friend bool operator==(segmented_range lhs, segmented_range rhs)
    {
        return lhs.begin_indices() == rhs.begin_indices() &&
               lhs.end_indices() == rhs.end_indices() &&
               lhs.num_segments() == rhs.num_segments();
    }

    /** Compares two ranges for inequality. */
    constexpr friend bool operator!=(segmented_range lhs, segmented_range rhs)
    {
        return !(lhs == rhs);
    }

private:
    const index_type* begins_;
    const index_type* ends_;
    index_type num_segments_;
};


/**
 * Represents a range of indices that is segmented into contiguous segments,
 * mapped into a value array. Each segment has the shape `[begin, end)`, i.e. it
 * is a half-open interval and points to corresponding entries of the value
 * array.
 *
 * @tparam IndexType  the type of indices used to represent the segments.
 * @tparam ValueIterator  the iterator type pointing to the values.
 */
template <typename IndexType, typename ValueIterator>
class segmented_value_range {
public:
    using index_type = IndexType;
    using index_iterator_type = index_iterator<index_type>;
    using value_iterator = ValueIterator;
    using segment_type = iterator_range<ValueIterator>;
    using enumerated_range = segmented_value_range<
        index_type, detail::zip_iterator<index_iterator_type, value_iterator>>;

    /**
     * An iterator pointing to (or past) a single segment in the range.
     */
    class iterator {
    public:
        constexpr explicit iterator(segmented_value_range range,
                                    index_type segment)
            : range_{range}, segment_{segment}
        {}

        struct enumerated_segment {
            index_type index;
            segment_type segment;
        };

        constexpr enumerated_segment operator*() const
        {
            assert(segment_ >= 0);
            assert(segment_ < range_.num_segments());
            return enumerated_segment{
                segment_,
                segment_type{range_.values() + range_.begin_index(segment_),
                             range_.values() + range_.end_index(segment_)}};
        }

        constexpr iterator& operator++()
        {
            ++segment_;
            return *this;
        }

        constexpr friend bool operator==(iterator lhs, iterator rhs)
        {
            assert(lhs.range_ == rhs.range_);
            return lhs.segment_ == rhs.segment_;
        }

        constexpr friend bool operator!=(iterator lhs, iterator rhs)
        {
            return !(lhs == rhs);
        }

    private:
        segmented_value_range range_;
        index_type segment_;
    };

    /**
     * Constructs a segmented values range from separate begin and end pointers.
     * The `i`th range is given by `[begins[i], ends[i])`.
     *
     * @param begins  a pointer to the array of beginning indices
     * @param ends  a pointer to the array of end indices
     * @param values  an iterator pointing to the values into which the
     *                beginning/end indices point.
     * @param num_segments  the number of segments, i.e. the size of the
     *                      beginning and end index arrays.
     */
    constexpr explicit segmented_value_range(const index_type* begins,
                                             const index_type* ends,
                                             value_iterator values,
                                             index_type num_segments)
        : begins_{begins},
          ends_{ends},
          values_{values},
          num_segments_{num_segments}
    {
        assert(num_segments_ >= 0);
    }

    /**
     * Constructs a segmented range from combined begin and end pointers.
     * The `i`th range is given by `[ptrs[i], ptrs[i + 1])`.
     *
     * @param ptrs  a pointer to the array of beginning and end indices
     * @param values  an iterator pointing to the values into which the
     *                beginning/end indices point.
     * @param num_segments  the number of segments, i.e. the size of the
     *                      ptrs index arrays.
     */
    constexpr explicit segmented_value_range(const index_type* ptrs,
                                             value_iterator values,
                                             index_type num_segments)
        : segmented_value_range{ptrs, ptrs + 1, values, num_segments}
    {}

    /**
     * Returns the segment at a given index.
     *
     * @param segment  the index to access. It must be in `[0, num_segments())`.
     * @return  the segment at this index.
     */
    constexpr segment_type operator[](index_type segment) const
    {
        assert(segment >= 0);
        assert(segment < num_segments());
        return (*iterator{*this, segment}).segment;
    }

    /** @return the number of segments in this range. */
    constexpr index_type num_segments() const { return num_segments_; }

    /** @return an index range representing all segment indices. */
    constexpr irange<index_type> segment_indices() const
    {
        return irange<index_type>{num_segments()};
    }

    constexpr enumerated_range enumerated() const
    {
        return enumerated_range{
            begin_indices(), end_indices(),
            detail::make_zip_iterator(index_iterator{0}, values()),
            num_segments()};
    }

    /** @return iterator pointing to the first segment. */
    constexpr iterator begin() const { return iterator{*this, 0}; }

    /** @return iterator pointing one past the last segment. */
    constexpr iterator end() const { return iterator{*this, num_segments()}; }

    /** @return iterator pointing to the first segment. */
    constexpr const index_type* begin_indices() const { return begins_; }

    /** @return iterator pointing one past the last segment. */
    constexpr const index_type* end_indices() const { return ends_; }

    /** @return the beginning index of the given segment. */
    constexpr index_type begin_index(index_type segment) const
    {
        assert(segment >= 0);
        assert(segment < num_segments());
        return begin_indices()[segment];
    }

    /** @return the end index of the given segment. */
    constexpr index_type end_index(index_type segment) const
    {
        assert(segment >= 0);
        assert(segment < num_segments());
        return end_indices()[segment];
    }

    /** @return the value iterator. */
    constexpr value_iterator values() const { return values_; }

    /** Compares two ranges for equality. */
    constexpr friend bool operator==(segmented_value_range lhs,
                                     segmented_value_range rhs)
    {
        return lhs.begin_indices() == rhs.begin_indices() &&
               lhs.end_indices() == rhs.end_indices() &&
               lhs.values() == rhs.values() &&
               lhs.num_segments() == rhs.num_segments();
    }

    /** Compares two ranges for inequality. */
    constexpr friend bool operator!=(segmented_value_range lhs,
                                     segmented_value_range rhs)
    {
        return !(lhs == rhs);
    }

private:
    const index_type* begins_;
    const index_type* ends_;
    value_iterator values_;
    index_type num_segments_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_SEGMENTED_RANGE_HPP_
