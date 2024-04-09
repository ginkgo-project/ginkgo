// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_SEGMENTED_RANGE_HPP_
#define GKO_CORE_BASE_SEGMENTED_RANGE_HPP_


#include <iterator>
#include <type_traits>


#include "core/base/integer_range.hpp"
#include "core/base/iterator_boilerplate.hpp"


namespace gko {


template <typename ValueIterator,
          typename IndexIterator = integer_iterator<
              typename std::iterator_traits<ValueIterator>::difference_type>>
class indexed_iterator {
public:
    using value_iterator = ValueIterator;
    using index_iterator = IndexIterator;

private:
    using value_traits = std::iterator_traits<value_iterator>;
    using index_traits = std::iterator_traits<index_iterator>;

public:
    using value_type = typename value_traits::value_type;
    using index_type = typename index_traits::value_type;
    using difference_type = typename index_traits::difference_type;
    using pointer = typename value_traits::pointer;
    using reference = typename value_traits::reference;
    using iterator_category = std::random_access_iterator_tag;

    constexpr explicit indexed_iterator(value_iterator value_it,
                                        index_iterator index_it)
        : value_it_{value_it}, index_it_{index_it}
    {}

    constexpr indexed_iterator() : indexed_iterator{{}, {}} {}

    constexpr indexed_iterator& operator+=(difference_type n)
    {
        index_it_ += n;
        return *this;
    }

    constexpr difference_type index() const { return *index_it_; }

    constexpr reference operator*() const { return value_it_[index()]; }

    constexpr pointer operator->() const { return &operator*(); }

    constexpr friend difference_type operator-(indexed_iterator a,
                                               indexed_iterator b)
    {
        return a.index_it_ - b.index_it_;
    }

    GKO_RANDOM_ACCESS_ITERATOR_BOILERPLATE(indexed_iterator);

private:
    value_iterator value_it_;
    index_iterator index_it_;
};


template <typename ValueIterator,
          typename IndexIterator = integer_iterator<
              typename std::iterator_traits<ValueIterator>::difference_type>>
class enumerating_indexed_iterator
    : public indexed_iterator<ValueIterator, IndexIterator> {
private:
    using base = indexed_iterator<ValueIterator, IndexIterator>;

public:
    using base_value_type = typename base::value_type;
    using index_type = typename base::index_type;

    struct enumerated {
        index_type index;
        base_value_type value;

        constexpr explicit enumerated(index_type index, base_value_type value)
            : index{index}, value{value}
        {}

        constexpr friend bool operator==(enumerated a, enumerated b)
        {
            return a.index == b.index && a.value == b.value;
        }

        constexpr friend bool operator!=(enumerated a, enumerated b)
        {
            return !(a == b);
        }
    };

    using value_type = enumerated;
    using difference_type = typename base::difference_type;
    using pointer = const enumerated*;
    using reference = value_type;  // we shouldn't hand out references to
                                   // the enumerated struct for lifetime safety
    using iterator_category = std::random_access_iterator_tag;

    constexpr explicit enumerating_indexed_iterator(
        typename base::value_iterator value_it,
        typename base::index_iterator index_it)
        : base{value_it, index_it},
          enumerated_(base::index(), base::operator*())
    {}

    constexpr enumerating_indexed_iterator& operator+=(difference_type n)
    {
        base::operator+=(n);
        this->enumerated_ = this->operator*();
        return *this;
    }

    constexpr reference operator*() const
    {
        return enumerated{this->index(), this->base::operator*()};
    }

    constexpr pointer operator->() const { return &enumerated_; }

    // override all boilerplate operators, in particular operator[]
    GKO_RANDOM_ACCESS_ITERATOR_BOILERPLATE(enumerating_indexed_iterator);

private:
    enumerated enumerated_;
};


template <typename ValueIterator, typename IndexIterator>
class indexed_range : public random_access_range<
                          indexed_iterator<ValueIterator, IndexIterator>> {
    using base =
        random_access_range<indexed_iterator<ValueIterator, IndexIterator>>;

public:
    using value_iterator = ValueIterator;
    using index_iterator = IndexIterator;
    using iterator = typename base::iterator;
    using value_type = typename base::value_type;
    using reference = typename base::reference;
    using difference_type = typename base::difference_type;

    constexpr explicit indexed_range(value_iterator value_it,
                                     index_iterator begin_index_it,
                                     index_iterator end_index_it)
        : base{iterator{value_it, begin_index_it},
               iterator{value_it, end_index_it}}
    {}

    constexpr friend bool operator==(indexed_range lhs, indexed_range rhs)
    {
        return lhs.begin() == rhs.begin() && lhs.end() == rhs.end();
    }

    constexpr friend bool operator!=(indexed_range lhs, indexed_range rhs)
    {
        return !(lhs == rhs);
    }
};


template <typename ValueIterator, typename IndexIterator>
class enumerating_indexed_range
    : public random_access_range<
          enumerating_indexed_iterator<ValueIterator, IndexIterator>> {
    using base = random_access_range<
        enumerating_indexed_iterator<ValueIterator, IndexIterator>>;

public:
    using value_iterator = ValueIterator;
    using index_iterator = IndexIterator;
    using iterator = typename base::iterator;
    using value_type = typename base::value_type;
    using reference = typename base::reference;
    using difference_type = typename base::difference_type;

    constexpr explicit enumerating_indexed_range(value_iterator value_it,
                                                 index_iterator begin_index_it,
                                                 index_iterator end_index_it)
        : base{iterator{value_it, begin_index_it},
               iterator{value_it, end_index_it}}
    {}

    constexpr friend bool operator==(enumerating_indexed_range lhs,
                                     enumerating_indexed_range rhs)
    {
        return lhs.begin() == rhs.begin() && lhs.end() == rhs.end();
    }

    constexpr friend bool operator!=(enumerating_indexed_range lhs,
                                     enumerating_indexed_range rhs)
    {
        return !(lhs == rhs);
    }
};


template <typename IndexIterator>
class segmented_range_iterator {
public:
    using index_iterator = IndexIterator;

private:
    using index_traits = std::iterator_traits<index_iterator>;

public:
    using index_type = typename index_traits::value_type;
    using value_type = irange<index_type>;
    using difference_type = typename index_traits::difference_type;
    using pointer = const value_type*;
    using reference = value_type;  // we shouldn't hand out references to
                                   // the segment for lifetime safety
    using iterator_category = std::random_access_iterator_tag;

    constexpr explicit segmented_range_iterator(index_iterator begins_it,
                                                index_iterator ends_it)
        : begins_it_{begins_it}, ends_it_{ends_it}
    {}

    constexpr segmented_range_iterator& operator+=(difference_type n)
    {
        begins_it_ += n;
        ends_it_ += n;
        return *this;
    }

    constexpr reference operator*() const
    {
        return value_type{*begins_it_, *ends_it_};
    }

    constexpr friend difference_type operator-(segmented_range_iterator a,
                                               segmented_range_iterator b)
    {
        assert(a.begins_it - b.begins_it == a.ends_it_ - b.ends_it_);
        return a.begins_it - b.begins_it;
    }

    GKO_RANDOM_ACCESS_ITERATOR_BOILERPLATE(segmented_range_iterator);

private:
    index_iterator begins_it_;
    index_iterator ends_it_;
};


template <typename IndexIterator>
class segmented_range
    : public random_access_range<segmented_range_iterator<IndexIterator>> {
    using base = random_access_range<segmented_range_iterator<IndexIterator>>;

public:
    using index_iterator = IndexIterator;
    using iterator = typename base::iterator;
    using value_type = typename base::value_type;
    using reference = typename base::reference;
    using difference_type = typename base::difference_type;

    constexpr explicit segmented_range(index_iterator begins_it,
                                       index_iterator ends_it,
                                       difference_type num_segments)
        : base{iterator{begins_it, ends_it},
               iterator{begins_it + num_segments, ends_it + num_segments}}
    {}

    constexpr friend bool operator==(segmented_range lhs, segmented_range rhs)
    {
        return lhs.begin() == rhs.begin() && lhs.end() == rhs.end();
    }

    constexpr friend bool operator!=(segmented_range lhs, segmented_range rhs)
    {
        return !(lhs == rhs);
    }
};


template <typename ValueIterator, typename IndexIterator>
class segmented_enumerating_value_range_iterator {
public:
    using value_iterator = ValueIterator;
    using index_iterator = IndexIterator;

private:
    using value_traits = std::iterator_traits<index_iterator>;
    using index_traits = std::iterator_traits<index_iterator>;

public:
    using index_type = typename index_traits::value_type;
    using index_range = irange<index_type>;
    using value_type =
        enumerating_indexed_range<value_iterator,
                                  typename index_range::iterator>;
    using difference_type = typename index_traits::difference_type;
    using pointer = const value_type*;
    using reference = value_type;  // we shouldn't hand out references to
                                   // the segment for lifetime safety
    using iterator_category = std::random_access_iterator_tag;

    constexpr explicit segmented_enumerating_value_range_iterator(
        value_iterator value_it, index_iterator begins_it,
        index_iterator ends_it)
        : value_it_{value_it}, begins_it_{begins_it}, ends_it_{ends_it}
    {}

    constexpr segmented_enumerating_value_range_iterator& operator+=(
        difference_type n)
    {
        begins_it_ += n;
        ends_it_ += n;
        return *this;
    }

    constexpr reference operator*() const
    {
        index_range range{*begins_it_, *ends_it_};
        return value_type{value_it_, range.begin(), range.end()};
    }

    constexpr friend difference_type operator-(
        segmented_enumerating_value_range_iterator a,
        segmented_enumerating_value_range_iterator b)
    {
        assert(a.value_it_ == b.value_it_);
        assert(a.begins_it - b.begins_it == a.ends_it_ - b.ends_it_);
        return a.begins_it - b.begins_it;
    }

    GKO_RANDOM_ACCESS_ITERATOR_BOILERPLATE(
        segmented_enumerating_value_range_iterator);

private:
    value_iterator value_it_;
    index_iterator begins_it_;
    index_iterator ends_it_;
};


template <typename ValueIterator, typename IndexIterator>
class segmented_value_range_iterator {
public:
    using value_iterator = ValueIterator;
    using index_iterator = IndexIterator;

private:
    using value_traits = std::iterator_traits<index_iterator>;
    using index_traits = std::iterator_traits<index_iterator>;

public:
    using index_type = typename index_traits::value_type;
    using index_range = irange<index_type>;
    using value_type =
        indexed_range<value_iterator, typename index_range::iterator>;
    using difference_type = typename index_traits::difference_type;
    using pointer = const value_type*;
    using reference = value_type;  // we shouldn't hand out references to
                                   // the segment for lifetime safety
    using enumerating_type =
        segmented_enumerating_value_range_iterator<value_iterator,
                                                   index_iterator>;
    using iterator_category = std::random_access_iterator_tag;

    constexpr explicit segmented_value_range_iterator(value_iterator value_it,
                                                      index_iterator begins_it,
                                                      index_iterator ends_it)
        : value_it_{value_it}, begins_it_{begins_it}, ends_it_{ends_it}
    {}

    constexpr segmented_value_range_iterator& operator+=(difference_type n)
    {
        begins_it_ += n;
        ends_it_ += n;
        return *this;
    }

    constexpr enumerating_type as_enumerating() const
    {
        return enumerating_type{value_it_, begins_it_, ends_it_};
    }

    constexpr reference operator*() const
    {
        index_range range{*begins_it_, *ends_it_};
        return value_type{value_it_, range.begin(), range.end()};
    }

    constexpr friend difference_type operator-(segmented_value_range_iterator a,
                                               segmented_value_range_iterator b)
    {
        assert(a.value_it_ == b.value_it_);
        assert(a.begins_it - b.begins_it == a.ends_it_ - b.ends_it_);
        return a.begins_it - b.begins_it;
    }

    GKO_RANDOM_ACCESS_ITERATOR_BOILERPLATE(segmented_value_range_iterator);

private:
    value_iterator value_it_;
    index_iterator begins_it_;
    index_iterator ends_it_;
};


template <typename ValueIterator, typename IndexIterator>
class segmented_value_range
    : public random_access_range<
          segmented_value_range_iterator<ValueIterator, IndexIterator>> {
    using base = random_access_range<
        segmented_value_range_iterator<ValueIterator, IndexIterator>>;

public:
    using value_iterator = ValueIterator;
    using index_iterator = IndexIterator;
    using iterator = typename base::iterator;
    using value_type = typename base::value_type;
    using reference = typename base::reference;
    using difference_type = typename base::difference_type;
    using enumerating_iterator =
        segmented_enumerating_value_range_iterator<ValueIterator,
                                                   IndexIterator>;
    using enumerating_value_type = typename enumerating_iterator::value_type;

    constexpr explicit segmented_value_range(value_iterator value_it,
                                             index_iterator begins_it,
                                             index_iterator ends_it,
                                             difference_type num_segments)
        : base{iterator{value_it, begins_it, ends_it},
               iterator{value_it, begins_it + num_segments,
                        ends_it + num_segments}}
    {}

    constexpr enumerating_iterator enumerate_begin() const
    {
        return this->begin().as_enumerating();
    }

    constexpr enumerating_iterator enumerate_end() const
    {
        return this->end().as_enumerating();
    }

    constexpr enumerating_value_type enumerate(difference_type n) const
    {
        return this->enumerate_begin()[n];
    }

    constexpr friend bool operator==(segmented_value_range lhs,
                                     segmented_value_range rhs)
    {
        return lhs.begin() == rhs.begin() && lhs.end() == rhs.end();
    }

    constexpr friend bool operator!=(segmented_value_range lhs,
                                     segmented_value_range rhs)
    {
        return !(lhs == rhs);
    }
};


}  // namespace gko


#endif  // GKO_CORE_BASE_SEGMENTED_RANGE_HPP_
