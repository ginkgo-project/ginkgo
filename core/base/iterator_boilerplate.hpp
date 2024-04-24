// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_ITERATOR_BOILERPLATE_HPP_
#define GKO_CORE_BASE_ITERATOR_BOILERPLATE_HPP_


/**
 * Implements all `random_access_iterator` operations for _iterator in terms of
 * the already implemented advance `operator +=(difference_type)`, the
 * difference operator `operator-(_iterator, _iterator)` and the deference
 * operator `operator*()`
 */
#define GKO_RANDOM_ACCESS_ITERATOR_BOILERPLATE(_iterator)                      \
    /** InputIterator requirements */                                          \
    /** Equality. */                                                           \
    constexpr friend bool operator==(_iterator a, _iterator b)                 \
    {                                                                          \
        return a - b == 0;                                                     \
    }                                                                          \
                                                                               \
    /** non-equality */                                                        \
    constexpr friend bool operator!=(_iterator a, _iterator b)                 \
    {                                                                          \
        return !(a == b);                                                      \
    }                                                                          \
                                                                               \
    /* BidirectionalIterator requirements*/                                    \
    /** Pre-increment. */                                                      \
    constexpr _iterator& operator++() { return *this += 1; }                   \
                                                                               \
    /** Post-increment. */                                                     \
    constexpr _iterator operator++(int)                                        \
    {                                                                          \
        auto tmp{*this};                                                       \
        operator++();                                                          \
        return tmp;                                                            \
    }                                                                          \
                                                                               \
    /** Pre-decrement. */                                                      \
    constexpr _iterator& operator--() { return *this -= 1; }                   \
                                                                               \
    /** Post-decrement. */                                                     \
    constexpr _iterator operator--(int)                                        \
    {                                                                          \
        auto tmp{*this};                                                       \
        operator--();                                                          \
        return tmp;                                                            \
    }                                                                          \
                                                                               \
    /** RandomAccessIterator requirements */                                   \
    /** Iterator advance. */                                                   \
    constexpr friend _iterator operator+(_iterator a, difference_type n)       \
    {                                                                          \
        return a += n;                                                         \
    }                                                                          \
                                                                               \
    /** reverse advance */                                                     \
    constexpr friend _iterator operator+(difference_type n, _iterator a)       \
    {                                                                          \
        return a + n;                                                          \
    }                                                                          \
                                                                               \
    /** Iterator backwards advance. */                                         \
    constexpr _iterator& operator-=(difference_type n) { return *this += -n; } \
                                                                               \
    /** Iterator backwards advance. */                                         \
    constexpr friend _iterator operator-(_iterator a, difference_type n)       \
    {                                                                          \
        return a + -n;                                                         \
    }                                                                          \
                                                                               \
    /** Subscript. */                                                          \
    constexpr reference operator[](difference_type n) const                    \
    {                                                                          \
        return *(*this + n);                                                   \
    }                                                                          \
                                                                               \
    /** less than */                                                           \
    constexpr friend bool operator<(_iterator a, _iterator b)                  \
    {                                                                          \
        return b - a > 0;                                                      \
    }                                                                          \
                                                                               \
    /** greater than */                                                        \
    constexpr friend bool operator>(_iterator a, _iterator b)                  \
    {                                                                          \
        return b < a;                                                          \
    }                                                                          \
                                                                               \
    /** greater equal */                                                       \
    constexpr friend bool operator>=(_iterator a, _iterator b)                 \
    {                                                                          \
        return !(a < b);                                                       \
    }                                                                          \
                                                                               \
    /** less equal */                                                          \
    constexpr friend bool operator<=(_iterator a, _iterator b)                 \
    {                                                                          \
        return !(a > b);                                                       \
    }


#endif  // GKO_CORE_BASE_ITERATOR_BOILERPLATE_HPP_
