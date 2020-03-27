/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_ITERATOR_FACTORY_HPP_
#define GKO_CORE_BASE_ITERATOR_FACTORY_HPP_


#include <iterator>


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace detail {


/**
 * @internal
 * @brief This class is used to sort two distinct arrays (`dominant_values` and
 * `secondary_values`) with the same number of elements (it can be different
 * types) according to the `dominant_values` in ascending order.
 *
 * Stores the pointers of two arrays, one storing the type `SecondaryType`, and
 * one storing the type `ToSortType`. This class also provides an iterator
 * class, which can be used for `std::sort`. Without a custom iterator, memory
 * copies would be necessary to create, for example, one array of `std::pairs`
 * to use `std::sort`, or a self written sort function.
 *
 * Example of using this class to sort a list of people according to their age:
 * -----------------------------------------------------------------
 * ```cpp
 * std::vector<int> age{50, 44, 43, 42};
 * std::vector<std::string> person{"Karl", "Susanne", "Max", "Hannah"};
 * IteratorFactory<int, std::string> factory{age.data(), person.data(), 4};
 * std::sort(factory.begin(), factory.end());
 * ```
 * Here, `person` now contains: `{"Hannah", "Max", "Susanne", "Karl"}` and
 * `age` is now `{42, 43, 44, 50}`. Therefore, both arrays are now sorted
 * according to the values in ascending order of `age`.
 *
 * @tparam ToSortType  Type of the values which will be used for sorting.
 *                     It must support `operator<`.
 * @tparam SecondaryType  Type of the values which will be moved synchronous
 *                        to the array of type `ToSortType`. No comparisons
 *                        with this type will be performed.
 */
template <typename ToSortType, typename SecondaryType>
class IteratorFactory {
    // All nested classes are hidden, so they can't be misused
private:
    /**
     * Helper struct, needed for the default construction and assignment inside
     * `std::sort` through Reference. They are used as an intermediate data
     * type for some of the swaps `std::sort` performs.
     */
    struct element {
        ToSortType dominant;
        SecondaryType secondary;

        friend bool operator<(const element &left, const element &right)
        {
            return left.dominant < right.dominant;
        }
    };

    /**
     * This class is used as a reference to a sorting target, which abstracts
     * the existence of two distinct arrays.
     * It meets the requirements of `MoveAssignable`, `MoveConstructible`
     * In all comparisons, only the values of `dominant_values_` matter, while
     * the corresponding value of `secondary_values_` will always be copied /
     * moved / swapped to the same place.
     */
    class Reference {
    public:
        using array_index_type = int64;

        // An empty reference makes no sense, so is is disabled
        Reference() = delete;

        ~Reference() {}

        Reference(IteratorFactory &parent, array_index_type array_index)
            : parent_(parent), arr_index_(array_index)
        {}

        // Since it must be `MoveConstructible`
        Reference(Reference &&other)
            : parent_(other.parent_), arr_index_(std::move(other.arr_index_))
        {}

        Reference(const Reference &other)
            : parent_(other.parent_), arr_index_(other.arr_index_)
        {}


        Reference &operator=(element other)
        {
            dominant() = other.dominant;
            secondary() = other.secondary;
            return *this;
        }

        Reference &operator=(const Reference &other)
        {
            dominant() = other.dominant();
            secondary() = other.secondary();
            return *this;
        }

        // Since it must be `MoveAssignable`
        Reference &operator=(Reference &&other)
        {
            // In C++11, it is legal for a nested class to access private
            // members of the parent class.
            parent_.dominant_values_[arr_index_] =
                std::move(other.parent_.dominant_values_[other.arr_index_]);
            parent_.secondary_values_[arr_index_] =
                std::move(other.parent_.secondary_values_[other.arr_index_]);
            return *this;
        }

        // Conversion operator to `element`
        operator element() const { return {dominant(), secondary()}; }

        friend void swap(Reference a, Reference b)
        {
            std::swap(a.dominant(), b.dominant());
            std::swap(a.secondary(), b.secondary());
        }

        friend bool operator<(const Reference &left, const Reference &right)
        {
            return left.dominant() < right.dominant();
        }

        friend bool operator<(const Reference &left, const element &right)
        {
            return left.dominant() < right.dominant;
        }

        friend bool operator<(const element &left, const Reference &right)
        {
            return left.dominant < right.dominant();
        }

        ToSortType &dominant() { return parent_.dominant_values_[arr_index_]; }

        const ToSortType &dominant() const
        {
            return parent_.dominant_values_[arr_index_];
        }

        SecondaryType &secondary()
        {
            return parent_.secondary_values_[arr_index_];
        }

        const SecondaryType &secondary() const
        {
            return parent_.secondary_values_[arr_index_];
        }

    private:
        IteratorFactory &parent_;
        array_index_type arr_index_;
    };

    /**
     * The iterator that can be used for `std::sort`. It meets the requirements
     * of `LegacyRandomAccessIterator` and `ValueSwappable`.
     * For performance reasons, it is expected that all iterators that are
     * compared / used with each other have the same `parent`, so the check
     * if they are the same can be omitted.
     * This class uses a single variable to keep track of where the iterator
     * points to both arrays.
     */
    class Iterator {
    public:
        // Needed to count as a `LegacyRandomAccessIterator`
        using difference_type = typename Reference::array_index_type;
        using value_type = element;
        using pointer = Reference;
        using reference = Reference;
        using iterator_category = std::random_access_iterator_tag;

        ~Iterator() {}

        Iterator(IteratorFactory &parent, difference_type array_index)
            : parent_(parent), arr_index_(array_index)
        {}

        Iterator(const Iterator &other)
            : parent_(other.parent_), arr_index_(other.arr_index_)
        {}

        Iterator &operator=(const Iterator &other)
        {
            arr_index_ = other.arr_index_;
            return *this;
        }

        // Operators needed for the std::sort requirement of
        // `LegacyRandomAccessIterator`
        Iterator &operator+=(difference_type i)
        {
            arr_index_ += i;
            return *this;
        }

        Iterator &operator-=(difference_type i)
        {
            arr_index_ -= i;
            return *this;
        }

        Iterator &operator++()  // Prefix increment (++i)
        {
            ++arr_index_;
            return *this;
        }

        Iterator operator++(int)  // Postfix increment (i++)
        {
            Iterator temp(*this);
            ++arr_index_;
            return temp;
        }

        Iterator &operator--()  // Prefix decrement (--i)
        {
            --arr_index_;
            return *this;
        }

        Iterator operator--(int)  // Postfix decrement (i--)
        {
            Iterator temp(*this);
            --arr_index_;
            return temp;
        }

        Iterator operator+(difference_type i) const
        {
            return {parent_, arr_index_ + i};
        }

        friend Iterator operator+(difference_type i, const Iterator &iter)
        {
            return {iter.parent_, iter.arr_index_ + i};
        }

        Iterator operator-(difference_type i) const
        {
            return {parent_, arr_index_ - i};
        }

        difference_type operator-(const Iterator &other) const
        {
            return arr_index_ - other.arr_index_;
        }

        Reference operator*() const { return {parent_, arr_index_}; }

        Reference operator[](difference_type idx) const
        {
            return {parent_, arr_index_ + idx};
        }

        // Comparable operators
        bool operator==(const Iterator &other)
        {
            return arr_index_ == other.arr_index_;
        }

        bool operator!=(const Iterator &other)
        {
            return arr_index_ != other.arr_index_;
        }

        bool operator<(const Iterator &other) const
        {
            return arr_index_ < other.arr_index_;
        }

        bool operator<=(const Iterator &other) const
        {
            return arr_index_ <= other.arr_index_;
        }

        bool operator>(const Iterator &other) const
        {
            return arr_index_ > other.arr_index_;
        }

        bool operator>=(const Iterator &other) const
        {
            return arr_index_ >= other.arr_index_;
        }

    private:
        IteratorFactory &parent_;
        difference_type arr_index_;
    };

public:
    /**
     * Allows creating an iterator, which makes it look like the data consists
     * of `size` `std::pair<ToSortType, SecondaryType>`s, which are also
     * sortable (according to the value of `dominant_values`) as long as
     * `ToSortType` can be comparable with `operator<`. No additional data is
     * allocated, all operations performed through the Iterator object will be
     * done on the given arrays. The iterators given by this object (through
     * `begin()` and `end()`) can be used with `std::sort()`.
     * @param dominant_values  Array of at least `size` values, which are the
     * only values considered when comparing values (for example while sorting)
     * @param secondary_values  Array of at least `size` values, which will not
     * be considered when comparing. However, they will be moved / copied to the
     * same place as their corresponding value in `dominant_values` (with the
     * same index).
     * @param size  Size of the arrays when constructiong the iterators.
     * @note Both arrays must have at least `size` elements, otherwise, the
     * behaviour is undefined.
     */
    IteratorFactory(ToSortType *dominant_values,
                    SecondaryType *secondary_values, size_type size)
        : dominant_values_(dominant_values),
          secondary_values_(secondary_values),
          size_(size)
    {}

    /**
     * Creates an iterator pointing to the beginning of both arrays
     * @returns  an iterator pointing to the beginning of both arrays
     */
    Iterator begin() { return {*this, 0}; }

    /**
     * Creates an iterator pointing to the (excluding) end of both arrays
     * @returns  an iterator pointing to the (excluding) end of both arrays
     */
    Iterator end()
    {
        return {*this, static_cast<typename Iterator::difference_type>(size_)};
    }

private:
    ToSortType *dominant_values_;
    SecondaryType *secondary_values_;
    size_type size_;
};


}  // namespace detail
}  // namespace gko


#endif  // GKO_CORE_BASE_ITERATOR_FACTORY_HPP_
