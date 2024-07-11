// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_ITERATOR_FACTORY_HPP_
#define GKO_CORE_BASE_ITERATOR_FACTORY_HPP_


#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <tuple>
#include <utility>

#include <ginkgo/core/base/types.hpp>

#include "core/base/copy_assignable.hpp"


namespace gko {
namespace detail {


template <typename... Iterators>
class zip_iterator;


template <typename... Iterators>
class zip_iterator_reference;


template <typename T, typename... Ts>
class device_tuple;


}  // namespace detail
}  // namespace gko


// structured binding specializations for device_tuple, zip_iterator_reference
namespace std {


template <typename... Ts>
struct tuple_size<gko::detail::device_tuple<Ts...>>
    : integral_constant<size_t, sizeof...(Ts)> {};


template <std::size_t I, typename... Ts>
struct tuple_element<I, gko::detail::device_tuple<Ts...>> {
    using type = typename tuple_element<I, tuple<Ts...>>::type;
};


template <typename... Iterators>
struct tuple_size<gko::detail::zip_iterator_reference<Iterators...>>
    : integral_constant<size_t, sizeof...(Iterators)> {};


template <std::size_t I, typename... Iterators>
struct tuple_element<I, gko::detail::zip_iterator_reference<Iterators...>> {
    using type = typename iterator_traits<
        typename tuple_element<I, tuple<Iterators...>>::type>::reference;
};


}  // namespace std


namespace gko {


/** std::get reimplementation for device_tuple. */
template <std::size_t index, typename... Ts>
constexpr typename std::tuple_element<index, detail::device_tuple<Ts...>>::type&
get(detail::device_tuple<Ts...>& tuple);


/** std::get reimplementation for const device_tuple. */
template <std::size_t index, typename... Ts>
constexpr const typename std::tuple_element<index,
                                            detail::device_tuple<Ts...>>::type&
get(const detail::device_tuple<Ts...>& tuple);


namespace detail {


/** simplified constexpr std::tuple reimplementation for use in device code. */
template <typename T, typename... Ts>
class device_tuple {
public:
    /** Constructs a device tuple from its elements. */
    constexpr explicit device_tuple(T value, Ts... others)
        : value_{value}, other_{others...}
    {}

    device_tuple() = default;

    /**
     * Copy-assigns a tuple.
     * This is necessary to make tuples of references work, which normally cause
     * the impliciy copy-assignment operator to be deleted.
     */
    constexpr device_tuple& operator=(const device_tuple& other)
    {
        value_ = other.value_;
        other_ = other.other_;
        return *this;
    }

    /** @return the index-th element in the tuple. */
    template <std::size_t index>
    constexpr typename std::tuple_element<index, device_tuple>::type& get()
    {
        if constexpr (index == 0) {
            return value_;
        } else {
            return other_.template get<index - 1>();
        }
    }

    /** @return the index-th element in the const tuple. */
    template <std::size_t index>
    constexpr const typename std::tuple_element<index, device_tuple>::type&
    get() const
    {
        if constexpr (index == 0) {
            return value_;
        } else {
            return other_.template get<index - 1>();
        }
    }

    // comparison operators
    constexpr friend bool operator<(const device_tuple& lhs,
                                    const device_tuple& rhs)
    {
        return lhs.value_ < rhs.value_ ||
               (lhs.value_ == rhs.value_ && lhs.other_ < rhs.other_);
    }

    constexpr friend bool operator>(const device_tuple& lhs,
                                    const device_tuple& rhs)
    {
        return rhs < lhs;
    }

    constexpr friend bool operator>=(const device_tuple& lhs,
                                     const device_tuple& rhs)
    {
        return !(lhs < rhs);
    }

    constexpr friend bool operator<=(const device_tuple& lhs,
                                     const device_tuple& rhs)
    {
        return !(lhs > rhs);
    }

    constexpr friend bool operator==(const device_tuple& lhs,
                                     const device_tuple& rhs)
    {
        return lhs.value_ == rhs.value_ && lhs.other_ == rhs.other_;
    }

    constexpr friend bool operator!=(const device_tuple& lhs,
                                     const device_tuple& rhs)
    {
        return !(lhs == rhs);
    }

private:
    T value_;
    device_tuple<Ts...> other_;
};


template <typename T>
class device_tuple<T> {
public:
    /** Constructs a device tuple from its elements. */
    constexpr explicit device_tuple(T value) : value_{value} {}

    device_tuple() = default;

    /**
     * Copy-assigns a tuple.
     * This is necessary to make tuples of references work, which normally cause
     * the impliciy copy-assignment operator to be deleted.
     */
    constexpr device_tuple& operator=(const device_tuple& other)
    {
        value_ = other.value_;
        return *this;
    }

    /** @return the index-th element in the tuple. */
    template <std::size_t index>
    constexpr T& get()
    {
        static_assert(index == 0, "invalid index");
        return value_;
    }

    /** @return the index-th element in the const tuple. */
    template <std::size_t index>
    constexpr const T& get() const
    {
        static_assert(index == 0, "invalid index");
        return value_;
    }

    // comparison operators
    constexpr friend bool operator<(const device_tuple& lhs,
                                    const device_tuple& rhs)
    {
        return lhs.value_ < rhs.value_;
    }

    constexpr friend bool operator>(const device_tuple& lhs,
                                    const device_tuple& rhs)
    {
        return rhs < lhs;
    }

    constexpr friend bool operator>=(const device_tuple& lhs,
                                     const device_tuple& rhs)
    {
        return !(lhs < rhs);
    }

    constexpr friend bool operator<=(const device_tuple& lhs,
                                     const device_tuple& rhs)
    {
        return !(lhs > rhs);
    }

    constexpr friend bool operator==(const device_tuple& lhs,
                                     const device_tuple& rhs)
    {
        return lhs.value_ == rhs.value_;
    }

    constexpr friend bool operator!=(const device_tuple& lhs,
                                     const device_tuple& rhs)
    {
        return !(lhs == rhs);
    }

private:
    T value_;
};


/**
 * A reference-like type pointing to a tuple of elements originating from a
 * tuple of iterators. A few caveats related to its use:
 *
 * 1. It should almost never be stored as a reference, i.e.
 * `auto& ref = *it` leads to a dangling reference, since the
 * `zip_iterator_reference` returned by `*it` is a temporary.
 *
 * 2. Any copy of the object is itself a reference to the same entry, i.e.
 * `auto ref_copy = ref` means that assigning values to `ref_copy` also changes
 * the data referenced by `ref`
 *
 * 3. If you want to copy the data, assign it to a variable of value_type:
 * `tuple<int, float> val = ref` or use the `copy` member function
 * `auto val = ref.copy()`
 *
 * @see zip_iterator
 * @tparam Iterators  the iterators that are zipped together
 */
template <typename... Iterators>
class zip_iterator_reference
    : public device_tuple<
          typename std::iterator_traits<Iterators>::reference...> {
    using ref_tuple_type =
        device_tuple<typename std::iterator_traits<Iterators>::reference...>;
    using value_type =
        device_tuple<typename std::iterator_traits<Iterators>::value_type...>;
    using index_sequence = std::index_sequence_for<Iterators...>;

    friend class zip_iterator<Iterators...>;

    template <std::size_t... idxs>
    constexpr value_type cast_impl(std::index_sequence<idxs...>) const
    {
        // gcc 5 throws error as using uninitialized array
        // std::tuple<int, char> t = { 1, '2' }; is not allowed.
        // converting to 'std::tuple<...>' from initializer list would use
        // explicit constructor
        return value_type(gko::get<idxs>(*this)...);
    }

    template <std::size_t... idxs>
    constexpr void assign_impl(std::index_sequence<idxs...>,
                               const value_type& other)
    {
        (void)std::initializer_list<int>{
            (gko::get<idxs>(*this) = gko::get<idxs>(other), 0)...};
    }

    constexpr explicit zip_iterator_reference(Iterators... it)
        : ref_tuple_type{*it...}
    {}

public:
    constexpr operator value_type() const
    {
        return cast_impl(index_sequence{});
    }

    constexpr zip_iterator_reference& operator=(const value_type& other)
    {
        assign_impl(index_sequence{}, other);
        return *this;
    }

    constexpr value_type copy() const { return *this; }
};


/**
 * A generic iterator adapter that combines multiple separate random access
 * iterators for types a, b, c, ... into an iterator over tuples of type
 * (a, b, c, ...).
 * Dereferencing it returns a reference-like zip_iterator_reference object,
 * similar to std::vector<bool> bit references. Accesses through that reference
 * to the individual tuple elements get translated to the corresponding
 * iterator's references.
 *
 * @note Two zip_iterators can only be compared if each individual pair of
 *       wrapped iterators has the same distance. Otherwise the behavior is
 *       undefined. This means that the only guaranteed safe way to use multiple
 *       zip_iterators is if they are all derived from the same iterator:
 *       ```
 *       Iterator i, j;
 *       auto it1 = make_zip_iterator(i, j);
 *       auto it2 = make_zip_iterator(i, j + 1);
 *       auto it3 = make_zip_iterator(i + 1, j + 1);
 *       auto it4 = it1 + 1;
 *       it1 == it2; // undefined
 *       it1 == it3; // well-defined false
 *       it3 == it4; // well-defined true
 *       ```
 *       This property is checked automatically in Debug builds and assumed in
 *       Release builds.
 *
 * @see zip_iterator_reference
 * @tparam Iterators  the iterators to zip together
 */
template <typename... Iterators>
class zip_iterator {
    static_assert(sizeof...(Iterators) > 0, "Can't build empty zip iterator");

public:
    using difference_type = std::ptrdiff_t;
    using value_type =
        device_tuple<typename std::iterator_traits<Iterators>::value_type...>;
    using pointer = value_type*;
    using reference = zip_iterator_reference<Iterators...>;
    using iterator_category = std::random_access_iterator_tag;
    using index_sequence = std::index_sequence_for<Iterators...>;

    constexpr zip_iterator() = default;

    constexpr explicit zip_iterator(Iterators... its) : iterators_{its...} {}

    constexpr zip_iterator& operator+=(difference_type i)
    {
        forall([i](auto& it) { it += i; });
        return *this;
    }

    constexpr zip_iterator& operator-=(difference_type i)
    {
        forall([i](auto& it) { it -= i; });
        return *this;
    }

    constexpr zip_iterator& operator++()
    {
        forall([](auto& it) { it++; });
        return *this;
    }

    constexpr zip_iterator operator++(int)
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    constexpr zip_iterator& operator--()
    {
        forall([](auto& it) { it--; });
        return *this;
    }

    constexpr zip_iterator operator--(int)
    {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    constexpr zip_iterator operator+(difference_type i) const
    {
        auto tmp = *this;
        tmp += i;
        return tmp;
    }

    constexpr friend zip_iterator operator+(difference_type i,
                                            const zip_iterator& iter)
    {
        return iter + i;
    }

    constexpr zip_iterator operator-(difference_type i) const
    {
        auto tmp = *this;
        tmp -= i;
        return tmp;
    }

    constexpr difference_type operator-(const zip_iterator& other) const
    {
        return forall_check_consistent(
            other, [](const auto& a, const auto& b) { return a - b; });
    }

    constexpr reference operator*() const
    {
        return deref_impl(std::index_sequence_for<Iterators...>{});
    }

    constexpr reference operator[](difference_type i) const
    {
        return *(*this + i);
    }

    constexpr bool operator==(const zip_iterator& other) const
    {
        return forall_check_consistent(
            other, [](const auto& a, const auto& b) { return a == b; });
    }

    constexpr bool operator!=(const zip_iterator& other) const
    {
        return !(*this == other);
    }

    constexpr bool operator<(const zip_iterator& other) const
    {
        return forall_check_consistent(
            other, [](const auto& a, const auto& b) { return a < b; });
    }

    constexpr bool operator<=(const zip_iterator& other) const
    {
        return forall_check_consistent(
            other, [](const auto& a, const auto& b) { return a <= b; });
    }

    constexpr bool operator>(const zip_iterator& other) const
    {
        return !(*this <= other);
    }

    constexpr bool operator>=(const zip_iterator& other) const
    {
        return !(*this < other);
    }

private:
    template <std::size_t... idxs>
    constexpr reference deref_impl(std::index_sequence<idxs...>) const
    {
        return reference{get<idxs>(iterators_)...};
    }

    template <typename Functor>
    constexpr void forall(Functor fn)
    {
        forall_impl(fn, index_sequence{});
    }

    template <typename Functor, std::size_t... idxs>
    constexpr void forall_impl(Functor fn, std::index_sequence<idxs...>)
    {
        (void)std::initializer_list<int>{(fn(get<idxs>(iterators_)), 0)...};
    }

    template <typename Functor, std::size_t... idxs>
    constexpr void forall_impl(const zip_iterator& other, Functor fn,
                               std::index_sequence<idxs...>) const
    {
        (void)std::initializer_list<int>{
            (fn(get<idxs>(iterators_), get<idxs>(other.iterators_)), 0)...};
    }

    template <typename Functor>
    constexpr auto forall_check_consistent(const zip_iterator& other,
                                           Functor fn) const
    {
        auto it = get<0>(iterators_);
        auto other_it = get<0>(other.iterators_);
        auto result = fn(it, other_it);
        forall_impl(
            other, [&](auto a, auto b) { GKO_ASSERT(it - other_it == a - b); },
            index_sequence{});
        return result;
    }

    device_tuple<Iterators...> iterators_;
};


template <typename... Iterators>
constexpr zip_iterator<std::decay_t<Iterators>...> make_zip_iterator(
    Iterators&&... it)
{
    return zip_iterator<std::decay_t<Iterators>...>{
        std::forward<Iterators>(it)...};
}


/**
 * Swap function for zip iterator references. It takes care of creating a
 * non-reference temporary to avoid the problem of a normal std::swap():
 * ```
 * // a and b are reference-like objects pointing to different entries
 * auto tmp = a; // tmp is a reference-like type, so this is not a copy!
 * a = b;        // copies value at b to a, which also modifies tmp
 * b = tmp;      // copies value at tmp (= a) to b
 * // now both a and b point to the same value that was originally at b
 * ```
 * It is modelled after the behavior of std::vector<bool> bit references.
 * To swap in generic code, use the pattern `using std::swap; swap(a, b);`
 *
 * @tparam Iterators  the iterator types inside the corresponding zip_iterator
 */
template <typename... Iterators>
constexpr void swap(zip_iterator_reference<Iterators...> a,
                    zip_iterator_reference<Iterators...> b)
{
    auto tmp = a.copy();
    a = b;
    b = tmp;
}


/**
 * @copydoc swap(zip_iterator_reference, zip_iterator_reference)
 */
template <typename... Iterators>
constexpr void swap(typename zip_iterator<Iterators...>::value_type& a,
                    zip_iterator_reference<Iterators...> b)
{
    auto tmp = a;
    a = b;
    b = tmp;
}


/**
 * @copydoc swap(zip_iterator_reference, zip_iterator_reference)
 */
template <typename... Iterators>
constexpr void swap(zip_iterator_reference<Iterators...> a,
                    typename zip_iterator<Iterators...>::value_type& b)
{
    auto tmp = a.copy();
    a = b;
    b = tmp;
}


/**
 * Random access iterator that uses a function to transform the index.
 *
 * For a function `fn` and an underlying iterator `it`, accessing the
 * permute_iterator at index `i` will result in accessing `it[fn(i)]`.
 *
 * @tparam IteratorType  Underlying iterator, has to be random access.
 * @tparam PermuteFn  A function `difference_type -> difference_type` that
 *                    transforms any given index. It doesn't have to be a strict
 *                    permutation of indices (i.e. not bijective).
 */
template <typename IteratorType, typename PermuteFn>
class permute_iterator {
public:
    using difference_type = std::ptrdiff_t;
    using value_type = typename std::iterator_traits<IteratorType>::value_type;
    using pointer = typename std::iterator_traits<IteratorType>::pointer;
    using reference = typename std::iterator_traits<IteratorType>::reference;
    using iterator_category = std::random_access_iterator_tag;

    permute_iterator() = default;

    explicit permute_iterator(IteratorType it, PermuteFn perm)
        : it_{std::move(it)}, idx_{}, perm_{std::move(perm)}
    {}

    permute_iterator& operator+=(difference_type i)
    {
        idx_ += i;
        return *this;
    }

    permute_iterator& operator-=(difference_type i) { return *this += -i; }

    permute_iterator& operator++() { return *this += 1; }

    permute_iterator operator++(int)
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    permute_iterator& operator--() { return *this -= 1; }

    permute_iterator operator--(int)
    {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    permute_iterator operator+(difference_type i) const
    {
        auto tmp = *this;
        tmp += i;
        return tmp;
    }

    friend permute_iterator operator+(difference_type i,
                                      const permute_iterator& iter)
    {
        return iter + i;
    }

    permute_iterator operator-(difference_type i) const
    {
        auto tmp = *this;
        tmp -= i;
        return tmp;
    }

    difference_type operator-(const permute_iterator& other) const
    {
        return idx_ - other.idx_;
    }

    reference operator*() const { return it_[perm_(idx_)]; }

    reference operator[](difference_type i) const { return *(*this + i); }

    bool operator==(const permute_iterator& other) const
    {
        return idx_ == other.idx_;
    }

    bool operator!=(const permute_iterator& other) const
    {
        return !(*this == other);
    }

    bool operator<(const permute_iterator& other) const
    {
        return idx_ < other.idx_;
    }

    bool operator<=(const permute_iterator& other) const
    {
        return idx_ <= other.idx_;
    }

    bool operator>(const permute_iterator& other) const
    {
        return !(*this <= other);
    }

    bool operator>=(const permute_iterator& other) const
    {
        return !(*this < other);
    }

private:
    IteratorType it_;
    difference_type idx_;
    copy_assignable<PermuteFn> perm_;
};


template <typename IteratorType, typename PermutationFn>
permute_iterator<IteratorType, PermutationFn> make_permute_iterator(
    IteratorType it, PermutationFn perm)
{
    return permute_iterator<IteratorType, PermutationFn>{std::move(it),
                                                         std::move(perm)};
}


}  // namespace detail


template <std::size_t index, typename... Ts>
constexpr typename std::tuple_element<index, detail::device_tuple<Ts...>>::type&
get(detail::device_tuple<Ts...>& tuple)
{
    return tuple.template get<index>();
}


template <std::size_t index, typename... Ts>
constexpr const typename std::tuple_element<index,
                                            detail::device_tuple<Ts...>>::type&
get(const detail::device_tuple<Ts...>& tuple)
{
    return tuple.template get<index>();
}


}  // namespace gko


#endif  // GKO_CORE_BASE_ITERATOR_FACTORY_HPP_
