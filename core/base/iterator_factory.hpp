/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <cassert>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>


namespace gko {
namespace detail {


template <typename... Args>
void tuple_unpack_sink(Args&&...)
{}


template <typename... Iterators>
class zip_iterator;


template <typename... Iterators>
class zip_iterator_reference
    : public std::tuple<
          typename std::iterator_traits<Iterators>::reference...> {
    using ref_tuple_type =
        std::tuple<typename std::iterator_traits<Iterators>::reference...>;
    using value_type =
        std::tuple<typename std::iterator_traits<Iterators>::value_type...>;
    using index_sequence = std::index_sequence_for<Iterators...>;

    friend class zip_iterator<Iterators...>;

    template <std::size_t... idxs>
    value_type cast_impl(std::index_sequence<idxs...>) const
    {
        return {std::get<idxs>(*this)...};
    }

    template <std::size_t... idxs>
    void assign_impl(std::index_sequence<idxs...>, const value_type& other)
    {
        tuple_unpack_sink(
            (std::get<idxs>(*this) = std::get<idxs>(other), 0)...);
    }

    zip_iterator_reference(Iterators... it) : ref_tuple_type{*it...} {}

public:
    operator value_type() const { return cast_impl(index_sequence{}); }

    zip_iterator_reference& operator=(const value_type& other)
    {
        assign_impl(index_sequence{}, other);
        return *this;
    }
};


template <typename... Iterators>
class zip_iterator {
    static_assert(sizeof...(Iterators) > 0, "Can't build empty zip iterator");

public:
    using difference_type = std::ptrdiff_t;
    using value_type =
        std::tuple<typename std::iterator_traits<Iterators>::value_type...>;
    using pointer = value_type*;
    using reference = zip_iterator_reference<Iterators...>;
    using iterator_category = std::random_access_iterator_tag;
    using index_sequence = std::index_sequence_for<Iterators...>;

    explicit zip_iterator() = default;

    explicit zip_iterator(Iterators... its) : iterators_{its...} {}

    zip_iterator& operator+=(difference_type i)
    {
        forall([i](auto& it) { it += i; });
        return *this;
    }

    zip_iterator& operator-=(difference_type i)
    {
        forall([i](auto& it) { it -= i; });
        return *this;
    }

    zip_iterator& operator++()
    {
        forall([](auto& it) { it++; });
        return *this;
    }

    zip_iterator operator++(int)
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    zip_iterator& operator--()
    {
        forall([](auto& it) { it--; });
        return *this;
    }

    zip_iterator operator--(int)
    {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    zip_iterator operator+(difference_type i) const
    {
        auto tmp = *this;
        tmp += i;
        return tmp;
    }

    friend zip_iterator operator+(difference_type i, const zip_iterator& iter)
    {
        return iter + i;
    }

    zip_iterator operator-(difference_type i) const
    {
        auto tmp = *this;
        tmp -= i;
        return tmp;
    }

    difference_type operator-(const zip_iterator& other) const
    {
        return forall_check_consistent(
            other, [](const auto& a, const auto& b) { return a - b; });
    }

    reference operator*() const
    {
        return deref_impl(std::index_sequence_for<Iterators...>{});
    }

    reference operator[](difference_type i) const { return *(*this + i); }

    bool operator==(const zip_iterator& other) const
    {
        return forall_check_consistent(
            other, [](const auto& a, const auto& b) { return a == b; });
    }

    bool operator!=(const zip_iterator& other) const
    {
        return !(*this == other);
    }

    bool operator<(const zip_iterator& other) const
    {
        return forall_check_consistent(
            other, [](const auto& a, const auto& b) { return a < b; });
    }

    bool operator<=(const zip_iterator& other) const
    {
        return forall_check_consistent(
            other, [](const auto& a, const auto& b) { return a <= b; });
    }

    bool operator>(const zip_iterator& other) const
    {
        return !(*this <= other);
    }

    bool operator>=(const zip_iterator& other) const
    {
        return !(*this < other);
    }

private:
    template <std::size_t... idxs>
    reference deref_impl(std::index_sequence<idxs...>) const
    {
        return reference{std::get<idxs>(iterators_)...};
    }

    template <typename Functor>
    void forall(Functor fn)
    {
        forall_impl(fn, index_sequence{});
    }

    template <typename Functor, std::size_t... idxs>
    void forall_impl(Functor fn, std::index_sequence<idxs...>)
    {
        tuple_unpack_sink((fn(std::get<idxs>(iterators_)), 0)...);
    }

    template <typename Functor, std::size_t... idxs>
    void forall_impl(const zip_iterator& other, Functor fn,
                     std::index_sequence<idxs...>) const
    {
        tuple_unpack_sink(
            (fn(std::get<idxs>(iterators_), std::get<idxs>(other.iterators_)),
             0)...);
    }

    template <typename Functor>
    auto forall_check_consistent(const zip_iterator& other, Functor fn) const
    {
        auto result =
            fn(std::get<0>(iterators_), std::get<0>(other.iterators_));
        forall_impl(
            other, [&](auto a, auto b) { assert(fn(a, b) == result); },
            index_sequence{});
        return result;
    }

    std::tuple<Iterators...> iterators_;
};


template <typename... Iterators>
zip_iterator<std::decay_t<Iterators>...> make_zip_iterator(Iterators&&... it)
{
    return zip_iterator<std::decay_t<Iterators>...>{
        std::forward<Iterators>(it)...};
}


template <typename... Iterators>
void swap(zip_iterator_reference<Iterators...> a,
          zip_iterator_reference<Iterators...> b)
{
    typename zip_iterator<Iterators...>::value_type tmp = a;
    a = b;
    b = tmp;
}


template <typename... Iterators>
void swap(typename zip_iterator<Iterators...>::value_type& a,
          zip_iterator_reference<Iterators...> b)
{
    typename zip_iterator<Iterators...>::value_type tmp = a;
    a = b;
    b = tmp;
}


template <typename... Iterators>
void swap(zip_iterator_reference<Iterators...> a,
          typename zip_iterator<Iterators...>::value_type& b)
{
    typename zip_iterator<Iterators...>::value_type tmp = a;
    a = b;
    b = tmp;
}


}  // namespace detail
}  // namespace gko


#endif  // GKO_CORE_BASE_ITERATOR_FACTORY_HPP_
