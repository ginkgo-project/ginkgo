// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_CONVERSION_HPP_
#define GKO_PUBLIC_CORE_BASE_CONVERSION_HPP_


#include <tuple>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>

namespace gko {
namespace detail {


using precision_list = std::tuple<
#if GINKGO_ENABLE_HALF
    half,
#endif
#if GINKGO_ENABLE_BFLOAT16
    bfloat16,
#endif
    float, double>;

template <typename Removed, typename Tuple, typename Current>
struct remove_type {};

template <typename Removed, typename... Current>
struct remove_type<Removed, std::tuple<>, std::tuple<Current...>> {
    using type = std::tuple<Current...>;
};

template <typename Removed, typename V, typename... Rest, typename... Current>
struct remove_type<Removed, std::tuple<V, Rest...>, std::tuple<Current...>> {
    using type = typename remove_type<Removed, std::tuple<Rest...>,
                                      std::tuple<Current..., V>>::type;
};

template <typename Removed, typename... Rest, typename... Current>
struct remove_type<Removed, std::tuple<Removed, Rest...>,
                   std::tuple<Current...>> {
    using type = typename remove_type<Removed, std::tuple<Rest...>,
                                      std::tuple<Current...>>::type;
};

template <typename T, typename List>
using get_precision_list = typename remove_type<
    T,
    std::conditional_t<is_complex_impl<T>::value,
                       typename to_complex_s<List>::type, List>,
    std::tuple<>>::type;


}  // namespace detail


template <typename ConcreteType, typename ResultType>
class EnableConvertibleTo : public ConvertibleTo<ResultType> {
public:
    void convert_to(ResultType* result) const override
    {
        self()->convert_to_impl(result);
    }

    void move_to(ResultType* result) override { self()->move_to_impl(result); }

private:
    GKO_ENABLE_SELF(ConcreteType);
};


template <typename Class, typename List>
class EnableConvertibleToList {};

template <template <typename, typename...> class Class, typename... List,
          typename V, typename... Rest>
class EnableConvertibleToList<Class<V, Rest...>, std::tuple<List...>>
    : public EnableConvertibleTo<Class<V, Rest...>, Class<List, Rest...>>... {};


}  // namespace gko

#endif  // GKO_PUBLIC_CORE_BASE_CONVERSION_HPP_
