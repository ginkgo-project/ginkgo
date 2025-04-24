// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_TYPE_TRAITS_HPP_
#define GKO_PUBLIC_CORE_BASE_TYPE_TRAITS_HPP_

#include <type_traits>

#include <ginkgo/core/base/lin_op.hpp>

namespace gko {
namespace detail {


template <typename Type>
constexpr bool is_ginkgo_linop = std::is_base_of_v<LinOp, Type>;


// helper to get factory type of concrete type or LinOp
template <typename Type>
struct factory_type_impl {
    using type = typename Type::Factory;
};

// It requires LinOp to be complete type
template <>
struct factory_type_impl<LinOp> {
    using type = LinOpFactory;
};


template <typename Type>
using factory_type = typename factory_type_impl<Type>::type;


// helper for handle the transposed type of concrete type and LinOp
template <typename Type, typename = void>
struct transposed_type_impl {
    using type = typename Type::transposed_type;
};

// It requires LinOp to be complete type
template <>
struct transposed_type_impl<LinOp, void> {
    using type = LinOp;
};


// return the same type when Type is the precision format.
// it is used in ILU.
template <typename Type>
struct transposed_type_impl<Type, std::enable_if_t<!is_ginkgo_linop<Type>>> {
    using type = Type;
};

template <typename Type>
using transposed_type = typename transposed_type_impl<Type>::type;


// helper to get value_type of concrete type or void for LinOp
template <typename Type, typename = void>
struct get_value_type_impl {
    using type = typename Type::value_type;
};

// We need to use SFINAE not conditional_t because both type needs to be
// valid in conditional_t
template <typename Type>
struct get_value_type_impl<Type, std::enable_if_t<!is_ginkgo_linop<Type>>> {
    using type = Type;
};


template <typename Type>
using get_value_type = typename get_value_type_impl<Type>::type;


}  // namespace detail
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_BASE_TYPE_TRAITS_HPP_
