// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_UTILS_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_UTILS_HPP_


#include <type_traits>

#include <ginkgo/core/preconditioner/isai.hpp>


namespace gko {
namespace preconditioner {
namespace detail {


// true_type if ExplicitType is an instantiation of TemplateType.
template <typename ExplicitType, template <typename...> class TemplateType>
struct is_instantiation_of : std::false_type {};

template <template <typename...> class Type, typename... Param>
struct is_instantiation_of<Type<Param...>, Type> : std::true_type {};

// LowerIsai will be treated as Isai<...> so it does not match LowerIsai<...>
template <typename ValueType, typename IndexType>
struct is_instantiation_of<LowerIsai<ValueType, IndexType>, LowerIsai>
    : std::true_type {};


}  // namespace detail
}  // namespace preconditioner
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_UTILS_HPP_
