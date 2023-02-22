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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_TYPE_RESOLVING_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_TYPE_RESOLVING_HPP_


#include <type_traits>


#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>


#include "resource_manager/base/type_pack.hpp"
#include "resource_manager/base/types.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


/**
 * get_the_type is to apply the type T into base.
 * get_the_type<base, T>::type = base<T>;
 *
 * @tparam base  the templated class
 * @tparam T  the arg of template
 */
template <template <typename...> class base, typename T>
struct get_the_type {
    using type = base<T>;
};

/**
 * get_the_type is to apply the type_list into base, which requires several
 * template parametrs. get_the_type<base, type_list<T...>>::type = base<T...>;
 *
 * @tparam base  the templated class
 * @tparam ...Rest  the args from type_list
 */
template <template <typename...> class base, typename... Rest>
struct get_the_type<base, type_list<Rest...>> {
    using type = base<Rest...>;
};


/**
 * get_the_factory_type is to get the Factory class from base<T>.
 * get_the_factory_type<base, T>::type = base<T>::Factory;
 *
 * @tparam base  the templated class
 * @tparam T  the arg of template
 */
template <template <typename...> class base, typename T>
struct get_the_factory_type {
    using type = typename get_the_type<base, T>::type::Factory;
};


/**
 * acutul_type is a struct to give the flexibilty to control some special cases.
 * For example, gives a integral_constant but use the value in practice.
 *
 * @tparam T  the type
 *
 * @note It is required because selection uses type_list as template input to
 * support several template. Thus, we use integral_constant to store value and
 * then transfer them by this class. For example,
 * ```
 * template <int i, typename VT>
 * struct actual_class{};
 * ```
 * We only pass the type_list as template input, so it only supports types.
 * We can also use type_list as base type or some specific class name to avoid
 * several type with the same template signature. In this case, we will call
 * `get_the_type<type_list, type_list<integral_constant<int i>, VT>>` and we
 * will get the same type `type_list<integral_constant<int i>, VT>>` as `Type`
 * and we specialize actual_type as the following
 * ```
 * template <int i, typename VT>
 * struct actual_type<type_list<integral_constant<int, i>, VT>> {
 *     using type = actual_class<i, VT>;
 * }
 * ```
 * Thus, from `actual_type<Type>::type`, we can get the actual_class with
 * desired templated parameters.
 */
template <typename T>
struct actual_type {
    using type = T;
};

template <gko::preconditioner::isai_type IsaiType, typename ValueType,
          typename IndexType>
struct actual_type<
    type_list<std::integral_constant<RM_LinOp, RM_LinOp::Isai>,
              std::integral_constant<gko::preconditioner::isai_type, IsaiType>,
              ValueType, IndexType>> {
    using type = gko::preconditioner::Isai<IsaiType, ValueType, IndexType>;
};

template <typename LSolverType, typename USolverType, bool ReverseApply,
          typename IndexType>
struct actual_type<type_list<
    std::integral_constant<RM_LinOp, RM_LinOp::Ilu>, LSolverType, USolverType,
    std::integral_constant<bool, ReverseApply>, IndexType>> {
    using type = gko::preconditioner::Ilu<LSolverType, USolverType,
                                          ReverseApply, IndexType>;
};

/**
 * get_actual_type uses `actual_type<get_the_type<base, T>::type>::type` to
 * handle those classes with value template.
 *
 * @tparam base  the templated class
 * @tparam T  the templated parameters.
 */
template <template <typename...> class base, typename T>
struct get_actual_type {
    using type =
        typename actual_type<typename get_the_type<base, T>::type>::type;
};


/**
 * get_actual_factory_type uses `actual_type<get_the_type<base,
 * T>::type>::type::Factory` to does the same thing as `get_actual_type` but get
 * the Factory class.
 *
 * @tparam base  the templated class
 * @tparam T  the templated parameters.
 */
template <template <typename...> class base, typename T>
struct get_actual_factory_type {
    using type = typename get_actual_type<base, T>::type::Factory;
};


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko

#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_TYPE_RESOLVING_HPP_
