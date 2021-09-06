/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKOEXT_RESOURCE_MANAGER_BASE_TYPES_HPP_
#define GKOEXT_RESOURCE_MANAGER_BASE_TYPES_HPP_

#include <ginkgo/ginkgo.hpp>
#include <resource_manager/base/macro_helper.hpp>


namespace gko {
namespace extension {
namespace resource_manager {


using Executor = ::gko::Executor;
using LinOp = ::gko::LinOp;
using LinOpFactory = ::gko::LinOpFactory;
using CriterionFactory = ::gko::stop::CriterionFactory;
using ExecutorMap = std::unordered_map<std::string, std::shared_ptr<Executor>>;
using LinOpMap = std::unordered_map<std::string, std::shared_ptr<LinOp>>;
using LinOpFactoryMap =
    std::unordered_map<std::string, std::shared_ptr<LinOpFactory>>;
using CriterionFactoryMap =
    std::unordered_map<std::string, std::shared_ptr<CriterionFactory>>;


/**
 * map_type gives the map type according to the base type of given type.
 *
 * @tparam T  the type
 */
template <typename T, typename = void>
struct map_type {
    using type = void;
};

template <typename T>
struct map_type<T, typename std::enable_if<
                       std::is_convertible<T *, Executor *>::value>::type> {
    using type = ExecutorMap;
};

template <typename T>
struct map_type<T, typename std::enable_if<
                       std::is_convertible<T *, LinOp *>::value>::type> {
    using type = LinOpMap;
};

template <typename T>
struct map_type<T, typename std::enable_if<
                       std::is_convertible<T *, LinOpFactory *>::value>::type> {
    using type = LinOpFactoryMap;
};

template <typename T>
struct map_type<T, typename std::enable_if<std::is_convertible<
                       T *, CriterionFactory *>::value>::type> {
    using type = CriterionFactoryMap;
};


/**
 * base_type gives the base type of given type
 *
 * @tparam T  the type
 */
template <typename T, typename = void>
struct base_type {
    using type = void;
};

template <typename T>
struct base_type<T, typename std::enable_if<
                        std::is_convertible<T *, Executor *>::value>::type> {
    using type = Executor;
};

template <typename T>
struct base_type<T, typename std::enable_if<
                        std::is_convertible<T *, LinOp *>::value>::type> {
    using type = LinOp;
};

template <typename T>
struct base_type<T, typename std::enable_if<std::is_convertible<
                        T *, LinOpFactory *>::value>::type> {
    using type = LinOpFactory;
};

template <typename T>
struct base_type<T, typename std::enable_if<std::is_convertible<
                        T *, CriterionFactory *>::value>::type> {
    using type = CriterionFactory;
};


#define ENUM_EXECUTER(_expand)                                           \
    _expand(Executor, 0), _expand(CudaExecutor), _expand(DpcppExecutor), \
        _expand(HipExecutor), _expand(OmpExecutor), _expand(ReferenceExecutor)

ENUM_CLASS(RM_Executor, int, ENUM_EXECUTER);


#define ENUM_LINOP(_expand)                                                   \
    _expand(LinOp, 0), _expand(LinOpWithFactory), _expand(Cg), _expand(Isai), \
        _expand(Jacobi), _expand(LinOpWithOutFactory), _expand(Csr),          \
        _expand(Dense), _expand(Ilu), _expand(LowerTrs), _expand(UpperTrs),   \
        _expand(IluFactorization)

ENUM_CLASS(RM_LinOp, int, ENUM_LINOP);


#define ENUM_LINOPFACTORY(_expand)                                             \
    _expand(LinOpFactory, 0), _expand(CgFactory), _expand(IsaiFactory),        \
        _expand(JacobiFactory), _expand(IluFactory), _expand(LowerTrsFactory), \
        _expand(UpperTrsFactory), _expand(IluFactorizationFactory)

ENUM_CLASS(RM_LinOpFactory, int, ENUM_LINOPFACTORY);


#define ENUM_CRITERIONFACTORY(_expand) \
    _expand(CriterionFactory, 0), _expand(Iteration), _expand(ResidualNorm)

ENUM_CLASS(RM_CriterionFactory, int, ENUM_CRITERIONFACTORY);


/**
 * gkobase give the base type according to the enum class type
 *
 * @tparam T  the enum class type
 */
template <typename T>
struct gkobase {
    using type = void;
};

template <>
struct gkobase<RM_Executor> {
    using type = Executor;
};

template <>
struct gkobase<RM_LinOp> {
    using type = LinOp;
};

template <>
struct gkobase<RM_LinOpFactory> {
    using type = LinOpFactory;
};

template <>
struct gkobase<RM_CriterionFactory> {
    using type = CriterionFactory;
};


/**
 * is_derived returns true is the Derived is derived from Base and is not Base,
 * or returns false
 *
 * @tparam Derived  the derived class
 * @tparam Base  the base class
 */
template <typename Derived, typename Base, typename U = void>
struct is_derived : public std::integral_constant<bool, false> {};

template <typename Derived, typename Base>
struct is_derived<
    Derived, Base,
    typename std::enable_if<std::is_convertible<const volatile Derived *,
                                                const volatile Base *>::value &&
                            !std::is_same<const volatile Derived,
                                          const volatile Base>::value>::type>
    : public std::integral_constant<bool, true> {};

/**
 * is_on_linopfactory is a shortcut for check whether T is derived from
 * LinOpFactory but not LinOpFactory
 *
 * @tparam T  the type
 */
template <typename T>
using is_on_linopfactory = is_derived<T, LinOpFactory>;


/**
 * is_on_criterionfactory is a shortcut for check whether T is derived from
 * CriterionFactory but not CriterionFactory
 *
 * @tparam T  the type
 */
template <typename T>
using is_on_criterionfactory = is_derived<T, CriterionFactory>;


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko

#endif  // GKOEXT_RESOURCE_MANAGER_BASE_TYPES_HPP_
