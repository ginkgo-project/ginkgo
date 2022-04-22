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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_GENERIC_CONSTRUCTOR_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_GENERIC_CONSTRUCTOR_HPP_


#include <memory>
#include <string>
#include <type_traits>


#include <rapidjson/document.h>


#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/template_helper.hpp"
#include "resource_manager/base/types.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


class ResourceManager;

/**
 * Declare the overloading selection and the corresponding implementation must
 * to be the end of all others implementation. Use overloading function to avoid
 * the specialization of ... after instantiation.
 */
DECLARE_SELECTION(LinOp, RM_LinOp);
DECLARE_SELECTION(LinOpFactory, RM_LinOpFactory);
DECLARE_SELECTION(Executor, RM_Executor);
DECLARE_SELECTION(CriterionFactory, RM_CriterionFactory);
DECLARE_SELECTION(Logger, RM_Logger);


/**
 * create_from_config is a function to use the base string to select the next
 * function.
 *
 * @tparam T  the type
 *
 * @param item  the RapidJson::Value
 * @param base  the string from the base
 * @param exec  the Executor from outside
 * @param linop  the LinOp from outside
 * @param manager  the ResourceManager pointer
 */
template <typename T>
std::shared_ptr<T> create_from_config(rapidjson::Value& item, std::string base,
                                      std::shared_ptr<const Executor> exec,
                                      std::shared_ptr<const LinOp> linop,
                                      ResourceManager* manager);


CREATE_DEFAULT_IMPL(Executor);
CREATE_DEFAULT_IMPL(LinOp);
CREATE_DEFAULT_IMPL(LinOpFactory);
CREATE_DEFAULT_IMPL(CriterionFactory);
CREATE_DEFAULT_IMPL(Logger);


/**
 * Generic struct to implement the actual object creation.
 *
 * @tparam T  the type
 * @tparam U  the helper type.
 *
 * @note U is T by default, when T is derived from LinOpFactory or
 *       CriterionFactory (not included the base type), U is the outer type of
 *       Factory. This can help the template deduction and reduce huge amount of
 *       manual macro usage for building each template case.
 */
template <typename T, typename U = T>
struct Generic {
    using type = std::shared_ptr<T>;

    /**
     * build is the implementation to create the object from the input.
     *
     * @param item  the RapidJson::Value
     * @param exec  the Executor from outside
     * @param linop  the LinOp from outside
     * @param manager  the ResourceManager pointer
     */
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager);
};

GENERIC_BASE_IMPL(Executor);
GENERIC_BASE_IMPL(LinOp);
GENERIC_BASE_IMPL(LinOpFactory);
GENERIC_BASE_IMPL(CriterionFactory);
GENERIC_BASE_IMPL(Logger);


/**
 * GenericHelper is the helper to call Generic build with correct template
 * parameters. The default case uses `Generic<T, T>`
 *
 * @tparam T  the type
 */
template <typename T, typename = void>
struct GenericHelper {
    using type = std::shared_ptr<T>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        return Generic<T, T>::build(item, exec, linop, manager);
    }
};

/**
 * GenericHelper is the helper to call Generic build with correct template
 * parameters. This is the specialization cases for Factory Type (except for
 * base type), which uses `Generic<T, T::base_type>` and `T::base_type::Factory`
 * must be `T`.
 *
 * @tparam T  the type is derived from LinOpFactory or CriterionFactory but not
 *            LinOpFactory or CriterionFactory
 */
template <typename T>
struct GenericHelper<
    T, typename std::enable_if<is_on_linopfactory<T>::value ||
                               is_on_criterionfactory<T>::value>::type> {
    using type = std::shared_ptr<T>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        return Generic<T, typename T::base_type>::build(item, exec, linop,
                                                        manager);
    }
};


/**
 * create_from_config is a free function to build object from input.
 *
 * @param item  the RapidJson::Value
 * @param exec  the Executor from outside
 * @param linop  the LinOp from outside
 * @param manager  the ResourceManager pointer
 */
template <typename T>
std::shared_ptr<T> create_from_config(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec = nullptr,
    std::shared_ptr<const LinOp> linop = nullptr,
    ResourceManager* manager = nullptr)
{
    return GenericHelper<T>::build(item, exec, linop, manager);
}


/**
 * create_from_config is another overloading to implement the function after
 * selection on enum map.
 *
 * @tparam T  the enum type
 * @tparam base  the enum item
 * @tparam U  the corresponding base type of the enum type
 */
template <typename T, T base, typename U = typename gkobase<T>::type>
std::shared_ptr<U> create_from_config(rapidjson::Value& item,
                                      std::shared_ptr<const Executor> exec,
                                      std::shared_ptr<const LinOp> linop,
                                      ResourceManager* manager)
{
    std::cout << "empty" << std::endl;
    return nullptr;
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_GENERIC_CONSTRUCTOR_HPP_
