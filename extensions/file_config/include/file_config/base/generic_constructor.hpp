/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_PUBLIC_EXT_FILE_CONFIG_BASE_GENERIC_CONSTRUCTOR_HPP_
#define GKO_PUBLIC_EXT_FILE_CONFIG_BASE_GENERIC_CONSTRUCTOR_HPP_


#include <nlohmann/json.hpp>


#include <ginkgo/ginkgo.hpp>


#include "file_config/base/macro_helper.hpp"
#include "file_config/base/template_helper.hpp"
#include "file_config/base/types.hpp"


namespace gko {
namespace extensions {
namespace file_config {


class ResourceManager;


/**
 * create_from_config is a function to use the base string to select the next
 * function.
 *
 * @tparam T  the type
 *
 * @param item  the const nlohmann::json
 * @param base  the string from the base
 * @param exec  the Executor from outside
 * @param linop  the LinOp from outside
 * @param manager  the ResourceManager pointer
 */
template <typename T>
std::shared_ptr<T> create_from_config(const nlohmann::json& item,
                                      std::string base,
                                      std::shared_ptr<const Executor> exec,
                                      std::shared_ptr<const LinOp> linop,
                                      ResourceManager* manager);

/**
 * DECLARE_SELECTION is to declare a specialization on create_from_config such
 * that call the overloading selection function on base type.
 *
 * @param _base_type  the base type
 */
#define DECLARE_SELECTION(_base_type)                           \
    template <>                                                 \
    std::shared_ptr<_base_type> create_from_config<_base_type>( \
        const nlohmann::json& item, std::string base,           \
        std::shared_ptr<const Executor> exec,                   \
        std::shared_ptr<const LinOp> linop, ResourceManager* manager)


DECLARE_SELECTION(Executor);
DECLARE_SELECTION(LinOp);
DECLARE_SELECTION(LinOpFactory);
DECLARE_SELECTION(CriterionFactory);
DECLARE_SELECTION(Logger);


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
     * @param item  the const nlohmann::json
     * @param exec  the Executor from outside
     * @param linop  the LinOp from outside
     * @param manager  the ResourceManager pointer
     */
    static type build(const nlohmann::json& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager);
};

/**
 * GENERIC_BASE_IMPL is a implementaion of Generic for base type
 *
 * @param _base_type  the base type
 */
#define GENERIC_BASE_IMPL(_base_type)                                       \
    template <>                                                             \
    struct Generic<_base_type> {                                            \
        using type = std::shared_ptr<_base_type>;                           \
        static type build(const nlohmann::json& item,                       \
                          std::shared_ptr<const Executor> exec,             \
                          std::shared_ptr<const LinOp> linop,               \
                          ResourceManager* manager)                         \
        {                                                                   \
            assert(item.contains("base"));                                  \
            std::cout << "build base" << item.at("base").get<std::string>() \
                      << " "                                                \
                      << get_base_class(item.at("base").get<std::string>()) \
                      << std::endl;                                         \
            return create_from_config<_base_type>(                          \
                item, get_base_class(item.at("base").get<std::string>()),   \
                exec, linop, manager);                                      \
        }                                                                   \
    }

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
    static type build(const nlohmann::json& item,
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
    static type build(const nlohmann::json& item,
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
 * @param item  the const nlohmann::json
 * @param exec  the Executor from outside
 * @param linop  the LinOp from outside
 * @param manager  the ResourceManager pointer
 */
template <typename T>
std::shared_ptr<T> create_from_config(
    const nlohmann::json& item, std::shared_ptr<const Executor> exec = nullptr,
    std::shared_ptr<const LinOp> linop = nullptr,
    ResourceManager* manager = nullptr)
{
    std::cout << "create_from_config directly to type" << std::endl;
    return GenericHelper<T>::build(item, exec, linop, manager);
}


/**
 * create_from_config is another overloading to implement the function after
 * selection on enum map. This is the major implementation to select different
 * template type from base class.
 *
 * @tparam T  the enum type
 * @tparam base  the enum item
 * @tparam U  the corresponding base type of the enum type
 */
template <typename T, T base, typename U = typename gkobase<T>::type>
std::shared_ptr<U> create_from_config(const nlohmann::json& item,
                                      std::shared_ptr<const Executor> exec,
                                      std::shared_ptr<const LinOp> linop,
                                      ResourceManager* manager);

// If the template does not contain definition, we do not need to declare
// everything but the "file_config/file_config.hpp" needs to be after user
// implementation ENUM_BRIDGE(ENUM_EXECUTER, DECLARE_BRIDGE_EXECUTOR);
// ENUM_BRIDGE(ENUM_LINOP, DECLARE_BRIDGE_LINOP);


#define IMPLEMENT_EMPTY_BRIDGE(_enum_type, _enum_item)                       \
    template <>                                                              \
    inline std::shared_ptr<typename gkobase<_enum_type>::type>               \
    create_from_config<_enum_type, _enum_type::_enum_item,                   \
                       typename gkobase<_enum_type>::type>(                  \
        const nlohmann::json& item, std::shared_ptr<const Executor> exec,    \
        std::shared_ptr<const LinOp> linop, ResourceManager* manager)        \
    {                                                                        \
        std::cout << "enter empty" << std::endl;                             \
        return nullptr;                                                      \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


}  // namespace file_config
}  // namespace extensions
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_FILE_CONFIG_BASE_GENERIC_CONSTRUCTOR_HPP_
