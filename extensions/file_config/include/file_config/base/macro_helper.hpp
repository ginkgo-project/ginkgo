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

#ifndef GKO_PUBLIC_EXT_FILE_CONFIG_BASE_MACRO_HELPER_HPP_
#define GKO_PUBLIC_EXT_FILE_CONFIG_BASE_MACRO_HELPER_HPP_


// MSVC tends to use __VA_ARGS__ as one item
// To expand __VA_ARGS__ as other compilers, another workaround is to use the
// `/Zc:preprocessor` or `/experimental:preprocessor`
#define UNPACK_VA_ARGS(_x) _x

/**
 * MACRO_OVERLOAD_ is a macro helper to overload two possible macros.
 */
#define MACRO_OVERLOAD_(_1, _2, _NAME, ...) _NAME

/**
 * ENUM_VALUE_ is a macro for enum to define the name only
 *
 * @param _name  the name of enum item
 */
#define ENUM_VALUE_(_name) _name

/**
 * ENUM_VALUE_ASSIGN_ is a macro for enum to define the name and the assigned
 * value
 *
 * @param _name  the name of enum item
 * @param _assign  the assigned value of enum item
 */
#define ENUM_VALUE_ASSIGN_(_name, _assign) _name = _assign

/**
 * ENUM_VALUE is a macro for enum with overloading. It can be used as
 * ENUM_VALUE(name) or ENUM_VALUE(name, assign).
 *
 * @param _name  the name of enum item
 * @param _assign  (optional) the assigned value of enum item
 */
#define ENUM_VALUE(...)                                             \
    UNPACK_VA_ARGS(MACRO_OVERLOAD_(__VA_ARGS__, ENUM_VALUE_ASSIGN_, \
                                   ENUM_VALUE_, UNUSED)(__VA_ARGS__))

// Avoid , to be preprocessed first
#define COMMA_() ,
#define EMPTY()
#define POSTPONE(_sep) _sep EMPTY()
#define COMMA POSTPONE(COMMA_)()
/**
 * ENUM_CLASS is a macro to generate the actual enum class.
 *
 * @param _enum_type  the name of enum class type
 * @param _type  the base type of enum
 * @param _list  the list of enum items, which is built by ENUM_VALUE.
 */
#define ENUM_CLASS(_enum_type, _type, _list) \
    enum class _enum_type : _type { _list(ENUM_VALUE, COMMA) }


/**
 * ENUM_LAMBDA_ is a macro to generate the item of enum map, which
 * calls the predefined function create_from_config
 *
 * @param _name  the name of enum item
 */
// clang-format off
#define ENUM_LAMBDA_(_name)                                                  \
    {                                                                        \
        #_name,                                                              \
        [&](const nlohmann::json&item, std::shared_ptr<const Executor> exec, \
            std::shared_ptr<const LinOp> linop,                              \
            ResourceManager *manager) {                                      \
            std::cout << #_name << std::endl;                                \
            return create_from_config<enum_type_alias,                       \
                                        enum_type_alias::_name>(             \
                item, exec, linop, manager);                                 \
            }                                                                \
    }
// clang-format on

/**
 * ENUM_LAMBDA_ASSIGN_ is a macro to accept the enum item with assigned value.
 * It will forward the name to ENUM_LAMBDA_.
 *
 * @param _name  the name of enum item
 * @param _assign  (not used) the assigned value of enum item
 */
#define ENUM_LAMBDA_ASSIGN_(_name, _assign) ENUM_LAMBDA_(_name)

/**
 * ENUM_LAMBDA is a macro to generate the item of enum map with overloading. It
 * can be used as ENUM_LAMBDA(name) or ENUM_LAMBDA(name, assign).
 *
 * @param _name  the name of enum item
 * @param _assign  (optional, not used) the assigned value of enum item
 */
#define ENUM_LAMBDA(...)                                             \
    UNPACK_VA_ARGS(MACRO_OVERLOAD_(__VA_ARGS__, ENUM_LAMBDA_ASSIGN_, \
                                   ENUM_LAMBDA_, UNUSED)(__VA_ARGS__))

/**
 * ENUM_MAP is a macro to generate the actual enum map, which uses ENUM_LAMBDA
 * to build each item.
 *
 * @param _name the name of map
 * @param _enum_type  the name of enum class type
 * @param _return_type  the return type of functions in the map
 * @param _list  the list of enum items, which is built by ENUM_VALUE
 * @param _keyword  the keyword before map declaration
 *
 * @note it uses enum_type_alias to make ENUM_LAMBDA use correct enum_type.
 */
#define ENUM_MAP(_name, _enum_type, _return_type, _list, _keyword)           \
    using enum_type_alias = _enum_type;                                      \
    _keyword                                                                 \
        std::map<std::string,                                                \
                 std::function<_return_type(                                 \
                     const nlohmann::json&, std::shared_ptr<const Executor>, \
                     std::shared_ptr<const LinOp>, ResourceManager*)>>       \
            _name                                                            \
    {                                                                        \
        _list(ENUM_LAMBDA, COMMA)                                            \
    }

/**
 * IMPLEMENT_SELECTION is a macro helper to implement selection from
 * DECLARE_SELECTION. If find a match, use the matched function call, or return
 * nullptr.
 *
 * @param _base_type  the base type
 * @param _enum_type  the corresponding enum type
 * @param _list  the list of enum items, which is built by ENUM_VALUE
 */
#define IMPLEMENT_SELECTION(_base_type, _enum_type, _list)                    \
    template <>                                                               \
    std::shared_ptr<_base_type> create_from_config<_base_type>(               \
        const nlohmann::json& item, std::string base,                         \
        std::shared_ptr<const Executor> exec,                                 \
        std::shared_ptr<const LinOp> linop, ResourceManager* manager)         \
    {                                                                         \
        std::cout << "search on enum " << base << std::endl;                  \
        ENUM_MAP(_base_type##Select, _enum_type, std::shared_ptr<_base_type>, \
                 _list, static);                                              \
        auto it = _base_type##Select.find(base);                              \
        if (it == _base_type##Select.end()) {                                 \
            std::cout << "Not Found" << std::endl;                            \
            return nullptr;                                                   \
        } else {                                                              \
            std::cout << "Found!" << std::endl;                               \
            return it->second(item, exec, linop, manager);                    \
        }                                                                     \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

/**
 * IMPLEMENT_BRIDGE is a macro helper to implement bridge between
 * `create_from_config<_enum_type, _enum_item>` to `call<_impl_type>`. It can be
 * used when the `_impl_type` does not need furthermore selection such as
 * ValueType or something else.
 *
 * @param _enum_type  the enum type
 * @param _enum_item  the enum item of `_enum_type`
 * @param _impl_type  the implementation type, which is used in
 *                    `call<_impl_type>(...)`
 */
#define IMPLEMENT_BRIDGE(_enum_type, _enum_item, _impl_type)                 \
    template <>                                                              \
    std::shared_ptr<typename gkobase<_enum_type>::type>                      \
    create_from_config<_enum_type, _enum_type::_enum_item,                   \
                       typename gkobase<_enum_type>::type>(                  \
        const nlohmann::json& item, std::shared_ptr<const Executor> exec,    \
        std::shared_ptr<const LinOp> linop, ResourceManager* manager)        \
    {                                                                        \
        std::cout << "enter bridge" << std::endl;                            \
        return call<_impl_type>(item, exec, linop, manager);                 \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define DECLARE_BRIDGE_EXECUTOR_(_enum_item)                              \
    template <>                                                           \
    std::shared_ptr<typename gkobase<RM_Executor>::type>                  \
    create_from_config<RM_Executor, RM_Executor::_enum_item,              \
                       typename gkobase<RM_Executor>::type>(              \
        const nlohmann::json& item, std::shared_ptr<const Executor> exec, \
        std::shared_ptr<const LinOp> linop, ResourceManager* manager)

#define DECLARE_BRIDGE_EXECUTOR_ASSIGN_(_name, _assign) \
    DECLARE_BRIDGE_EXECUTOR_(_name)

/**
 * ENUM_LAMBDA is a macro to generate the item of enum map with overloading. It
 * can be used as ENUM_LAMBDA(name) or ENUM_LAMBDA(name, assign).
 *
 * @param _name  the name of enum item
 * @param _assign  (optional, not used) the assigned value of enum item
 */
#define DECLARE_BRIDGE_EXECUTOR(...)                                  \
    UNPACK_VA_ARGS(                                                   \
        MACRO_OVERLOAD_(__VA_ARGS__, DECLARE_BRIDGE_EXECUTOR_ASSIGN_, \
                        DECLARE_BRIDGE_EXECUTOR_, UNUSED)(__VA_ARGS__))

#define DECLARE_BRIDGE_LINOP(_enum_item)                                   \
    template <>                                                            \
    std::shared_ptr<typename gkobase<RM_LinOp>::type> create_from_config<  \
        RM_LinOp, RM_LinOp::_enum_item, typename gkobase<RM_LinOp>::type>( \
        const nlohmann::json& item, std::shared_ptr<const Executor> exec,  \
        std::shared_ptr<const LinOp> linop, ResourceManager* manager)


#define ENUM_BRIDGE(_list, _declare) _list(_declare, ;)


#endif  // GKO_PUBLIC_EXT_FILE_CONFIG_BASE_MACRO_HELPER_HPP_
