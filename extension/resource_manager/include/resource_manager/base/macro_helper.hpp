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

#ifndef GKOEXT_RESOURCE_MANAGER_BASE_MACRO_HELPER_HPP_
#define GKOEXT_RESOURCE_MANAGER_BASE_MACRO_HELPER_HPP_


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


/**
 * ENUM_CLASS is a macro to generate the actual enum class.
 *
 * @param _enum_type  the name of enum class type
 * @param _type  the base type of enum
 * @param _list  the list of enum items, which is built by ENUM_VALUE.
 */
#define ENUM_CLASS(_enum_type, _type, _list) \
    enum class _enum_type : _type { _list(ENUM_VALUE) }


/**
 * ENUM_LAMBDA_ is a macro to generate the item of enum map, which
 * calls the predefined function create_from_config
 *
 * @param _name  the name of enum item
 */
// clang-format off
#define ENUM_LAMBDA_(_name)                                                   \
    {                                                                         \
        #_name,                                                               \
            [&](rapidjson::Value &item, std::shared_ptr<const Executor> exec, \
                std::shared_ptr<const LinOp> linop,                           \
                ResourceManager *manager) {                                   \
                return create_from_config<enum_type_alias,                    \
                                          enum_type_alias::_name>(            \
                    item, exec, linop, manager);                              \
            }                                                                 \
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
#define ENUM_MAP(_name, _enum_type, _return_type, _list, _keyword)             \
    using enum_type_alias = _enum_type;                                        \
    _keyword std::map<std::string,                                             \
                      std::function<_return_type(                              \
                          rapidjson::Value &, std::shared_ptr<const Executor>, \
                          std::shared_ptr<const LinOp>, ResourceManager *)>>   \
        _name                                                                  \
    {                                                                          \
        _list(ENUM_LAMBDA)                                                     \
    }

/**
 * DECLARE_SELECTION is a macro helper to declare the selection overloading
 * function.
 *
 * @param _base_type  the base type
 * @param _enum_type  the corresponding enum type
 */
#define DECLARE_SELECTION(_base_type, _enum_type)                      \
    std::shared_ptr<_base_type> create_from_config_(                   \
        _enum_type, rapidjson::Value &, std::string,                   \
        std::shared_ptr<const Executor>, std::shared_ptr<const LinOp>, \
        ResourceManager *)

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
    std::shared_ptr<_base_type> create_from_config_(                          \
        _enum_type, rapidjson::Value &item, std::string base,                 \
        std::shared_ptr<const Executor> exec,                                 \
        std::shared_ptr<const LinOp> linop, ResourceManager *manager)         \
    {                                                                         \
        std::cout << "search on enum" << std::endl;                           \
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
        rapidjson::Value & item, std::shared_ptr<const Executor> exec,       \
        std::shared_ptr<const LinOp> linop, ResourceManager * manager)       \
    {                                                                        \
        std::cout << "enter bridge" << std::endl;                            \
        return call<_impl_type>(item, exec, linop, manager);                 \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

/**
 * BUILD_FACTORY is to implement the beginning of setting factory. It sets the
 * several alias to following usage and get the executor, which can be used the
 * subitem.
 *
 * @param _type  the outer type of factory
 * @param _manager  the outside parameter name of ResourceManager
 * @param _item  the outside parameter name of RapidJson Value&
 * @param _linop  the outside parameter name of LinOp
 * @param _exec  the outside parameter name of Executor
 *
 * @note This should be in a lambda function. It generates `factory_alias_` from
 *       `_type` factory, `exec_alias` from `_exec` and `_item` and alias name
 *       `*_alias_` from `_manager`, `_item`, `_linop`.
 */
#define BUILD_FACTORY(_type, _manager, _item, _exec, _linop) \
    auto factory_alias_ = _type::build();                    \
    auto &manager_alias_ = _manager;                         \
    auto &item_alias_ = _item;                               \
    auto &linop_alias_ = _linop;                             \
    auto exec_alias_ =                                       \
        get_pointer_check<Executor>(_item, "exec", _exec, _linop, _manager)

/**
 * SET_POINTER is to set one pointer for the factory. It is for the
 * `std::shared_ptr<type> GKO_FACTORY_PARAMETER_SCALAR(name, initial)`. It only
 * sets the pointer when the RapidJson contain the value. The corresponding call
 * should be `SET_POINTER(type, name);`
 *
 * @note This should be used in the of a lambda function.
 */
#define SET_POINTER(_param_type, _param_name)                                \
    if (item_alias_.HasMember(#_param_name)) {                               \
        factory_alias_.with_##_param_name(                                   \
            get_pointer<_param_type>(item_alias_[#_param_name], exec_alias_, \
                                     linop_alias_, manager_alias_));         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * SET_POINTER_VECTOR is to set pointer array for the factory. It is for the
 * `std::vector<std::shared_ptr<type>> GKO_FACTORY_PARAMETER_VECTOR(name,
 * initial)`. It only sets the pointer array when the RapidJson contain the
 * value. The corresponding call should be `SET_POINTER_VECTOR(type, name);`
 *
 * @note This should be used in the of a lambda function.
 */
#define SET_POINTER_VECTOR(_param_type, _param_name)                         \
    if (item_alias_.HasMember(#_param_name)) {                               \
        std::cout << exec_alias_.get() << std::endl;                         \
        factory_alias_.with_##_param_name(get_pointer_vector<_param_type>(   \
            item_alias_[#_param_name], exec_alias_, linop_alias_,            \
            manager_alias_));                                                \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * SET_VALUE is to set one value for the factory. It is for the
 * `type GKO_FACTORY_PARAMETER_SCALAR(name, initial)`. It only
 * sets the value when the RapidJson contain the value. The corresponding call
 * should be `SET_VALUE(type, name);`
 *
 * @note This should be used in the of a lambda function.
 */
#define SET_VALUE(_param_type, _param_name)                                  \
    if (item_alias_.HasMember(#_param_name)) {                               \
        std::string name{#_param_name};                                      \
        factory_alias_.with_##_param_name(                                   \
            get_value<_param_type>(item_alias_, name));                      \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * SET_EXECUTOR is to set the executor of factory and return the factory type
 * from this operation.
 *
 * @note This should be used in the end of a lambda function.
 */
#define SET_EXECUTOR return factory_alias_.on(exec_alias_)


/**
 * CREATE_DEFAULT_IMPL is to create a specialization on create_from_config such
 * that call the overloading selection function on base type.
 *
 * @param _base_type  the base type
 */
#define CREATE_DEFAULT_IMPL(_base_type)                                      \
    template <>                                                              \
    std::shared_ptr<_base_type> create_from_config<_base_type>(              \
        rapidjson::Value & item, std::string base,                           \
        std::shared_ptr<const Executor> exec,                                \
        std::shared_ptr<const LinOp> linop, ResourceManager * manager)       \
    {                                                                        \
        return create_from_config_(RM_##_base_type::_base_type, item, base,  \
                                   exec, linop, manager);                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

/**
 * PACK is to pack several args for one parameter
 */
#define PACK(...) __VA_ARGS__


/**
 * GENERIC_BASE_IMPL is a implementaion of Generic for base type
 *
 * @param _base_type  the base type
 */
#define GENERIC_BASE_IMPL(_base_type)                                  \
    template <>                                                        \
    struct Generic<_base_type> {                                       \
        using type = std::shared_ptr<_base_type>;                      \
        static type build(rapidjson::Value &item,                      \
                          std::shared_ptr<const Executor> exec,        \
                          std::shared_ptr<const LinOp> linop,          \
                          ResourceManager *manager)                    \
        {                                                              \
            assert(item.HasMember("base"));                            \
            return create_from_config<_base_type>(                     \
                item, item["base"].GetString(), exec, linop, manager); \
        }                                                              \
    }

/**
 * SIMPLE_LINOP_FACTORY_IMPL is a implementation for those LinOp, taking the
 * factory and generating the LinOp with matrix.
 *
 * @param _base  the LinOp type base without the template parameter
 * @param _template  the template parameter for LinOp type base
 * @param _type  the template type usage from `_template`
 *
 * @note Use PACK to pack more than one args for `_template` and `_type`.
 */
#define SIMPLE_LINOP_WITH_FACTORY_IMPL(_base, _template, _type)             \
    template <_template>                                                    \
    struct Generic<_base<_type>> {                                          \
        using type = std::shared_ptr<_base<_type>>;                         \
        static type build(rapidjson::Value &item,                           \
                          std::shared_ptr<const Executor> exec,             \
                          std::shared_ptr<const LinOp> linop,               \
                          ResourceManager *manager)                         \
        {                                                                   \
            std::cout << #_base << exec.get() << std::endl;                 \
            auto factory = get_pointer<typename _base<_type>::Factory>(     \
                item["factory"], exec, linop, manager);                     \
            auto mtx =                                                      \
                get_pointer<LinOp>(item["generate"], exec, linop, manager); \
            auto ptr = factory->generate(mtx);                              \
            return std::move(ptr);                                          \
        }                                                                   \
    }


#endif  // GKOEXT_RESOURCE_MANAGER_BASE_MACRO_HELPER_HPP_
