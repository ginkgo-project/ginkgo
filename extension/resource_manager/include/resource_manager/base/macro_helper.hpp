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
#define ENUM_VALUE_(_name) _name
#define ENUM_VALUE_ASSIGN_(_name, _assign) _name = _assign
#define MACRO_OVERLOAD_(_1, _2, _NAME, ...) _NAME
#define ENUM_VALUE(...)                                             \
    UNPACK_VA_ARGS(MACRO_OVERLOAD_(__VA_ARGS__, ENUM_VALUE_ASSIGN_, \
                                   ENUM_VALUE_, UNUSED)(__VA_ARGS__))


// clang-format off
#define ENUM_LAMBDA_(_name)                                             \
    {                                                                    \
        #_name, [&](arg_type item, std::shared_ptr<const Executor> exec, \
                    std::shared_ptr<const LinOp> linop,                  \
                    ResourceManager *manager) {                          \
            return create_from_config<enum_type, enum_type::_name>(      \
                item, exec, linop, manager);                             \
        }                                                                \
    }
// clang-format on

#define ENUM_LAMBDA_ASSIGN_(_name, _assign) ENUM_LAMBDA_(_name)

#define ENUM_LAMBDA(...)                                             \
    UNPACK_VA_ARGS(MACRO_OVERLOAD_(__VA_ARGS__, ENUM_LAMBDA_ASSIGN_, \
                                   ENUM_LAMBDA_, UNUSED)(__VA_ARGS__))

#define ENUM_CLASS(_enum_type, _type, _list) \
    enum class _enum_type : _type { _list(ENUM_VALUE) }


#define ENUM_MAP(_name, _enum_type, _return_type, _arg_type, _list, _keyword) \
    using enum_type = _enum_type;                                             \
    using arg_type = _arg_type;                                               \
    _keyword std::map<std::string,                                            \
                      std::function<_return_type(                             \
                          arg_type, std::shared_ptr<const Executor>,          \
                          std::shared_ptr<const LinOp>, ResourceManager *)>>  \
        _name                                                                 \
    {                                                                         \
        _list(ENUM_LAMBDA)                                                    \
    }

#define DECLARE_SELECTION(_base_type, _enum_type)                      \
    std::shared_ptr<_base_type> create_from_config_(                   \
        _enum_type, rapidjson::Value &, std::string,                   \
        std::shared_ptr<const Executor>, std::shared_ptr<const LinOp>, \
        ResourceManager *)

#define IMPLEMENT_SELECTION(_base_type, _enum_type, _list)                    \
    std::shared_ptr<_base_type> create_from_config_(                          \
        _enum_type, rapidjson::Value &item, std::string base,                 \
        std::shared_ptr<const Executor> exec,                                 \
        std::shared_ptr<const LinOp> linop, ResourceManager *manager)         \
    {                                                                         \
        std::cout << "search on enum" << std::endl;                           \
        ENUM_MAP(_base_type##Select, _enum_type, std::shared_ptr<_base_type>, \
                 rapidjson::Value &, _list, static);                          \
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


#define BUILD_FACTORY(type_, rm_, item_, exec_, linop_) \
    auto factory_alias_ = type_::build();               \
    auto &rm_alias_ = rm_;                              \
    auto &item_alias_ = item_;                          \
    auto &exec_alias_ = exec_;                          \
    auto &linop_alias_ = linop_

#define SET_EXECUTOR                                                \
    auto executor = get_pointer_check<Executor>(                    \
        rm_alias_, item_alias_, "exec", exec_alias_, linop_alias_); \
    return factory_alias_.on(executor)


#define SET_POINTER(_param_type, _param_name)                                  \
    if (item_alias_.HasMember(#_param_name)) {                                 \
        factory_alias_.with_##_param_name(get_pointer<_param_type>(            \
            rm_alias_, item_alias_[#_param_name], exec_alias_, linop_alias_)); \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define SET_POINTER_ARRAY(_param_type, _param_name)                            \
    if (item_alias_.HasMember(#_param_name)) {                                 \
        factory_alias_.with_##_param_name(get_pointer_vector<_param_type>(     \
            rm_alias_, item_alias_[#_param_name], exec_alias_, linop_alias_)); \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define SET_VALUE(_param_type, _param_name)                                  \
    if (item_alias_.HasMember(#_param_name)) {                               \
        std::string name{#_param_name};                                      \
        factory_alias_.with_##_param_name(                                   \
            get_value<_param_type>(item_alias_, name));                      \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define CONNECT_GENERIC_SUB(base, T, inner_type, func)              \
    template <>                                                     \
    struct Generic<typename base<T>::inner_type> {                  \
        using type = std::shared_ptr<typename base<T>::inner_type>; \
        static type build(rapidjson::Value &item,                   \
                          std::shared_ptr<const Executor> exec,     \
                          std::shared_ptr<const LinOp> linop,       \
                          ResourceManager *manager)                 \
        {                                                           \
            return func<T>(item, exec, linop, manager);             \
        }                                                           \
    }

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

#define PACK(...) __VA_ARGS__

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
                manager, item["factory"], exec, linop);                     \
            auto mtx =                                                      \
                get_pointer<LinOp>(manager, item["generate"], exec, linop); \
            auto ptr = factory->generate(mtx);                              \
            return std::move(ptr);                                          \
        }                                                                   \
    }


#endif  // GKOEXT_RESOURCE_MANAGER_BASE_MACRO_HELPER_HPP_
