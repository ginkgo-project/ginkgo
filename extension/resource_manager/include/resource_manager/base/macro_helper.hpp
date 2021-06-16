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
#define ENUM_LAMBDA_(_name)                                            \
    {                                                                  \
        #_name, [&](arg_type item) {                                   \
            return this->build_item<enum_type, enum_type::_name>(item); \
        }                                                              \
    }

#define ENUM_LAMBDA2_(_name)                                             \
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

#define ENUM_LAMBDA2_ASSIGN_(_name, _assign) ENUM_LAMBDA2_(_name)

#define ENUM_LAMBDA(...)                                             \
    UNPACK_VA_ARGS(MACRO_OVERLOAD_(__VA_ARGS__, ENUM_LAMBDA_ASSIGN_, \
                                   ENUM_LAMBDA_, UNUSED)(__VA_ARGS__))


#define ENUM_LAMBDA2(...)                                             \
    UNPACK_VA_ARGS(MACRO_OVERLOAD_(__VA_ARGS__, ENUM_LAMBDA2_ASSIGN_, \
                                   ENUM_LAMBDA2_, UNUSED)(__VA_ARGS__))

#define ENUM_CLASS(_enum_type, _type, _list) \
    enum class _enum_type : _type { _list(ENUM_VALUE) }

#define ENUM_MAP(_name, _enum_type, _return_type, _arg_type, _list, _keyword) \
    using enum_type = _enum_type;                                             \
    using arg_type = _arg_type;                                               \
    _keyword std::map<std::string, std::function<_return_type(arg_type)>>     \
        _name                                                                 \
    {                                                                         \
        _list(ENUM_LAMBDA)                                                    \
    }

#define ENUM_MAP2(_name, _enum_type, _return_type, _arg_type, _list, _keyword) \
    using enum_type = _enum_type;                                              \
    using arg_type = _arg_type;                                                \
    _keyword std::map<std::string,                                             \
                      std::function<_return_type(                              \
                          arg_type, std::shared_ptr<const Executor>,           \
                          std::shared_ptr<const LinOp>, ResourceManager *)>>   \
        _name                                                                  \
    {                                                                          \
        _list(ENUM_LAMBDA2)                                                    \
    }


#define DECLARE_BASE_BUILD_ITEM(_base_type, _enum_type)                    \
    std::shared_ptr<_base_type> build_item_impl(_enum_type, std::string &, \
                                                rapidjson::Value &)

#define IMPLEMENT_BASE_BUILD_ITEM_IMPL(_base_type, _enum_type, _list)         \
    std::shared_ptr<_base_type> ResourceManager::build_item_impl(             \
        _enum_type, std::string &base, rapidjson::Value &item)                \
    {                                                                         \
        ENUM_MAP(_base_type##Select, _enum_type, std::shared_ptr<_base_type>, \
                 rapidjson::Value &, _list, static);                          \
        auto it = _base_type##Select.find(base);                              \
        if (it == _base_type##Select.end()) {                                 \
            return nullptr;                                                   \
        } else {                                                              \
            std::cout << "Found!" << std::endl;                               \
            return it->second(item);                                          \
        }                                                                     \
    }

#define IMPLEMENT_SELECTION(_base_type, _enum_type, _list)                     \
    template <>                                                                \
    std::shared_ptr<_base_type> create_from_config_<_base_type>(               \
        rapidjson::Value & item, std::string base,                             \
        std::shared_ptr<const Executor> exec,                                  \
        std::shared_ptr<const LinOp> linop, ResourceManager * manager)         \
    {                                                                          \
        std::cout << "search on enum" << std::endl;                            \
        ENUM_MAP2(_base_type##Select, _enum_type, std::shared_ptr<_base_type>, \
                  rapidjson::Value &, _list, static);                          \
        auto it = _base_type##Select.find(base);                               \
        if (it == _base_type##Select.end()) {                                  \
            std::cout << "Not Found" << std::endl;                             \
            return nullptr;                                                    \
        } else {                                                               \
            std::cout << "Found!" << std::endl;                                \
            return it->second(item, exec, linop, manager);                     \
        }                                                                      \
    }

#define IMPLEMENT_BASE_BUILD_ITEM(_base_type, _enum_base_item)           \
    template <>                                                          \
    std::shared_ptr<_base_type> ResourceManager::build_item<_base_type>( \
        std::string & base, rapidjson::Value & item)                     \
    {                                                                    \
        return this->build_item_impl(_enum_base_item, base, item);       \
    }


#define IMPLEMENT_BRIDGE(_enum_type, _enum_item, _impl_type)             \
    template <>                                                          \
    std::shared_ptr<typename gkobase<_enum_type>::type>                  \
        ResourceManager::build_item<_enum_type, _enum_type::_enum_item,  \
                                    typename gkobase<_enum_type>::type>( \
            rapidjson::Value & item)                                     \
    {                                                                    \
        return this->build_item<_impl_type>(item);                       \
    }

#define IMPLEMENT_BRIDGE2(_enum_type, _enum_item, _impl_type)                  \
    template <>                                                                \
    std::shared_ptr<typename gkobase<_enum_type>::type>                        \
    create_from_config<_enum_type, _enum_type::_enum_item,                     \
                       typename gkobase<_enum_type>::type>(                    \
        rapidjson::Value & item, std::shared_ptr<const Executor> exec,         \
        std::shared_ptr<const LinOp> linop, ResourceManager * manager)         \
    {                                                                          \
        if (manager == nullptr) {                                              \
            return create_from_config<_impl_type>(item, exec, linop, manager); \
        } else {                                                               \
            return manager->build_item<_impl_type>(item);                      \
        }                                                                      \
    }

#define IMPLEMENT_TINY_BRIDGE(_enum_type, _enum_item, _impl_type)          \
    template <>                                                            \
    std::shared_ptr<_impl_type> ResourceManager::build_item<               \
        _enum_type, _enum_type::_enum_item, _impl_type>(rapidjson::Value & \
                                                        item)              \
    {                                                                      \
        return this->build_item<_impl_type>(item);                         \
    }

// clang-format off
#define BUILD_FACTORY(type_, rm_, item_, exec_, linop_)     \
    [&]() {                                   \
        auto factory_alias_ = type_::build(); \
        auto &rm_alias_ = rm_;               \
        auto &item_alias_ = item_;\
        auto &exec_alias_ = exec_;\
        auto &linop_alias_ = linop_;

#define ON_EXECUTOR                                                \
        auto executor =                                            \
            get_pointer<Executor>(rm_alias_, item_alias_["exec"], exec_alias_, linop_alias_); \
        return factory_alias_.on(executor);                        \
    }                                                              \
    ();
// clang-format on


#define WITH_POINTER(_param_type, _param_name)                                 \
    if (item_alias_.HasMember(#_param_name)) {                                 \
        factory_alias_.with_##_param_name(get_pointer<_param_type>(            \
            rm_alias_, item_alias_[#_param_name], exec_alias_, linop_alias_)); \
    }

#define WITH_POINTER_ARRAY(_param_type, _param_name)                           \
    if (item_alias_.HasMember(#_param_name)) {                                 \
        factory_alias_.with_##_param_name(get_pointer_vector<_param_type>(     \
            rm_alias_, item_alias_[#_param_name], exec_alias_, linop_alias_)); \
    }

#define WITH_VALUE(_param_type, _param_name)            \
    if (item_alias_.HasMember(#_param_name)) {          \
        std::string name{#_param_name};                 \
        factory_alias_.with_##_param_name(              \
            get_value<_param_type>(item_alias_, name)); \
    }

#endif  // GKOEXT_RESOURCE_MANAGER_BASE_MACRO_HELPER_HPP_
