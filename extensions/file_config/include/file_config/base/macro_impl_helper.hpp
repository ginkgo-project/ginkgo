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

#ifndef GKO_PUBLIC_EXT_FILE_CONFIG_BASE_MACRO_IMPL_HELPER_HPP_
#define GKO_PUBLIC_EXT_FILE_CONFIG_BASE_MACRO_IMPL_HELPER_HPP_


/**
 * PACK is to pack several args for one parameter
 */
#define PACK(...) __VA_ARGS__


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
#define BUILD_FACTORY(_type, _manager, _item, _exec, _linop)                   \
    auto factory_alias_ = _type::build();                                      \
    auto& manager_alias_ = _manager;                                           \
    auto& item_alias_ = _item;                                                 \
    auto& linop_alias_ = _linop;                                               \
    auto exec_alias_ = get_pointer_check<const Executor>(_item, "exec", _exec, \
                                                         _linop, _manager)

/**
 * SET_POINTER is to set one pointer for the factory. It is for the
 * `std::shared_ptr<type> GKO_FACTORY_PARAMETER_SCALAR(name, initial)`. It only
 * sets the pointer when the RapidJson contain the value. The corresponding call
 * should be `SET_POINTER(type, name);`
 *
 * @note This should be used in the of a lambda function.
 */
#define SET_POINTER(_param_type, _param_name)                                \
    if (item_alias_.contains(#_param_name)) {                                \
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
    if (item_alias_.contains(#_param_name)) {                                \
        std::cout << "pointer_vector executor " << exec_alias_.get()         \
                  << std::endl;                                              \
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
    if (item_alias_.contains(#_param_name)) {                                \
        std::string name{#_param_name};                                      \
        factory_alias_.with_##_param_name(                                   \
            get_value<_param_type>(item_alias_, #_param_name));              \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define SET_ARRAY(_param_type, _param_name)                                  \
    if (item_alias_.contains(#_param_name)) {                                \
        std::string name{#_param_name};                                      \
        factory_alias_.with_##_param_name(                                   \
            get_array<_param_type>(item_alias_, #_param_name, exec_alias_)); \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define SET_FUNCTION(_param_type, _param_name)                               \
    if (item_alias_.contains(#_param_name)) {                                \
        std::string name{#_param_name};                                      \
        factory_alias_.with_##_param_name(                                   \
            _param_name##_map[get_value<std::string>(item_alias_,            \
                                                     #_param_name)]);        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define SET_CSR_STRATEGY(_param_type, _param_name)                            \
    if (item_alias_.contains(#_param_name)) {                                 \
        std::string name{#_param_name};                                       \
        factory_alias_.with_##_param_name(get_csr_strategy<_param_type>(      \
            get_value<std::string>(item_alias_, #_param_name), exec_alias_)); \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")


/**
 * SET_EXECUTOR is to set the executor of factory and return the factory type
 * from this operation.
 *
 * @note This should be used in the end of a lambda function.
 */
#define SET_EXECUTOR return factory_alias_.on(exec_alias_)

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
#define SIMPLE_LINOP_WITH_FACTORY_IMPL(_base, _template, _type)                \
    template <_template>                                                       \
    struct Generic<_base<_type>> {                                             \
        using type = std::shared_ptr<_base<_type>>;                            \
        static type build(const nlohmann::json& item,                          \
                          std::shared_ptr<const Executor> exec,                \
                          std::shared_ptr<const LinOp> linop,                  \
                          ResourceManager* manager)                            \
        {                                                                      \
            std::cout << #_base << exec.get() << std::endl;                    \
            auto factory = get_pointer<typename _base<_type>::Factory>(        \
                item["factory"], exec, linop, manager);                        \
            auto mtx = get_pointer<const LinOp>(item["generate"], exec, linop, \
                                                manager);                      \
            auto ptr = factory->generate(mtx);                                 \
            add_logger(ptr, item, exec, linop, manager);                       \
            return std::move(ptr);                                             \
        }                                                                      \
    }

#define SIMPLE_LINOP_WITH_FACTORY_IMPL_BASE(_base)                             \
    template <>                                                                \
    struct Generic<_base> {                                                    \
        using type = std::shared_ptr<_base>;                                   \
        static type build(const nlohmann::json& item,                          \
                          std::shared_ptr<const Executor> exec,                \
                          std::shared_ptr<const LinOp> linop,                  \
                          ResourceManager* manager)                            \
        {                                                                      \
            std::cout << #_base << exec.get() << std::endl;                    \
            auto factory = get_pointer<typename _base::Factory>(               \
                item["factory"], exec, linop, manager);                        \
            auto mtx = get_pointer<const LinOp>(item["generate"], exec, linop, \
                                                manager);                      \
            auto ptr = factory->generate(mtx);                                 \
            return std::move(ptr);                                             \
        }                                                                      \
    }


/**
 * ENABLE_SELECTION is to build a template selection on the given tt_list. It
 * will take each item (single type or type_list) of tt_list and the
 * corresponding identifier string. If the string is accepted by the Predicate,
 * it will launch the function with the accepted type.
 *
 * @param _name  the selection function name
 * @param _callable  the function to launch
 * @param _return  the return type of the function (pointer)
 * @param _get_type  the method to get the type (get_actual_type or
 *                   get_actual_factory_type)
 */
#define ENABLE_SELECTION(_name, _callable, _return, _get_type)                 \
    template <template <typename...> class Base, typename Predicate,           \
              typename... InferredArgs>                                        \
    _return _name(tt_list<>, Predicate is_eligible,                            \
                  const nlohmann::json& item, InferredArgs... args)            \
    {                                                                          \
        GKO_KERNEL_NOT_FOUND;                                                  \
        return nullptr;                                                        \
    }                                                                          \
                                                                               \
    template <template <typename...> class Base, typename K, typename... Rest, \
              typename Predicate, typename... InferredArgs>                    \
    _return _name(tt_list<K, Rest...>, Predicate is_eligible,                  \
                  const nlohmann::json& item, InferredArgs... args)            \
    {                                                                          \
        auto key = get_string(K{});                                            \
        if (is_eligible(key)) {                                                \
            return _callable<typename _get_type<Base, K>::type>(               \
                item, std::forward<InferredArgs>(args)...);                    \
        } else {                                                               \
            return _name<Base>(tt_list<Rest...>(), is_eligible, item,          \
                               std::forward<InferredArgs>(args)...);           \
        }                                                                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


#define ENABLE_SELECTION_ID(_name, _callable, _return, _get_type, _enum_type,  \
                            _enum_item)                                        \
    template <template <typename...> class Base, typename Predicate,           \
              typename... InferredArgs>                                        \
    _return _name(tt_list<>, Predicate is_eligible,                            \
                  const nlohmann::json& item, InferredArgs... args)            \
    {                                                                          \
        GKO_KERNEL_NOT_FOUND;                                                  \
        return nullptr;                                                        \
    }                                                                          \
                                                                               \
    template <template <typename...> class Base, typename K, typename... Rest, \
              typename Predicate, typename... InferredArgs>                    \
    _return _name(tt_list<K, Rest...>, Predicate is_eligible,                  \
                  const nlohmann::json& item, InferredArgs... args)            \
    {                                                                          \
        auto key = get_string(K{});                                            \
        if (is_eligible(key)) {                                                \
            return _callable<typename _get_type<                               \
                Base, typename concat<std::integral_constant<                  \
                                          _enum_type, _enum_type::_enum_item>, \
                                      K>::type>::type>(                        \
                item, std::forward<InferredArgs>(args)...);                    \
        } else {                                                               \
            return _name<Base>(tt_list<Rest...>(), is_eligible, item,          \
                               std::forward<InferredArgs>(args)...);           \
        }                                                                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#endif  // GKO_PUBLIC_EXT_FILE_CONFIG_BASE_MACRO_IMPL_HELPER_HPP_
