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

#ifndef GKOEXT_RESOURCE_MANAGER_BASE_ELEMENT_TYPES_HPP_
#define GKOEXT_RESOURCE_MANAGER_BASE_ELEMENT_TYPES_HPP_

#include <ginkgo/ginkgo.hpp>
// #include <resource_manager/base/macro_helper.hpp>


namespace gko {
namespace extension {
namespace resource_manager {


template <typename... Types>
using type_list = ::gko::syn::type_list<Types...>;

#define GET_STRING_PARTIAL(_type, _str) \
    template <>                         \
    std::string get_string<_type>()     \
    {                                   \
        return #_str;                   \
    }

template <typename T>
std::string get_string();

GET_STRING_PARTIAL(double, double);
GET_STRING_PARTIAL(float, float);
GET_STRING_PARTIAL(gko::int32, int);
GET_STRING_PARTIAL(gko::int64, int64);

template <typename T>
std::string get_string(T)
{
    return get_string<T>();
}

template <typename K>
std::string get_string(type_list<K>)
{
    return get_string<K>();
}

template <typename K, typename... Rest>
std::string get_string(type_list<K, Rest...>)
{
    return get_string<K>() + "+" + get_string(type_list<Rest...>());
}

template <template <typename...> class base, typename T>
struct get_the_type {
    using type = base<T>;
};

template <template <typename...> class base, typename... Rest>
struct get_the_type<base, type_list<Rest...>> {
    using type = base<Rest...>;
};

template <template <typename...> class base, typename T>
struct get_the_factory_type {
    using type = typename base<T>::Factory;
};

template <template <typename...> class base, typename... Rest>
struct get_the_factory_type<base, type_list<Rest...>> {
    using type = typename base<Rest...>::Factory;
};

#define ENABLE_SELECTION(_name, _callable, _return, _get_type)                 \
    template <template <typename...> class Base, typename Predicate,           \
              typename... InferredArgs>                                        \
    _return _name(type_list<>, Predicate is_eligible, rapidjson::Value &item,  \
                  InferredArgs... args)                                        \
    {                                                                          \
        GKO_KERNEL_NOT_FOUND;                                                  \
        return nullptr;                                                        \
    }                                                                          \
                                                                               \
    template <template <typename...> class Base, typename K, typename... Rest, \
              typename Predicate, typename... InferredArgs>                    \
    _return _name(type_list<K, Rest...>, Predicate is_eligible,                \
                  rapidjson::Value &item, InferredArgs... args)                \
    {                                                                          \
        auto key = get_string(K{});                                            \
        if (is_eligible(key)) {                                                \
            return _callable<typename _get_type<Base, K>::type>(               \
                item, std::forward<InferredArgs>(args)...);                    \
        } else {                                                               \
            return _name<Base>(type_list<Rest...>(), is_eligible, item,        \
                               std::forward<InferredArgs>(args)...);           \
        }                                                                      \
    }

const std::string default_valuetype{get_string<gko::default_precision>()};
const std::string default_indextype{get_string<gko::int32>()};

}  // namespace resource_manager
}  // namespace extension
}  // namespace gko

#endif  // GKOEXT_RESOURCE_MANAGER_BASE_ELEMENT_TYPES_HPP_
