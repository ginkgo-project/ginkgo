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

#ifndef GKOEXT_RESOURCE_MANAGER_BASE_RAPIDJDON_HELPER_HPP_
#define GKOEXT_RESOURCE_MANAGER_BASE_RAPIDJDON_HELPER_HPP_

#include <rapidjson/document.h>
#include <ginkgo/ginkgo.hpp>
#include <resource_manager/base/resource_manager.hpp>

namespace gko {
namespace extension {
namespace resource_manager {

template <typename T>
T get_value(rapidjson::Value &item, std::string &key)
{
    GKO_NOT_IMPLEMENTED;
    return T{};
}

template <>
int get_value<int>(rapidjson::Value &item, std::string &key)
{
    return item[key.c_str()].GetInt();
}

template <>
size_type get_value<size_type>(rapidjson::Value &item, std::string &key)
{
    return item[key.c_str()].GetInt64();
}

template <>
dim<2> get_value<dim<2>>(rapidjson::Value &item, std::string &key)
{
    auto array = item[key.c_str()].GetArray();
    assert(array.Size() == 2);
    return dim<2>(array[0].GetInt64(), array[1].GetInt64());
}

template <>
std::string get_value<std::string>(rapidjson::Value &item, std::string &key)
{
    return item[key.c_str()].GetString();
}

template <typename T>
T get_value_with_default(rapidjson::Value &item, std::string key, T default_val)
{
    if (!item.HasMember(key.c_str())) {
        return default_val;
    } else {
        return get_value<T>(item, key);
    }
}


template <typename T>
std::shared_ptr<T> get_pointer(
    ResourceManager *rm, rapidjson::Value &item,
    std::shared_ptr<const gko::Executor> exec = nullptr,
    std::shared_ptr<const LinOp> linop = nullptr)
{
    std::shared_ptr<T> ptr;
    if (rm == nullptr) {
        if (item.IsObject()) {
            ptr = Generic<T>::build(item, exec, linop, rm);
        } else {
            assert(false);
        }
    } else {
        if (item.IsString()) {
            std::string opt = item.GetString();
            ptr = std::dynamic_pointer_cast<T>(rm->search_data<T>(opt));

        } else if (item.IsObject()) {
            // create a object
            ptr = rm->build_item<T>(item);
        } else {
            assert(false);
        }
    }
    assert(ptr.get() == nullptr);
    return std::move(ptr);
}


template <typename T>
std::vector<std::shared_ptr<const T>> get_pointer_vector(
    ResourceManager *rm, rapidjson::Value &item,
    std::shared_ptr<const gko::Executor> exec = nullptr,
    std::shared_ptr<const LinOp> linop = nullptr)
{
    std::vector<std::shared_ptr<const T>> vec;
    if (item.IsArray()) {
        for (auto &v : item.GetArray()) {
            auto ptr = get_pointer<T>(rm, v, exec, linop);
            std::cout << "array " << ptr << std::endl;
            vec.emplace_back(ptr);
        }
    } else {
        auto ptr = get_pointer<T>(rm, item, exec, linop);
        std::cout << "item " << ptr << std::endl;
        vec.emplace_back(ptr);
    }
    std::cout << "vec " << vec.size() << std::endl;
    return vec;
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_BASE_RAPIDJDON_HELPER_HPP_
