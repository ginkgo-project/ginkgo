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


/**
 * get_value is to get data on the key from the RapidJson
 *
 * @tparam T  the type
 *
 * @param item  the RapidJson item
 * @param key  the key of value
 *
 * @return the type value
 */
template <typename T>
T get_value(rapidjson::Value &item, std::string &key);


template <>
bool get_value<bool>(rapidjson::Value &item, std::string &key)
{
    return item[key.c_str()].GetBool();
}

template <>
int get_value<int>(rapidjson::Value &item, std::string &key)
{
    return item[key.c_str()].GetInt();
}

template <>
size_type get_value<size_type>(rapidjson::Value &item, std::string &key)
{
    return item[key.c_str()].GetUint64();
}

template <>
gko::uint32 get_value<gko::uint32>(rapidjson::Value &item, std::string &key)
{
    return item[key.c_str()].GetUint();
}

template <>
dim<2> get_value<dim<2>>(rapidjson::Value &item, std::string &key)
{
    if (item[key.c_str()].IsArray()) {
        auto array = item[key.c_str()].GetArray();
        if (array.Size() == 2) {
            return dim<2>(array[0].GetInt64(), array[1].GetInt64());
        } else if (array.Size() == 1) {
            return dim<2>(array[0].GetInt64(), array[0].GetInt64());
        } else {
            assert(false);
        }
    } else if (item[key.c_str()].IsUint64()) {
        return dim<2>(item[key.c_str()].GetUint64(),
                      item[key.c_str()].GetUint64());
    } else {
        assert(false);
    }
}

template <>
std::string get_value<std::string>(rapidjson::Value &item, std::string &key)
{
    return item[key.c_str()].GetString();
}

template <>
float get_value<float>(rapidjson::Value &item, std::string &key)
{
    return (float)item[key.c_str()].GetDouble();
}

template <>
double get_value<double>(rapidjson::Value &item, std::string &key)
{
    return item[key.c_str()].GetDouble();
}

template <>
gko::stop::mode get_value<gko::stop::mode>(rapidjson::Value &item,
                                           std::string &key)
{
    auto mode = get_value<std::string>(item, key);
    if (mode == "absolute") {
        return gko::stop::mode::absolute;
    } else if (mode == "initial_resnorm") {
        return gko::stop::mode::initial_resnorm;
    } else if (mode == "rhs_norm") {
        return gko::stop::mode::rhs_norm;
    } else {
        assert(false);
    }
}


/**
 * get_value_with_default gives the value if the key is existed, or the given
 * default value.
 *
 * @tparam T  the type
 *
 * @param item  the RapidJson item
 * @param key  the key of value
 * @param default_val  the default value of the type if the key is not existed
 *
 * @return the type value
 */
template <typename T>
T get_value_with_default(rapidjson::Value &item, std::string key, T default_val)
{
    if (!item.HasMember(key.c_str())) {
        return default_val;
    } else {
        return get_value<T>(item, key);
    }
}


/**
 * get_pointer gives the shared_ptr<const type> from inputs.
 *
 * @tparam T  the type
 *
 * @param item  the RapidJson::Value
 * @param exec  the Executor from outside
 * @param linop  the LinOp from outside
 * @param manager  the ResourceManager pointer
 *
 * @return std::shared_ptr<const T>
 */
template <typename T>
std::shared_ptr<T> get_pointer(rapidjson::Value &item,
                               std::shared_ptr<const gko::Executor> exec,
                               std::shared_ptr<const LinOp> linop,
                               ResourceManager *manager)
{
    std::shared_ptr<T> ptr;
    using T_non_const = std::remove_const_t<T>;
    if (manager == nullptr) {
        if (item.IsObject()) {
            ptr = GenericHelper<T_non_const>::build(item, exec, linop, manager);
        } else {
            assert(false);
        }
    } else {
        if (item.IsString()) {
            std::string opt = item.GetString();
            ptr = manager->search_data<T_non_const>(opt);
        } else if (item.IsObject()) {
            ptr = manager->build_item<T_non_const>(item, exec, linop);
        } else {
            assert(false);
        }
    }
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
std::shared_ptr<const Executor> get_pointer<const Executor>(
    rapidjson::Value &item, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::shared_ptr<const Executor> ptr;
    if (manager == nullptr) {
        if (item.IsObject()) {
            ptr = GenericHelper<Executor>::build(item, exec, linop, manager);
        } else if (item.IsString() &&
                   std::string(item.GetString()) == std::string("inherit")) {
            ptr = exec;
        } else {
            assert(false);
        }
    } else {
        if (item.IsString()) {
            std::string opt = item.GetString();
            if (opt == std::string("inherit")) {
                ptr = exec;
            } else {
                ptr = manager->search_data<Executor>(opt);
            }
        } else if (item.IsObject()) {
            ptr = manager->build_item<Executor>(item);
        } else {
            assert(false);
        }
    }
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
std::shared_ptr<const LinOp> get_pointer<const LinOp>(
    rapidjson::Value &item, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::shared_ptr<const LinOp> ptr;
    if (manager == nullptr) {
        if (item.IsObject()) {
            ptr = GenericHelper<LinOp>::build(item, exec, linop, manager);
        } else if (item.IsString() &&
                   std::string(item.GetString()) == std::string("given")) {
            ptr = linop;
        } else {
            assert(false);
        }
    } else {
        if (item.IsString()) {
            std::string opt = item.GetString();
            if (opt == std::string("given")) {
                ptr = linop;
            } else {
                ptr = manager->search_data<LinOp>(opt);
            }
        } else if (item.IsObject()) {
            ptr = manager->build_item<LinOp>(item);
        } else {
            assert(false);
        }
    }
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}


/**
 * get_pointer_check considers existence of the key to decide the behavior.
 *
 * @tparam T  the type
 *
 * @param item  the RapidJson::Value
 * @param key  the key string
 * @param exec  the Executor from outside
 * @param linop  the LinOp from outside
 * @param manager  the ResourceManager pointer
 *
 * @return std::shared_ptr<const T>
 */
template <typename T>
std::shared_ptr<const T> get_pointer_check(
    rapidjson::Value &item, std::string key,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    assert(item.HasMember(key.c_str()));
    return get_pointer<const T>(item[key.c_str()], exec, linop, manager);
}

template <>
std::shared_ptr<const Executor> get_pointer_check<const Executor>(
    rapidjson::Value &item, std::string key,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    if (item.HasMember(key.c_str())) {
        return get_pointer<const Executor>(item[key.c_str()], exec, linop,
                                           manager);
    } else if (exec != nullptr) {
        return exec;
    } else {
        assert(false);
        return nullptr;
    }
}

/**
 * get_pointer_vector creates a vector of the shared pointer of const type.
 *
 * @tparam T  the type
 *
 * @param item  the RapidJson::Value
 * @param exec  the Executor from outside
 * @param linop  the LinOp from outside
 * @param manager  the ResourceManager pointer
 *
 * @return std::vector<std::shared_ptr<const T>>
 */
template <typename T>
std::vector<std::shared_ptr<const T>> get_pointer_vector(
    rapidjson::Value &item, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::vector<std::shared_ptr<const T>> vec;
    if (item.IsArray()) {
        for (auto &v : item.GetArray()) {
            std::cout << "item " << exec.get() << std::endl;
            auto ptr = get_pointer<const T>(v, exec, linop, manager);
            std::cout << "array " << ptr << std::endl;
            vec.emplace_back(ptr);
        }
    } else {
        auto ptr = get_pointer<const T>(item, exec, linop, manager);
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
