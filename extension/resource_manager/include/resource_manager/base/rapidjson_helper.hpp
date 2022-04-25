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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_RAPIDJSON_HELPER_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_RAPIDJSON_HELPER_HPP_


#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


#include <rapidjson/allocators.h>
#include <rapidjson/document.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "resource_manager/base/generic_constructor.hpp"
#include "resource_manager/base/resource_manager.hpp"
#include "resource_manager/base/types.hpp"
#include "resource_manager/log/mask_type.hpp"


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
T get_value(rapidjson::Value& item, std::string& key);


template <>
bool get_value<bool>(rapidjson::Value& item, std::string& key)
{
    return item[key.c_str()].GetBool();
}

template <>
int get_value<int>(rapidjson::Value& item, std::string& key)
{
    return item[key.c_str()].GetInt();
}

template <>
gko::int64 get_value<gko::int64>(rapidjson::Value& item, std::string& key)
{
    return item[key.c_str()].GetInt64();
}

template <>
size_type get_value<size_type>(rapidjson::Value& item, std::string& key)
{
    return item[key.c_str()].GetUint64();
}

template <>
gko::uint32 get_value<gko::uint32>(rapidjson::Value& item, std::string& key)
{
    return item[key.c_str()].GetUint();
}

template <>
dim<2> get_value<dim<2>>(rapidjson::Value& item, std::string& key)
{
    if (item[key.c_str()].IsArray()) {
        auto array = item[key.c_str()].GetArray();
        if (array.Size() == 2) {
            return dim<2>(array[0].GetInt64(), array[1].GetInt64());
        } else if (array.Size() == 1) {
            return dim<2>(array[0].GetInt64(), array[0].GetInt64());
        } else {
            assert(false);
            // avoid the warning about return type
            return dim<2>();
        }
    } else if (item[key.c_str()].IsUint64()) {
        return dim<2>(item[key.c_str()].GetUint64(),
                      item[key.c_str()].GetUint64());
    } else {
        assert(false);
        // avoid the warning about return type
        return dim<2>();
    }
}

template <>
std::string get_value<std::string>(rapidjson::Value& item, std::string& key)
{
    return item[key.c_str()].GetString();
}

template <>
float get_value<float>(rapidjson::Value& item, std::string& key)
{
    return static_cast<float>(item[key.c_str()].GetDouble());
}

template <>
double get_value<double>(rapidjson::Value& item, std::string& key)
{
    return item[key.c_str()].GetDouble();
}

template <>
std::complex<double> get_value<std::complex<double>>(rapidjson::Value& item,
                                                     std::string& key)
{
    const auto& item_value = item[key.c_str()];
    if (item_value.IsNumber()) {
        return static_cast<std::complex<double>>(get_value<double>(item, key));
    } else if (item_value.IsArray()) {
        // [real, imag]
        double real = 0;
        double imag = 0;
        assert(item_value.Size() <= 2);
        if (item_value.Size() >= 1) {
            real = item[0].GetDouble();
        }
        if (item_value.Size() >= 2) {
            imag = item[1].GetDouble();
        }
        return std::complex<double>(real, imag);
    } else if (item_value.IsString()) {
        // allow real, (real), (real,imag)
        auto item_string = get_value<std::string>(item, key);
        std::istringstream str(item_string);
        std::complex<double> value(0);
        str >> value;
        return value;
    }
    return std::complex<double>(0);
}

template <>
std::complex<float> get_value<std::complex<float>>(rapidjson::Value& item,
                                                   std::string& key)
{
    // rapidjson only has double
    return static_cast<std::complex<float>>(
        get_value<std::complex<double>>(item, key));
}

template <>
gko::stop::mode get_value<gko::stop::mode>(rapidjson::Value& item,
                                           std::string& key)
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
        // avoid the warning about return type
        return gko::stop::mode::absolute;
    }
}


// It can not use the overload get_value because mask_type is uint.
gko::log::Logger::mask_type get_mask_value_with_default(
    rapidjson::Value& item, std::string key,
    gko::log::Logger::mask_type default_val)
{
    gko::log::Logger::mask_type mask_value = 0;
    if (item.HasMember(key.c_str())) {
        auto& mask_item = item[key.c_str()];
        if (mask_item.IsString()) {
            mask_value |= mask_type_map.at(get_value<std::string>(item, key));
        } else if (mask_item.IsArray()) {
            for (auto& it : mask_item.GetArray()) {
                mask_value |= mask_type_map.at(it.GetString());
            }
        } else {
            assert(false);
        }
    } else {
        mask_value = default_val;
    }

    return mask_value;
}


template <typename ValueType>
gko::Array<ValueType> get_array(rapidjson::Value& item, std::string& key,
                                std::shared_ptr<const gko::Executor> exec)
{
    std::size_t size = 0;
    if (item[key.c_str()].IsNumber()) {
        size = 1;
    } else if (item[key.c_str()].IsArray()) {
        size = item[key.c_str()].Size();
    }
    gko::Array<ValueType> array(exec, size);
    if (size == 0) {
        return array;
    }

    gko::Array<ValueType> host_array(exec->get_master(), size);
    if (item[key.c_str()].IsNumber()) {
        host_array.get_data()[0] = get_value<ValueType>(item, key);
    } else if (item[key.c_str()].IsArray()) {
        for (std::size_t i = 0; i < item[key.c_str()].Size(); i++) {
            // RODO: get_value does not check the existence, so maybe use the
            // item directly not with key
            host_array.get_data()[i] =
                static_cast<ValueType>(item[key.c_str()][i].GetDouble());
        }
    }
    array = host_array;
    return array;
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
T get_value_with_default(rapidjson::Value& item, std::string key, T default_val)
{
    if (!item.HasMember(key.c_str())) {
        return default_val;
    } else {
        return get_value<T>(item, key);
    }
}


template <typename T>
T get_required_value(rapidjson::Value& item, std::string key)
{
    if (!item.HasMember(key.c_str())) {
        std::cerr << "the value of key " << key << " must not be empty"
                  << std::endl;
        assert(false);
        return T{};
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
std::shared_ptr<T> get_pointer(rapidjson::Value& item,
                               std::shared_ptr<const gko::Executor> exec,
                               std::shared_ptr<const LinOp> linop,
                               ResourceManager* manager)
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
            std::cout << "search item" << std::endl;
            std::string opt = item.GetString();
            ptr = manager->search_data<T_non_const>(opt);
            std::cout << "get ptr " << ptr.get() << std::endl;
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
    rapidjson::Value& item, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
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
    rapidjson::Value& item, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
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
    rapidjson::Value& item, std::string key,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    assert(item.HasMember(key.c_str()));
    return get_pointer<const T>(item[key.c_str()], exec, linop, manager);
}

template <>
std::shared_ptr<const Executor> get_pointer_check<const Executor>(
    rapidjson::Value& item, std::string key,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
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
    rapidjson::Value& item, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    std::vector<std::shared_ptr<const T>> vec;
    if (item.IsArray()) {
        for (auto& v : item.GetArray()) {
            std::cout << "item " << exec.get() << std::endl;
            auto ptr = get_pointer<const T>(v, exec, linop, manager);
            std::cout << "array " << ptr << std::endl;
            vec.emplace_back(ptr);
        }
    } else {
        std::cout << "object or string" << std::endl;
        auto ptr = get_pointer<const T>(item, exec, linop, manager);
        std::cout << "item " << ptr << std::endl;
        vec.emplace_back(ptr);
    }
    std::cout << "vec " << vec.size() << std::endl;
    return vec;
}


template <typename Type>
void add_logger(Type& obj, rapidjson::Value& item,
                std::shared_ptr<const Executor> exec,
                std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    if (item.HasMember("logger")) {
        auto& logger = item["logger"];
        if (logger.IsArray()) {
            for (auto& it : logger.GetArray()) {
                obj->add_logger(get_pointer<Logger>(it, exec, linop, manager));
            }
        } else {
            obj->add_logger(get_pointer<Logger>(logger, exec, linop, manager));
        }
    }
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_BASE_RAPIDJSON_HELPER_HPP_
