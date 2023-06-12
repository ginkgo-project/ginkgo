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

#ifndef GKO_PUBLIC_EXT_FILE_CONFIG_BASE_RESOURCE_MANAGER_HPP_
#define GKO_PUBLIC_EXT_FILE_CONFIG_BASE_RESOURCE_MANAGER_HPP_


#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>


#include <nlohmann/json.hpp>


#include "file_config/base/generic_constructor.hpp"
#include "file_config/base/template_helper.hpp"
#include "file_config/base/types.hpp"

namespace gko {
namespace extensions {
namespace file_config {


using ExecutorMap = std::unordered_map<std::string, std::shared_ptr<Executor>>;
using LinOpMap = std::unordered_map<std::string, std::shared_ptr<LinOp>>;
using LinOpFactoryMap =
    std::unordered_map<std::string, std::shared_ptr<LinOpFactory>>;
using CriterionFactoryMap =
    std::unordered_map<std::string, std::shared_ptr<CriterionFactory>>;
using LoggerMap = std::unordered_map<std::string, std::shared_ptr<Logger>>;
using ConfigMap = std::unordered_map<std::string, nlohmann::json>;


/**
 * map_type gives the map type according to the base type of given type.
 *
 * @tparam T  the type
 */
template <typename T, typename = void>
struct map_type {
    using type = void;
};

template <typename T>
struct map_type<T, typename std::enable_if<
                       std::is_convertible<T*, Executor*>::value>::type> {
    using type = ExecutorMap;
};

template <typename T>
struct map_type<
    T, typename std::enable_if<std::is_convertible<T*, LinOp*>::value>::type> {
    using type = LinOpMap;
};

template <typename T>
struct map_type<T, typename std::enable_if<
                       std::is_convertible<T*, LinOpFactory*>::value>::type> {
    using type = LinOpFactoryMap;
};

template <typename T>
struct map_type<T, typename std::enable_if<std::is_convertible<
                       T*, CriterionFactory*>::value>::type> {
    using type = CriterionFactoryMap;
};

template <typename T>
struct map_type<
    T, typename std::enable_if<std::is_convertible<T*, Logger*>::value>::type> {
    using type = LoggerMap;
};

/**
 * ResourceManager is a class to maintain the database storing the generated
 * Executor, LinOp, LinOpFactory, and CriterionFacory. It also can
 * build/insert/search item from the database.
 */
class ResourceManager {
public:
    /**
     * insert_data stores the data with the key.
     *
     * @tparam T  the type
     *
     * @param key  the unique key string
     * @param data  the shared pointer of the object
     */
    template <typename T>
    void insert_data(std::string key, std::shared_ptr<T> data)
    {
        this->get_map<T>().emplace(key, data);
    }

    /**
     * build_item is to build one object, which should contain name.
     *
     * @param item  the RapidJson::Value
     */
    void build_item(const nlohmann::json& item);

    /**
     * put_item is to build one object, which should contain name.
     *
     * @param item  the RapidJson::Value
     */
    void put_item(const nlohmann::json& item);

    /**
     * build_item is to build one object. If the object contains a name, add it
     * into database.
     *
     * @tparam T  the type
     *
     * @param item  the RapidJson::Value
     * @param exec  the Executor from outside
     * @param linop  the LinOp from outside
     *
     * @return the shared pointer of the type
     */
    template <typename T>
    std::shared_ptr<T> build_item(
        const nlohmann::json& item,
        std::shared_ptr<const Executor> exec = nullptr,
        std::shared_ptr<const LinOp> linop = nullptr)
    {
        std::cout << "create_from_config" << std::endl;
        auto ptr = create_from_config<T>(item, exec, linop, this);
        std::cout << "finish build" << std::endl;
        // if need to store the data, how to do that
        if (item.contains("name")) {
            this->insert_data<T>(item.at("name").get<std::string>(), ptr);
        }
        std::cout << "insert_data" << ptr << std::endl;
        return ptr;
    }

    /**
     * read is to build one object or list of objects, which top components
     * should contain name.
     *
     * @param item  the RapidJson::Value
     */
    void read(const nlohmann::json& dom)
    {
        if (dom.is_array()) {
            for (auto& item : dom) {
                this->build_item(item);
            }
        } else if (dom.is_object()) {
            this->build_item(dom);
        }
    }

    /**
     * search_data searches the key on the corresponding map.
     *
     * @tparam T  the type
     *
     * @param key  the key string
     *
     * @return the shared pointer of the object
     */
    template <typename T>
    std::shared_ptr<T> search_data(std::string key)
    {
        auto idx = this->get_map<T>().find(key);
        if (idx != this->get_map<T>().end()) {
            return std::dynamic_pointer_cast<T>(idx->second);
        } else {
            auto val = config_map_.find(key);
            if (val != config_map_.end()) {
                std::cout << "build map" << key << std::endl;
                return this->build_item<T>(val->second);
            }
        }
        return nullptr;
    }

    /**
     * put is to put one config or list of config, which top components
     * should contain name.
     *
     * @param item  the RapidJson::Value
     */
    void put(const nlohmann::json& dom)
    {
        if (dom.is_array()) {
            for (auto& item : dom) {
                this->put_item(item);
            }
        } else if (dom.is_object()) {
            this->put_item(dom);
        }
    }

    /**
     * output_map_info print the pairs of each map to standard output
     */
    void output_map_info()
    {
        this->output_map_info<ExecutorMap>();
        this->output_map_info<LinOpMap>();
        this->output_map_info<LinOpFactoryMap>();
        this->output_map_info<CriterionFactoryMap>();
        this->output_map_info<LoggerMap>();
    }

    /**
     * output_map_info print the pairs of certain map to standard output
     *
     * @tparam T  the map type
     */
    template <typename T>
    void output_map_info()
    {
        for (auto const& x : this->get_map_impl<T>()) {
            std::cout << x.first << ": " << x.second.get() << std::endl;
        }
    }

    /**
     * get_map gets the member map
     *
     * @tparam T  the type
     *
     * @return the map
     */
    template <typename T>
    typename map_type<T>::type& get_map()
    {
        return this->get_map_impl<typename map_type<T>::type>();
    }

protected:
    /**
     * get_map_impl is the implementation of get_map
     *
     * @tparam T  the map type
     *
     * @return the map
     */
    template <typename T>
    T& get_map_impl();

private:
    ExecutorMap executor_map_;
    LinOpMap linop_map_;
    LinOpFactoryMap linopfactory_map_;
    CriterionFactoryMap criterionfactory_map_;
    LoggerMap logger_map_;
    ConfigMap config_map_;
};


template <>
inline ExecutorMap& ResourceManager::get_map_impl<ExecutorMap>()
{
    return executor_map_;
}

template <>
inline LinOpMap& ResourceManager::get_map_impl<LinOpMap>()
{
    return linop_map_;
}

template <>
inline LinOpFactoryMap& ResourceManager::get_map_impl<LinOpFactoryMap>()
{
    return linopfactory_map_;
}

template <>
inline CriterionFactoryMap& ResourceManager::get_map_impl<CriterionFactoryMap>()
{
    return criterionfactory_map_;
}

template <>
inline LoggerMap& ResourceManager::get_map_impl<LoggerMap>()
{
    return logger_map_;
}

template <>
inline ConfigMap& ResourceManager::get_map_impl<ConfigMap>()
{
    return config_map_;
}


inline void ResourceManager::build_item(const nlohmann::json& item)
{
    assert(item.contains("name"));
    assert(item.contains("base"));
    std::string name = item.at("name").get<std::string>();
    std::string base = get_base_class(item["base"].get<std::string>());

    {
        auto ptr =
            create_from_config<Executor>(item, base, nullptr, nullptr, this);
        if (ptr != nullptr) {
            return;
        }
    }
    {
        auto ptr =
            create_from_config<LinOp>(item, base, nullptr, nullptr, this);
        if (ptr != nullptr) {
            return;
        }
    }
    {
        std::cout << "LinOpFactory" << std::endl;
        auto ptr = create_from_config<LinOpFactory>(item, base, nullptr,
                                                    nullptr, this);
        if (ptr != nullptr) {
            return;
        }
    }
    {
        std::cout << "StopFactory" << std::endl;
        auto ptr = create_from_config<CriterionFactory>(item, base, nullptr,
                                                        nullptr, this);
        if (ptr != nullptr) {
            return;
        }
    }
    {
        std::cout << "Logger" << std::endl;
        auto ptr =
            create_from_config<Logger>(item, base, nullptr, nullptr, this);
    }

    // go through all possiblilty from map and call the build_item<>
    // must contain name
}

inline void ResourceManager::put_item(const nlohmann::json& item)
{
    assert(item.contains("name"));
    std::string name = item.at("name").get<std::string>();
    std::cout << "put_item " << name << std::endl;
    config_map_[name] = item;
}


}  // namespace file_config
}  // namespace extensions
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_FILE_CONFIG_BASE_RESOURCE_MANAGER_HPP_
