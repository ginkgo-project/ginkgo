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

#ifndef GKOEXT_RESOURCE_MANAGER_BASE_RESOURCE_MANAGER_HPP_
#define GKOEXT_RESOURCE_MANAGER_BASE_RESOURCE_MANAGER_HPP_

#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <unordered_map>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>

#include "resource_manager/base/generic_constructor.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/types.hpp"

namespace gko {
namespace extension {
namespace resource_manager {


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
        auto it = this->get_map<T>().find(key);
        return std::dynamic_pointer_cast<T>(it->second);
    }

    /**
     * build_item is to build one object, which should contain name.
     *
     * @param item  the RapidJson::Value
     */
    void build_item(rapidjson::Value &item);

    /**
     * build_item is to build one object. If the object contains a name, add it
     * into database.
     *
     * @tparam T  the type
     *
     * @param item  the RapidJson::Value
     * @param base  the base string
     * @param exec  the Executor from outside
     * @param linop  the LinOp from outside
     *
     * @return the shared pointer of the type
     */
    template <typename T>
    std::shared_ptr<T> build_item(
        rapidjson::Value &item, std::string base,
        std::shared_ptr<const Executor> exec = nullptr,
        std::shared_ptr<const LinOp> linop = nullptr)
    {
        std::cout << "create_from_config" << std::endl;
        auto ptr = create_from_config<T>(item, base, exec, linop, this);
        // if need to store the data, how to do that
        if (item.HasMember("name")) {
            this->insert_data<T>(item["name"].GetString(), ptr);
        }
        return ptr;
    }

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
        rapidjson::Value &item, std::shared_ptr<const Executor> exec = nullptr,
        std::shared_ptr<const LinOp> linop = nullptr)
    {
        std::cout << "create_from_config" << std::endl;
        auto ptr = GenericHelper<T>::build(item, exec, linop, this);
        // if need to store the data, how to do that
        if (item.HasMember("name")) {
            this->insert_data<T>(item["name"].GetString(), ptr);
        }
        return ptr;
    }

    /**
     * read is to build one object or list of objects, which top components
     * should contain name.
     *
     * @param item  the RapidJson::Value
     */
    void read(rapidjson::Value &dom)
    {
        if (dom.IsArray()) {
            for (auto &item : dom.GetArray()) {
                this->build_item(item);
            }
        } else if (dom.IsObject()) {
            this->build_item(dom);
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
    }

    /**
     * output_map_info print the pairs of certain map to standard output
     *
     * @tparam T  the map type
     */
    template <typename T>
    void output_map_info()
    {
        for (auto const &x : this->get_map_impl<T>()) {
            std::cout << x.first << ": " << x.second.get() << std::endl;
        }
    }

protected:
    /**
     * get_map gets the member map
     *
     * @tparam T  the type
     *
     * @return the map
     */
    template <typename T>
    typename map_type<T>::type &get_map()
    {
        return this->get_map_impl<typename map_type<T>::type>();
    }

    /**
     * get_map_impl is the implementation of get_map
     *
     * @tparam T  the map type
     *
     * @return the map
     */
    template <typename T>
    T &get_map_impl();

private:
    std::unordered_map<std::string, std::shared_ptr<Executor>> executor_map_;
    std::unordered_map<std::string, std::shared_ptr<LinOp>> linop_map_;
    std::unordered_map<std::string, std::shared_ptr<LinOpFactory>>
        linopfactory_map_;
    std::unordered_map<std::string, std::shared_ptr<CriterionFactory>>
        criterionfactory_map_;
};


template <>
ExecutorMap &ResourceManager::get_map_impl<ExecutorMap>()
{
    return executor_map_;
}

template <>
LinOpMap &ResourceManager::get_map_impl<LinOpMap>()
{
    return linop_map_;
}

template <>
LinOpFactoryMap &ResourceManager::get_map_impl<LinOpFactoryMap>()
{
    return linopfactory_map_;
}

template <>
CriterionFactoryMap &ResourceManager::get_map_impl<CriterionFactoryMap>()
{
    return criterionfactory_map_;
}


void ResourceManager::build_item(rapidjson::Value &item)
{
    assert(item.HasMember("name"));
    assert(item.HasMember("base"));
    std::string name = item["name"].GetString();
    std::string base = item["base"].GetString();

    // if (base == std::string{})
    auto ptr = create_from_config<Executor>(item, base, nullptr, nullptr, this);
    if (ptr == nullptr) {
        auto ptr =
            create_from_config<LinOp>(item, base, nullptr, nullptr, this);
        std::cout << "123" << std::endl;
        if (ptr == nullptr) {
            std::cout << "LinOpFactory" << std::endl;
            auto ptr = create_from_config<LinOpFactory>(item, base, nullptr,
                                                        nullptr, this);
            if (ptr == nullptr) {
                std::cout << "StopFactory" << std::endl;
                auto ptr = create_from_config<CriterionFactory>(
                    item, base, nullptr, nullptr, this);
            }
        } else {
        }
    } else {
        // this->insert_data<Executor>(name, ptr);
    }
    // go through all possiblilty from map and call the build_item<>
    // must contain name
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_BASE_RESOURCE_MANAGER_HPP_
