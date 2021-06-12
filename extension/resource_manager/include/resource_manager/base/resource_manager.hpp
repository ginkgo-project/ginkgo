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

#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/types.hpp"

namespace gko {
namespace extension {
namespace resource_manager {


class ResourceManager {
public:
    template <typename T>
    void insert_data(std::string key, std::shared_ptr<T> data)
    {
        this->get_map<T>().emplace(key, data);
    }

    template <typename T>
    std::shared_ptr<T> search_data(std::string key)
    {
        auto it = this->get_map<T>().find(key);
        return it->second;
    }

    // contain name
    void build_item(rapidjson::Value &item);

    template <typename T>
    std::shared_ptr<T> build_item(rapidjson::Value &item)
    {
        return nullptr;
    }

    template <typename T>
    std::shared_ptr<T> build_item(std::string &base, rapidjson::Value &item)
    {
        return nullptr;
    };

    template <typename T, T base, typename U = typename gkobase<T>::type>
    std::shared_ptr<U> build_item(rapidjson::Value &item)
    {
        // LinOp go through all LinOp
        // explicit type go directly
        // if contain a name store it
        return nullptr;
    }

    template <typename T>
    std::vector<std::shared_ptr<T>> build_array(rapidjson::Value &array)
    {
        // for create a sequence of the same type. like criterion list
        assert(array.IsArray());
        std::vector<std::shared_ptr<T>> result(array.Size());
        for (auto &item : array.GetArray()) {
            result.emplace_back(this->build_item<T>(item));
        }
    }

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


protected:
    template <typename T>
    typename map_type<T>::type &get_map()
    {
        // GKO_NOT_IMPLEMENTED;
    }

    DECLARE_BASE_BUILD_ITEM(Executor, RM_Executor);
    DECLARE_BASE_BUILD_ITEM(LinOp, RM_LinOp);
    DECLARE_BASE_BUILD_ITEM(LinOpFactory, RM_LinOpFactory);
    DECLARE_BASE_BUILD_ITEM(CriterionFactory, RM_CriterionFactory);

private:
    std::unordered_map<std::string, std::shared_ptr<Executor>> executor_map_;
    std::unordered_map<std::string, std::shared_ptr<LinOp>> linop_map_;
    std::unordered_map<std::string, std::shared_ptr<LinOpFactory>>
        linopfactory_map_;
    std::unordered_map<std::string, std::shared_ptr<CriterionFactory>>
        criterionfactory_map_;
};


template <>
map_type<Executor>::type &ResourceManager::get_map<Executor>()
{
    return executor_map_;
}

template <>
map_type<LinOp>::type &ResourceManager::get_map<LinOp>()
{
    return linop_map_;
}

template <>
map_type<LinOpFactory>::type &ResourceManager::get_map<LinOpFactory>()
{
    return linopfactory_map_;
}

template <>
map_type<CriterionFactory>::type &ResourceManager::get_map<CriterionFactory>()
{
    return criterionfactory_map_;
}


IMPLEMENT_BASE_BUILD_ITEM(Executor, RM_Executor::Executor);
IMPLEMENT_BASE_BUILD_ITEM(LinOp, RM_LinOp::LinOp);
IMPLEMENT_BASE_BUILD_ITEM(LinOpFactory, RM_LinOpFactory::LinOpFactory);
IMPLEMENT_BASE_BUILD_ITEM(CriterionFactory,
                          RM_CriterionFactory::CriterionFactory);


void ResourceManager::build_item(rapidjson::Value &item)
{
    assert(item.HasMember("name"));
    assert(item.HasMember("base"));
    std::string name = item["name"].GetString();
    std::string base = item["base"].GetString();

    // if (base == std::string{})
    auto ptr = this->build_item<Executor>(base, item);
    if (ptr == nullptr) {
        auto ptr = this->build_item<LinOp>(base, item);
        std::cout << "123" << std::endl;
        if (ptr == nullptr) {
        } else {
            this->insert_data<LinOp>(name, ptr);
        }
    } else {
        this->insert_data<Executor>(name, ptr);
    }
    // go through all possiblilty from map and call the build_item<>
    // must contain name
}

template <>
std::shared_ptr<Executor> ResourceManager::build_item<Executor>(
    rapidjson::Value &item)
{
    assert(item.HasMember("base"));
    std::string base = item["base"].GetString();
    return this->build_item<Executor>(base, item);
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_BASE_RESOURCE_MANAGER_HPP_
