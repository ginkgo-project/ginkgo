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

#ifndef GKOEXT_RESOURCE_MANAGER_EXECUTOR_EXECUTOR_HPP_
#define GKOEXT_RESOURCE_MANAGER_EXECUTOR_EXECUTOR_HPP_


#include <iostream>
#include <memory>
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


// template <typename T>
// std::shared_ptr<T> get_gko_ptr(rapidjson::Value &item,
//                                std::shared_ptr<T> default_ptr)
// {
//     if (item.IsString()) {
//         this->search_<T>(item.GetString());
//     } else if (item.IsObject()) {
//         this->build_item<>
//     }
// }

// template <typename T>
// std::shared_ptr<T> get_gko_ptr_arr(rapidjson::Value &item,
//                                    std::shared_ptr<T> default_ptr)
// {
//     if
//         arr
//         {
//         for
//             loop this->get_gko_ptr
//         }
//     else {
//         this->get_gko_ptr
//     }
// }

template <>
std::shared_ptr<gko::CudaExecutor>
ResourceManager::build_item_impl<gko::CudaExecutor>(rapidjson::Value &item)
{
    std::cout << "Cuda" << std::endl;
    auto device_id = get_value_with_default(item, "device_id", 0);
    return CudaExecutor::create(device_id, ReferenceExecutor::create());
}

IMPLEMENT_BRIDGE(RM_Executor, CudaExecutor, CudaExecutor)
// IMPLEMENT_TINY_BRIDGE(RM_Executor, CudaExecutor, CudaExecutor)

template <>
std::shared_ptr<gko::HipExecutor>
ResourceManager::build_item_impl<gko::HipExecutor>(rapidjson::Value &item)
{
    std::cout << "Hip" << std::endl;
    auto device_id = get_value_with_default<int>(item, "device_id", 0);
    return HipExecutor::create(device_id, ReferenceExecutor::create());
}


IMPLEMENT_BRIDGE(RM_Executor, HipExecutor, HipExecutor)
// IMPLEMENT_TINY_BRIDGE(RM_Executor, HipExecutor, HipExecutor)

template <>
std::shared_ptr<gko::DpcppExecutor>
ResourceManager::build_item_impl<gko::DpcppExecutor>(rapidjson::Value &item)
{
    std::cout << "DPCPP" << std::endl;
    auto device_id = get_value_with_default<int>(item, "device_id", 0);
    return DpcppExecutor::create(device_id, ReferenceExecutor::create());
}


IMPLEMENT_BRIDGE(RM_Executor, DpcppExecutor, DpcppExecutor)
// IMPLEMENT_TINY_BRIDGE(RM_Executor, DpcppExecutor, DpcppExecutor)

template <>
std::shared_ptr<gko::ReferenceExecutor>
ResourceManager::build_item_impl<gko::ReferenceExecutor>(rapidjson::Value &item)
{
    std::cout << "REFERENCE" << std::endl;
    return ReferenceExecutor::create();
}


IMPLEMENT_BRIDGE(RM_Executor, ReferenceExecutor, ReferenceExecutor)
// IMPLEMENT_TINY_BRIDGE(RM_Executor, ReferenceExecutor, ReferenceExecutor)


template <>
std::shared_ptr<gko::OmpExecutor>
ResourceManager::build_item_impl<gko::OmpExecutor>(rapidjson::Value &item)
{
    std::cout << "OMP" << std::endl;
    return OmpExecutor::create();
}

IMPLEMENT_BRIDGE(RM_Executor, OmpExecutor, OmpExecutor)
// IMPLEMENT_TINY_BRIDGE(RM_Executor, OmpExecutor, OmpExecutor)


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_EXECUTOR_EXECUTOR_HPP_
