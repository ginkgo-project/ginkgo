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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_EXECUTOR_EXECUTOR_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_EXECUTOR_EXECUTOR_HPP_


#include <iostream>
#include <memory>


#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


template <>
struct Generic<gko::CudaExecutor> {
    using type = std::shared_ptr<gko::CudaExecutor>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        std::cout << "Cuda" << std::endl;
        auto device_id = get_value_with_default(item, "device_id", 0);
        auto ptr = CudaExecutor::create(device_id, ReferenceExecutor::create());
        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};


IMPLEMENT_BRIDGE(RM_Executor, CudaExecutor, CudaExecutor);


template <>
struct Generic<gko::HipExecutor> {
    using type = std::shared_ptr<gko::HipExecutor>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        std::cout << "Hip" << std::endl;
        auto device_id = get_value_with_default(item, "device_id", 0);
        auto ptr = HipExecutor::create(device_id, ReferenceExecutor::create())
            add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};


IMPLEMENT_BRIDGE(RM_Executor, HipExecutor, HipExecutor);


template <>
struct Generic<gko::DpcppExecutor> {
    using type = std::shared_ptr<gko::DpcppExecutor>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        std::cout << "Dpcpp" << std::endl;
        auto device_id = get_value_with_default(item, "device_id", 0);
        auto device_type =
            get_value_with_default(item, "device_type", std::string("all"));
        auto ptr = DpcppExecutor::create(device_id, ReferenceExecutor::create(),
                                         device_type);
        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};


IMPLEMENT_BRIDGE(RM_Executor, DpcppExecutor, DpcppExecutor);


template <>
struct Generic<gko::ReferenceExecutor> {
    using type = std::shared_ptr<gko::ReferenceExecutor>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        std::cout << "Reference" << std::endl;
        auto ptr = ReferenceExecutor::create();
        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};


IMPLEMENT_BRIDGE(RM_Executor, ReferenceExecutor, ReferenceExecutor);


template <>
struct Generic<gko::OmpExecutor> {
    using type = std::shared_ptr<gko::OmpExecutor>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        std::cout << "Omp" << std::endl;
        auto ptr = OmpExecutor::create();
        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};


IMPLEMENT_BRIDGE(RM_Executor, OmpExecutor, OmpExecutor);


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_EXECUTOR_EXECUTOR_HPP_
