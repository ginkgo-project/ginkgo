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

#ifndef GKOEXT_RESOURCE_MANAGER_STOP_ITERATION_HPP_
#define GKOEXT_RESOURCE_MANAGER_STOP_ITERATION_HPP_


#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"

#include <type_traits>


namespace gko {
namespace extension {
namespace resource_manager {
namespace {


std::shared_ptr<typename gko::stop::Iteration::Factory> build_iteraion_factory(
    ResourceManager *rm, rapidjson::Value &item)
{
    auto exec_ptr = get_pointer<Executor>(rm, item["exec"]);
    auto ptr = BUILD_FACTORY(gko::stop::Iteration, rm, item)
        WITH_VALUE(size_type, max_iters) ON_EXECUTOR;
    return ptr;
}


}  // namespace


#define CONNECT_STOP_FACTORY(base, func)                          \
    template <>                                                   \
    std::shared_ptr<typename base::Factory>                       \
        ResourceManager::build_item_impl<typename base::Factory>( \
            rapidjson::Value & item)                              \
    {                                                             \
        return func(this, item);                                  \
    }


CONNECT_STOP_FACTORY(gko::stop::Iteration, build_iteraion_factory);


template <>
std::shared_ptr<CriterionFactory>
ResourceManager::build_item<RM_CriterionFactory, RM_CriterionFactory::Iteration,
                            CriterionFactory>(rapidjson::Value &item)
{
    std::cout << "build_iteraion_factory" << std::endl;

    return this->build_item<gko::stop::Iteration::Factory>(item);
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_STOP_ITERATION_HPP_
