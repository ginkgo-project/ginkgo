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

#ifndef GKOEXT_RESOURCE_MANAGER_BASE_GENERIC_CONSTRUCTOR_HPP_
#define GKOEXT_RESOURCE_MANAGER_BASE_GENERIC_CONSTRUCTOR_HPP_

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

class ResourceManager;

DECLARE_SELECTION(LinOp, RM_LinOp);
DECLARE_SELECTION(LinOpFactory, RM_LinOpFactory);
DECLARE_SELECTION(Executor, RM_Executor);
DECLARE_SELECTION(CriterionFactory, RM_CriterionFactory);


template <typename T>
std::shared_ptr<T> create_from_config(rapidjson::Value &item, std::string base,
                                      std::shared_ptr<const Executor> exec,
                                      std::shared_ptr<const LinOp> linop,
                                      ResourceManager *manager);


CREATE_DEFAULT_IMPL(Executor);
CREATE_DEFAULT_IMPL(LinOp);
CREATE_DEFAULT_IMPL(LinOpFactory);
CREATE_DEFAULT_IMPL(CriterionFactory);

template <typename T>
struct Generic {
    using type = std::shared_ptr<T>;
    static type build(rapidjson::Value &item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager *manager);
};


GENERIC_BASE_IMPL(Executor);
GENERIC_BASE_IMPL(LinOp);
GENERIC_BASE_IMPL(LinOpFactory);
GENERIC_BASE_IMPL(CriterionFactory);


template <typename T>
std::shared_ptr<T> create_from_config(rapidjson::Value &item,
                                      std::shared_ptr<const Executor> exec,
                                      std::shared_ptr<const LinOp> linop,
                                      ResourceManager *manager = nullptr)
{
    return Generic<T>::build(item, exec, linop, manager);
}


template <typename T>
std::shared_ptr<T> create_from_config(rapidjson::Value &item,
                                      std::shared_ptr<const Executor> exec,
                                      ResourceManager *manager = nullptr)
{
    return create_from_config<T>(item, exec, nullptr, manager);
}

template <typename T>
std::shared_ptr<T> create_from_config(rapidjson::Value &item,
                                      std::shared_ptr<const LinOp> linop,
                                      ResourceManager *manager = nullptr)
{
    return create_from_config<T>(item, nullptr, linop, manager);
}

template <typename T>
std::shared_ptr<T> create_from_config(rapidjson::Value &item,
                                      ResourceManager *manager = nullptr)
{
    return create_from_config<T>(item, nullptr, nullptr, manager);
}


template <typename T, T base, typename U = typename gkobase<T>::type>
std::shared_ptr<U> create_from_config(rapidjson::Value &item,
                                      std::shared_ptr<const Executor> exec,
                                      std::shared_ptr<const LinOp> linop,
                                      ResourceManager *manager)
{
    return nullptr;
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_BASE_GENERIC_CONSTRUCTOR_HPP_
