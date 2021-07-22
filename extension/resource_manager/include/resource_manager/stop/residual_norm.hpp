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

#ifndef GKOEXT_RESOURCE_MANAGER_STOP_RESIDUAL_NORM_HPP_
#define GKOEXT_RESOURCE_MANAGER_STOP_RESIDUAL_NORM_HPP_


#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"

#include <type_traits>


namespace gko {
namespace extension {
namespace resource_manager {


template <typename T>
struct Generic<typename gko::stop::ResidualNorm<T>::Factory,
               gko::stop::ResidualNorm<T>> {
    using type = std::shared_ptr<typename gko::stop::ResidualNorm<T>::Factory>;
    static type build(rapidjson::Value &item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager *manager)
    {
        std::cout << "ResidualNorm exec:" << exec.get() << std::endl;
        auto ptr = [&]() {
            BUILD_FACTORY(gko::stop::ResidualNorm<T>, manager, item, exec,
                          linop);
            std::cout << "Iter 1:" << std::endl;
            SET_VALUE(T, reduction_factor);
            std::cout << "Iter 2:" << std::endl;
            SET_VALUE(gko::stop::mode, baseline);
            SET_EXECUTOR;
        }();
        std::cout << "Iter 3:" << std::endl;
        return ptr;
    }
};


ENABLE_SELECTION(residual_norm_select, call,
                 std::shared_ptr<gko::stop::CriterionFactory>,
                 get_the_factory_type);
constexpr auto residual_norm_list = tt_list<double, float>();


template <>
std::shared_ptr<gko::stop::CriterionFactory>
create_from_config<RM_CriterionFactory, RM_CriterionFactory::ResidualNorm,
                   gko::stop::CriterionFactory>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::cout << "build_residual_norm" << std::endl;
    // go though the type
    auto vt = get_value_with_default(item, "ValueType", default_valuetype);
    auto type_string = vt;
    auto ptr = residual_norm_select<gko::stop::ResidualNorm>(
        residual_norm_list, [=](std::string key) { return key == type_string; },
        item, exec, linop, manager);
    return ptr;
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_STOP_RESIDUAL_NORM_HPP_
