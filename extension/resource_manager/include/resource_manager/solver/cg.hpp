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

#ifndef GKOEXT_RESOURCE_MANAGER_SOLVER_CG_HPP_
#define GKOEXT_RESOURCE_MANAGER_SOLVER_CG_HPP_


#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"

#include <type_traits>


namespace gko {
namespace extension {
namespace resource_manager {


template <typename T>
std::shared_ptr<typename gko::solver::Cg<T>::Factory> build_cg_factory(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    auto exec_ptr =
        get_pointer_check<Executor>(manager, item, "exec", exec, linop);
    std::cout << "CgFactory " << exec_ptr.get() << std::endl;
    auto ptr = [&]() {
        BUILD_FACTORY(gko::solver::Cg<T>, manager, item, exec_ptr, linop);
        SET_POINTER(LinOp, generated_preconditioner);
        SET_POINTER(LinOpFactory, preconditioner);
        SET_POINTER_ARRAY(CriterionFactory, criteria);
        SET_EXECUTOR;
    }();


    return ptr;
}


CONNECT_GENERIC_SUB(gko::solver::Cg, PACK(float), Factory, build_cg_factory);
CONNECT_GENERIC_SUB(gko::solver::Cg, double, Factory, build_cg_factory);


SIMPLE_LINOP_WITH_FACTORY_IMPL(gko::solver::Cg, typename T, T);


template <>
std::shared_ptr<gko::LinOpFactory> create_from_config<
    RM_LinOpFactory, RM_LinOpFactory::CgFactory, gko::LinOpFactory>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::cout << "build_cg_factory" << std::endl;
    // go though the type
    std::string vt{"double"};
    if (item.HasMember("type")) {
        vt = item["type"].GetString();
    }
    if (vt == std::string{"double"}) {
        using type = double;
        return call<typename gko::solver::Cg<type>::Factory>(item, exec, linop,
                                                             manager);
    } else {
        using type = float;
        return call<typename gko::solver::Cg<type>::Factory>(item, exec, linop,
                                                             manager);
    }
}

template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Cg, gko::LinOp>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::cout << "build_cg" << std::endl;
    // go though the type
    std::string vt{"double"};
    if (item.HasMember("type")) {
        vt = item["type"].GetString();
    }
    if (vt == std::string{"double"}) {
        using type = double;
        return call<gko::solver::Cg<type>>(item, exec, linop, manager);
    } else {
        using type = float;
        return call<gko::solver::Cg<type>>(item, exec, linop, manager);
    }
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_SOLVER_CG_HPP_
