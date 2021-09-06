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

#ifndef GKOEXT_RESOURCE_MANAGER_PRECONDITIONER_ILU_HPP_
#define GKOEXT_RESOURCE_MANAGER_PRECONDITIONER_ILU_HPP_


#include "resource_manager/base/element_types.hpp"
#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"

#include <type_traits>


namespace gko {
namespace extension {
namespace resource_manager {


template <typename LSolverType, typename USolverType, bool ReverseApply,
          typename IndexType>
struct Generic<typename gko::preconditioner::Ilu<
                   LSolverType, USolverType, ReverseApply, IndexType>::Factory,
               gko::preconditioner::Ilu<LSolverType, USolverType, ReverseApply,
                                        IndexType>> {
    using type = std::shared_ptr<typename gko::preconditioner::Ilu<
        LSolverType, USolverType, ReverseApply, IndexType>::Factory>;
    static type build(rapidjson::Value &item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager *manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY(
                PACK(gko::preconditioner::Ilu<LSolverType, USolverType,
                                              ReverseApply, IndexType>),
                manager, item, exec, linop);
            SET_NON_CONST_POINTER(typename LSolverType::Factory,
                                  l_solver_factory);
            SET_NON_CONST_POINTER(typename USolverType::Factory,
                                  u_solver_factory);
            SET_NON_CONST_POINTER(LinOpFactory, factorization_factory);
            SET_EXECUTOR;
        }();

        std::cout << "123" << std::endl;
        return ptr;
    }
};


SIMPLE_LINOP_WITH_FACTORY_IMPL(gko::preconditioner::Ilu,
                               PACK(typename LSolverType, typename USolverType,
                                    bool ReverseApply, typename IndexType),
                               PACK(LSolverType, USolverType, ReverseApply,
                                    IndexType));

ENABLE_SELECTION(ilufactory_select, call, std::shared_ptr<gko::LinOpFactory>,
                 get_actual_factory_type);
ENABLE_SELECTION(ilu_select, call, std::shared_ptr<gko::LinOp>,
                 get_actual_type);
constexpr auto ilu_list =
    typename span_list<gko::solver::LowerTrs<>, gko::solver::UpperTrs<>,
                       tt_list<std::true_type, std::false_type>,
                       tt_list<gko::int32, gko::int64>>::type();

template <>
std::shared_ptr<gko::LinOpFactory> create_from_config<
    RM_LinOpFactory, RM_LinOpFactory::IluFactory, gko::LinOpFactory>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::cout << "ilu_factory" << std::endl;
    // go though the type
    using namespace std::literals::string_literals;
    auto it = get_value_with_default(item, "IndexType", default_indextype);
    auto ltr_t = get_value_with_default(item, "LowerTrs", "LowerTrs"s);
    auto utr_t = get_value_with_default(item, "UpperTrs", "UpperTrs"s);
    auto reverse_apply_t =
        get_value_with_default(item, "reverse_apply", "false"s);
    auto type_string = create_type_name(ltr_t, utr_t, reverse_apply_t, it);
    auto ptr = ilufactory_select<type_list>(
        ilu_list, [=](std::string key) { return key == type_string; }, item,
        exec, linop, manager);
    return ptr;
}


template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Ilu, gko::LinOp>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::cout << "build_ilu" << std::endl;
    // go though the type
    using namespace std::literals::string_literals;
    auto it = get_value_with_default(item, "IndexType", default_indextype);
    auto ltr_t = get_value_with_default(item, "LowerTrs", "LowerTrs"s);
    auto utr_t = get_value_with_default(item, "UpperTrs", "UpperTrs"s);
    auto reverse_apply_t =
        get_value_with_default(item, "reverse_apply", "false"s);
    auto type_string = create_type_name(ltr_t, utr_t, reverse_apply_t, it);
    auto ptr = ilu_select<type_list>(
        ilu_list, [=](std::string key) { return key == type_string; }, item,
        exec, linop, manager);
    return ptr;
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_PRECONDITIONER_ILU_HPP_
