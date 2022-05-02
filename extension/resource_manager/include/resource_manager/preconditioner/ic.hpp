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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_PRECONDITIONER_IC_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_PRECONDITIONER_IC_HPP_


#include <ginkgo/core/preconditioner/ic.hpp>


#include "resource_manager/base/generic_constructor.hpp"
#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"
#include "resource_manager/base/template_helper.hpp"
#include "resource_manager/base/type_default.hpp"
#include "resource_manager/base/type_pack.hpp"
#include "resource_manager/base/type_resolving.hpp"
#include "resource_manager/base/type_string.hpp"
#include "resource_manager/base/types.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


template <typename LSolverType, typename IndexType>
struct Generic<
    typename gko::preconditioner::Ic<LSolverType, IndexType>::Factory,
    gko::preconditioner::Ic<LSolverType, IndexType>> {
    using type = std::shared_ptr<
        typename gko::preconditioner::Ic<LSolverType, IndexType>::Factory>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY(PACK(gko::preconditioner::Ic<LSolverType, IndexType>),
                          manager, item, exec, linop);
            SET_POINTER(typename LSolverType::Factory, l_solver_factory);
            SET_POINTER(LinOpFactory, factorization_factory);
            SET_EXECUTOR;
        }();
        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};

SIMPLE_LINOP_WITH_FACTORY_IMPL(gko::preconditioner::Ic,
                               PACK(typename LSolverType, typename IndexType),
                               PACK(LSolverType, IndexType));


ENABLE_SELECTION(ic_factory_select, call, std::shared_ptr<gko::LinOpFactory>,
                 get_actual_factory_type);
ENABLE_SELECTION(ic_select, call, std::shared_ptr<gko::LinOp>, get_actual_type);


constexpr auto ic_list = typename span_list<
    tt_list<solver::LowerTrs<>>,  // TODO: Can not find LSolverType in with
                                  // TT_LIST_G_PARTIAL, please condider adding
                                  // it into type_default.hpp if it reused for
                                  // many times.
    tt_list_g_t<handle_type::IndexType>>::type();


template <>
std::shared_ptr<gko::LinOpFactory> create_from_config<
    RM_LinOpFactory, RM_LinOpFactory::IcFactory, gko::LinOpFactory>(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    // get the template from base
    std::string base_string;
    if (item.HasMember("base")) {
        base_string = get_base_template(item["base"].GetString());
    }
    // get the individual type
    auto type_string = create_type_name(  // trick for clang-format
        /* TODO: can not find LSolverType with GET_DEFAULT_STRING_PARTIAL,
           please condider adding it into type_default.hpp if it reused for many
           times. */
        get_value_with_default(item, "LSolverType",
                               get_string<solver::LowerTrs<>>()),
        get_value_with_default(item, "IndexType",
                               get_default_string<handle_type::IndexType>()));
    // combine them together, base_string has higher priority than type_string
    auto combined = combine_template(base_string, remove_space(type_string));
    auto ptr = ic_factory_select<gko::preconditioner::Ic>(
        ic_list, [=](std::string key) { return key == combined; }, item, exec,
        linop, manager);
    return std::move(ptr);
}

template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Ic, gko::LinOp>(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    // get the template from base
    std::string base_string;
    if (item.HasMember("base")) {
        base_string = get_base_template(item["base"].GetString());
    }
    // get the individual type
    auto type_string = create_type_name(  // trick for clang-format
        /* TODO: can not find LSolverType with GET_DEFAULT_STRING_PARTIAL,
           please condider adding it into type_default.hpp if it reused for many
           times. */
        get_value_with_default(item, "LSolverType",
                               get_string<solver::LowerTrs<>>()),
        get_value_with_default(item, "IndexType",
                               get_default_string<handle_type::IndexType>()));
    // combine them together, base_string has higher priority than type_string
    auto combined = combine_template(base_string, remove_space(type_string));
    auto ptr = ic_select<gko::preconditioner::Ic>(
        ic_list, [=](std::string key) { return key == combined; }, item, exec,
        linop, manager);
    return std::move(ptr);
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_PRECONDITIONER_IC_HPP_
