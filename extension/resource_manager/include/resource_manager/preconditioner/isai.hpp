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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_PRECONDITIONER_ISAI_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_PRECONDITIONER_ISAI_HPP_


#include <type_traits>


#include "resource_manager/base/element_types.hpp"
#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"
#include "resource_manager/base/type_list.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


// TODO: Please add the corresponding to the resource_manager/base/types.hpp
// Add _expand(IsaiFactory) to ENUM_LINOPFACTORY
// Add _expand(Isai) to ENUM_LINOP
// If need to override the generated enum for RM, use RM_CLASS or
// RM_CLASS_FACTORY env and rerun the generated script. Or replace the
// (RM_LinOpFactory::)IsaiFactory and (RM_LinOp::)Isai and their snake case in
// IMPLEMENT_BRIDGE, ENABLE_SELECTION, *_select, ...


template <isai_type IsaiType, typename ValueType, typename IndexType>
struct Generic<
    typename gko::preconditioner::Isai<IsaiType, ValueType, IndexType>::Factory,
    gko::preconditioner::Isai<IsaiType, ValueType, IndexType>> {
    using type =
        std::shared_ptr<typename gko::preconditioner::Isai<IsaiType, ValueType,
                                                           IndexType>::Factory>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY(
                PACK(gko::preconditioner::Isai<IsaiType, ValueType, IndexType>),
                manager, item, exec, linop);
            SET_VALUE(bool, skip_sorting);
            SET_VALUE(int, sparsity_power);
            SET_VALUE(size_type, excess_limit);
            SET_POINTER(LinOpFactory, excess_solver_factory);
            SET_EXECUTOR;
        }();
        return std::move(ptr);
    }
};


SIMPLE_LINOP_WITH_FACTORY_IMPL(gko::preconditioner::Isai,
                               PACK(isai_type IsaiType, typename ValueType,
                                    typename IndexType),
                               PACK(IsaiType, ValueType, IndexType));


// TODO: the class contain non type template, please create corresponding
// actual_type like following
/*
template <isai_type IsaiType, typename ValueType, typename IndexType>
struct actual_type<type_list<
    std::integral_constant<RM_LinOp, RM_LinOp::Isai>,
std::integral_constant<isai_type, IsaiType>, ValueType, IndexType>> { using type
= gko::preconditioner::Isai<IsaiType, ValueType, IndexType>;
};
*/
ENABLE_SELECTION_ID(isai_factory_select, call,
                    std::shared_ptr<gko::LinOpFactory>, get_actual_factory_type,
                    RM_LinOp, Isai);
ENABLE_SELECTION_ID(isai_select, call, std::shared_ptr<gko::LinOp>,
                    get_actual_type, RM_LinOp, Isai);


constexpr auto isai_list =
    typename span_list</* TODO: can not find IsaiType in tt_list_g, please
                          condider add it if it reused for many times*/
                       tt_list<>, tt_list_g_t<handle_type::ValueType>,
                       tt_list_g_t<handle_type::IndexType>>::type();


template <>
std::shared_ptr<gko::LinOpFactory> create_from_config<
    RM_LinOpFactory, RM_LinOpFactory::IsaiFactory, gko::LinOpFactory>(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    // go though the type
    auto type_string = create_type_name(  // trick for clang-format
        /*TODO: can not find IsaiType in get_default_string, please condider add
           it if it reused for many times*/
        get_required_value<std::string>(item, "IsaiType"),
        get_value_with_default(item, "ValueType",
                               get_default_string<handle_type::ValueType>()),
        get_value_with_default(item, "IndexType",
                               get_default_string<handle_type::IndexType>()));
    auto ptr = isai_factory_select<type_list>(
        isai_list, [=](std::string key) { return key == type_string; }, item,
        exec, linop, manager);
    return std::move(ptr);
}

template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Isai, gko::LinOp>(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    // go though the type
    auto type_string = create_type_name(  // trick for clang-format
        /*TODO: can not find IsaiType in get_default_string, please condider add
           it if it reused for many times*/
        get_required_value<std::string>(item, "IsaiType"),
        get_value_with_default(item, "ValueType",
                               get_default_string<handle_type::ValueType>()),
        get_value_with_default(item, "IndexType",
                               get_default_string<handle_type::IndexType>()));
    auto ptr = isai_select<type_list>(
        isai_list, [=](std::string key) { return key == type_string; }, item,
        exec, linop, manager);
    return std::move(ptr);
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_PRECONDITIONER_ISAI_HPP_
