/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/isai.hpp>


#include "file_config/base/generic_constructor.hpp"
#include "file_config/base/helper.hpp"
#include "file_config/base/json_helper.hpp"
#include "file_config/base/macro_impl_helper.hpp"
#include "file_config/base/template_helper.hpp"
#include "file_config/base/type_default.hpp"
#include "file_config/base/type_pack.hpp"
#include "file_config/base/type_resolving.hpp"
#include "file_config/base/type_string.hpp"
#include "file_config/base/types.hpp"


namespace gko {
namespace extensions {
namespace file_config {


template <gko::preconditioner::isai_type IsaiType, typename ValueType,
          typename IndexType>
struct Generic<
    typename gko::preconditioner::Isai<IsaiType, ValueType, IndexType>::Factory,
    gko::preconditioner::Isai<IsaiType, ValueType, IndexType>> {
    using type =
        std::shared_ptr<typename gko::preconditioner::Isai<IsaiType, ValueType,
                                                           IndexType>::Factory>;
    static type build(const nlohmann::json& item,
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
        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};

SIMPLE_LINOP_WITH_FACTORY_IMPL(gko::preconditioner::Isai,
                               PACK(gko::preconditioner::isai_type IsaiType,
                                    typename ValueType, typename IndexType),
                               PACK(IsaiType, ValueType, IndexType));


ENABLE_SELECTION_ID(isai_factory_select, call,
                    std::shared_ptr<gko::LinOpFactory>, get_actual_factory_type,
                    RM_LinOp, Isai);
ENABLE_SELECTION_ID(isai_select, call, std::shared_ptr<gko::LinOp>,
                    get_actual_type, RM_LinOp, Isai);


constexpr auto isai_list =
    typename span_list<tt_list<isai_lower, isai_upper, isai_general, isai_spd>,
                       tt_list_g_t<handle_type::ValueType>,
                       tt_list_g_t<handle_type::IndexType>>::type();


template <>
std::shared_ptr<gko::LinOpFactory> create_from_config<
    RM_LinOpFactory, RM_LinOpFactory::IsaiFactory, gko::LinOpFactory>(
    const nlohmann::json& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    // get the template from base
    std::string base_string;
    if (item.contains("base")) {
        base_string = get_base_template(item["base"].get<std::string>());
    }
    // get the individual type
    auto type_string = create_type_name(  // trick for clang-format
        get_required_value<std::string>(item, "IsaiType"),
        get_value_with_default(item, "ValueType",
                               get_default_string<handle_type::ValueType>()),
        get_value_with_default(item, "IndexType",
                               get_default_string<handle_type::IndexType>()));
    // combine them together, base_string has higher priority than type_string
    auto combined = combine_template(base_string, remove_space(type_string));
    auto ptr = isai_factory_select<type_list>(
        isai_list, [=](std::string key) { return key == combined; }, item, exec,
        linop, manager);
    return std::move(ptr);
}

template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Isai, gko::LinOp>(
    const nlohmann::json& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    // get the template from base
    std::string base_string;
    if (item.contains("base")) {
        base_string = get_base_template(item["base"].get<std::string>());
    }
    // get the individual type
    auto type_string = create_type_name(  // trick for clang-format
        get_required_value<std::string>(item, "IsaiType"),
        get_value_with_default(item, "ValueType",
                               get_default_string<handle_type::ValueType>()),
        get_value_with_default(item, "IndexType",
                               get_default_string<handle_type::IndexType>()));
    // combine them together, base_string has higher priority than type_string
    auto combined = combine_template(base_string, remove_space(type_string));
    auto ptr = isai_select<type_list>(
        isai_list, [=](std::string key) { return key == combined; }, item, exec,
        linop, manager);
    return std::move(ptr);
}


}  // namespace file_config
}  // namespace extensions
}  // namespace gko
