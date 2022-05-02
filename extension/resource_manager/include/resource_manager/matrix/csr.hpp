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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_MATRIX_CSR_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_MATRIX_CSR_HPP_


#include <fstream>


#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/matrix/csr.hpp>


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


// TODO: Please add this header file into resource_manager/resource_manager.hpp
// TODO: Please add the corresponding to the resource_manager/base/types.hpp
// Add _expand(Csr) to ENUM_LINOP
// If need to override the generated enum for RM, use RM_CLASS or
// RM_CLASS_FACTORY env and rerun the generated script. Or replace the
// (RM_LinOpFactory::)CsrFactory and (RM_LinOp::)Csr and their snake case in
// IMPLEMENT_BRIDGE, ENABLE_SELECTION, *_select, ...


template <typename ValueType, typename IndexType>
struct Generic<gko::matrix::Csr<ValueType, IndexType>> {
    using type = std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto exec_ptr =
            get_pointer_check<Executor>(item, "exec", exec, linop, manager);
        auto size = get_value_with_default(item, "dim", gko::dim<2>{});
        // TODO: consider other thing from constructor
        auto ptr = share(
            gko::matrix::Csr<ValueType, IndexType>::create(exec_ptr, size));

        if (item.HasMember("read")) {
            std::ifstream mtx_fd(item["read"].GetString());
            auto data = gko::read_raw<
                typename gko::matrix::Csr<ValueType, IndexType>::value_type,
                typename gko::matrix::Csr<ValueType, IndexType>::index_type>(
                mtx_fd);
            ptr->read(data);
        }

        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};


ENABLE_SELECTION(csr_select, call, std::shared_ptr<gko::LinOp>,
                 get_actual_type);


constexpr auto csr_list =
    typename span_list<tt_list_g_t<handle_type::ValueType>,
                       tt_list_g_t<handle_type::IndexType>>::type();


template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Csr, gko::LinOp>(
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
        get_value_with_default(item, "ValueType",
                               get_default_string<handle_type::ValueType>()),
        get_value_with_default(item, "IndexType",
                               get_default_string<handle_type::IndexType>()));
    // combine them together, base_string has higher priority than type_string
    auto combined = combine_template(base_string, remove_space(type_string));
    auto ptr = csr_select<gko::matrix::Csr>(
        csr_list, [=](std::string key) { return key == combined; }, item, exec,
        linop, manager);
    return std::move(ptr);
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_MATRIX_CSR_HPP_
