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

#ifndef GKOEXT_RESOURCE_MANAGER_PRECONDITIONER_ISAI_HPP_
#define GKOEXT_RESOURCE_MANAGER_PRECONDITIONER_ISAI_HPP_


#include "resource_manager/base/element_types.hpp"
#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"

#include <type_traits>


namespace gko {
namespace extension {
namespace resource_manager {


template <gko::preconditioner::isai_type isai_value, typename ValueType,
          typename IndexType>
struct Generic<typename gko::preconditioner::Isai<isai_value, ValueType,
                                                  IndexType>::Factory,
               gko::preconditioner::Isai<isai_value, ValueType, IndexType>> {
    using type = std::shared_ptr<typename gko::preconditioner::Isai<
        isai_value, ValueType, IndexType>::Factory>;
    static type build(rapidjson::Value &item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager *manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY(PACK(gko::preconditioner::Isai<isai_value, ValueType,
                                                         IndexType>),
                          manager, item, exec, linop);
            SET_VALUE(bool, skip_sorting);
            SET_VALUE(int, sparsity_power);
            SET_VALUE(size_type, excess_limit);
            // SET_POINTER(LinOpFactory, excess_solver_factory);
            SET_EXECUTOR;
        }();

        std::cout << "123" << std::endl;
        return ptr;
    }
};

// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::lower, double,
//                          gko::int32),
//                     PACK(isai_lower, double, gko::int32), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::upper, double,
//                          gko::int32),
//                     PACK(isai_upper, double, gko::int32), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::general, double,
//                          gko::int32),
//                     PACK(isai_general, double, gko::int32), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::spd, double,
//                          gko::int32),
//                     PACK(isai_spd, double, gko::int32), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::lower, float,
//                          gko::int32),
//                     PACK(isai_lower, float, gko::int32), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::upper, float,
//                          gko::int32),
//                     PACK(isai_upper, float, gko::int32), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::general, float,
//                          gko::int32),
//                     PACK(isai_general, float, gko::int32), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::spd, float,
//                          gko::int32),
//                     PACK(isai_spd, float, gko::int32), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::lower, double,
//                          gko::int64),
//                     PACK(isai_lower, double, gko::int64), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::upper, double,
//                          gko::int64),
//                     PACK(isai_upper, double, gko::int64), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::general, double,
//                          gko::int64),
//                     PACK(isai_general, double, gko::int64), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::spd, double,
//                          gko::int64),
//                     PACK(isai_spd, double, gko::int64), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::lower, float,
//                          gko::int64),
//                     PACK(isai_lower, float, gko::int64), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::upper, float,
//                          gko::int64),
//                     PACK(isai_upper, float, gko::int64), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::general, float,
//                          gko::int64),
//                     PACK(isai_general, float, gko::int64), Factory,
//                     build_isai_factory);
// CONNECT_GENERIC_SUB(gko::preconditioner::Isai,
//                     PACK(gko::preconditioner::isai_type::spd, float,
//                          gko::int64),
//                     PACK(isai_spd, float, gko::int64), Factory,
//                     build_isai_factory);


SIMPLE_LINOP_WITH_FACTORY_IMPL(gko::preconditioner::Isai,
                               PACK(gko::preconditioner::isai_type isai_value,
                                    typename ValueType, typename IndexType),
                               PACK(isai_value, ValueType, IndexType));

ENABLE_SELECTION(isaifactory_select, call, std::shared_ptr<gko::LinOpFactory>,
                 get_actual_factory_type);
ENABLE_SELECTION(isai_select, call, std::shared_ptr<gko::LinOp>,
                 get_actual_type);
constexpr auto isai_list =
    typename span_list<tt_list<isai_lower, isai_upper, isai_general, isai_spd>,
                       tt_list<double, float>,
                       tt_list<gko::int32, gko::int64>>::type();

template <>
std::shared_ptr<gko::LinOpFactory> create_from_config<
    RM_LinOpFactory, RM_LinOpFactory::IsaiFactory, gko::LinOpFactory>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::cout << "build_isai_factory" << std::endl;
    // go though the type
    // std::string vt{"double"};
    // if (item.HasMember("type")) {
    //     vt = item["type"].GetString();
    // }
    std::string vt{"isai_lower+double+int"};
    auto ptr = isaifactory_select<type_list>(
        isai_list, [=](std::string key) { return key == vt; }, item, exec,
        linop, manager);
    return ptr;
}

template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Isai, gko::LinOp>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::cout << "build_isai" << std::endl;
    // go though the type
    std::string vt{"isai_lower+double+int"};
    // if (item.HasMember("type")) {
    //     vt = item["type"].GetString();
    // }
    auto ptr = isai_select<type_list>(
        isai_list, [=](std::string key) { return key == vt; }, item, exec,
        linop, manager);
    return ptr;
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_PRECONDITIONER_ISAI_HPP_
