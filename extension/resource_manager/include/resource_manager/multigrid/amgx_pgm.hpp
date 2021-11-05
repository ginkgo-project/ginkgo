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

#ifndef GKOEXT_RESOURCE_MANAGER_MULTIGRID_AMGX_PGM_HPP_
#define GKOEXT_RESOURCE_MANAGER_MULTIGRID_AMGX_PGM_HPP_


#include "resource_manager/base/element_types.hpp"
#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"

#include <type_traits>


namespace gko {
namespace extension {
namespace resource_manager {


template <typename ValueType, typename IndexType>
struct Generic<typename gko::multigrid::AmgxPgm<ValueType, IndexType>::Factory,
               gko::multigrid::AmgxPgm<ValueType, IndexType>> {
    using type = std::shared_ptr<
        typename gko::multigrid::AmgxPgm<ValueType, IndexType>::Factory>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY(PACK(gko::multigrid::AmgxPgm<ValueType, IndexType>),
                          manager, item, exec, linop);
            SET_VALUE(unsigned, max_iterations);
            SET_VALUE(double, max_unassigned_ratio);
            SET_VALUE(bool, deterministic);
            SET_VALUE(bool, skip_sorting);
            SET_EXECUTOR;
        }();

        std::cout << "123AMG" << std::endl;
        return ptr;
    }
};


SIMPLE_LINOP_WITH_FACTORY_IMPL(gko::multigrid::AmgxPgm,
                               PACK(typename ValueType, typename IndexType),
                               PACK(ValueType, IndexType));

ENABLE_SELECTION(amgxpgmfactory_select, call,
                 std::shared_ptr<gko::LinOpFactory>, get_actual_factory_type);
ENABLE_SELECTION(amgxpgm_select, call, std::shared_ptr<gko::LinOp>,
                 get_actual_type);
constexpr auto amgx_pgm_list =
    typename span_list<tt_list<double, float>,
                       tt_list<gko::int32, gko::int64>>::type();

template <>
std::shared_ptr<gko::LinOpFactory> create_from_config<
    RM_LinOpFactory, RM_LinOpFactory::AmgxPgmFactory, gko::LinOpFactory>(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    std::cout << "amgx_pgm_factory" << std::endl;
    // go though the type
    using namespace std::literals::string_literals;
    auto vt = get_value_with_default(item, "ValueType", default_valuetype);
    auto it = get_value_with_default(item, "IndexType", default_indextype);
    auto type_string = create_type_name(vt, it);
    auto ptr = amgxpgmfactory_select<gko::multigrid::AmgxPgm>(
        amgx_pgm_list, [=](std::string key) { return key == type_string; },
        item, exec, linop, manager);
    return ptr;
}


template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::AmgxPgm, gko::LinOp>(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    std::cout << "build_amgx_pgm" << std::endl;
    // go though the type
    using namespace std::literals::string_literals;
    auto vt = get_value_with_default(item, "ValueType", default_valuetype);
    auto it = get_value_with_default(item, "IndexType", default_indextype);
    auto type_string = create_type_name(vt, it);
    auto ptr = amgxpgm_select<gko::multigrid::AmgxPgm>(
        amgx_pgm_list, [=](std::string key) { return key == type_string; },
        item, exec, linop, manager);
    return ptr;
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_MULTIGRID_AMGX_PGM_HPP_
