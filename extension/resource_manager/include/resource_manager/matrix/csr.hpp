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

#ifndef GKOEXT_RESOURCE_MANAGER_MATRIX_CSR_HPP_
#define GKOEXT_RESOURCE_MANAGER_MATRIX_CSR_HPP_


#include <type_traits>
#include "resource_manager/base/element_types.hpp"
#include "resource_manager/base/generic_constructor.hpp"
#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


template <typename V, typename I>
struct Generic<gko::matrix::Csr<V, I>> {
    using type = std::shared_ptr<gko::matrix::Csr<V, I>>;
    static type build(rapidjson::Value &item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager *manager)
    {
        using Csr = gko::matrix::Csr<V, I>;
        using strategy_type = typename Csr::strategy_type;
        auto exec_ptr =
            get_pointer_check<Executor>(manager, item, "exec", exec, linop);
        auto size = get_value_with_default(item, "dim", gko::dim<2>{});
        auto nnz = get_value_with_default(item, "nnz", 0);
        std::shared_ptr<strategy_type> strategy_ptr;
        auto strategy =
            get_value_with_default(item, "strategy", std::string("sparselib"));
        if (strategy == std::string("sparselib") ||
            strategy == std::string("cusparse")) {
            strategy_ptr = std::make_shared<typename Csr::sparselib>();
        } else if (strategy == std::string("automatical")) {
            if (auto explicit_exec =
                    std::dynamic_pointer_cast<const gko::CudaExecutor>(
                        exec_ptr)) {
                strategy_ptr =
                    std::make_shared<typename Csr::automatical>(explicit_exec);
            } else if (auto explicit_exec =
                           std::dynamic_pointer_cast<const gko::HipExecutor>(
                               exec_ptr)) {
                strategy_ptr =
                    std::make_shared<typename Csr::automatical>(explicit_exec);
            } else {
                strategy_ptr = std::make_shared<typename Csr::automatical>(256);
            }
        } else if (strategy == std::string("load_balance")) {
            if (auto explicit_exec =
                    std::dynamic_pointer_cast<const gko::CudaExecutor>(
                        exec_ptr)) {
                strategy_ptr =
                    std::make_shared<typename Csr::load_balance>(explicit_exec);
            } else if (auto explicit_exec =
                           std::dynamic_pointer_cast<const gko::HipExecutor>(
                               exec_ptr)) {
                strategy_ptr =
                    std::make_shared<typename Csr::load_balance>(explicit_exec);
            } else {
                strategy_ptr =
                    std::make_shared<typename Csr::load_balance>(256);
            }

        } else if (strategy == std::string("merge_path")) {
            strategy_ptr = std::make_shared<typename Csr::merge_path>();
        } else if (strategy == std::string("classical")) {
            strategy_ptr = std::make_shared<typename Csr::classical>();
        }
        auto ptr = share(
            gko::matrix::Csr<V, I>::create(exec_ptr, size, nnz, strategy_ptr));
        if (item.HasMember("read")) {
            std::ifstream mtx_fd(item["read"].GetString());
            auto data = gko::read_raw<V, I>(mtx_fd);
            ptr->read(data);
        }
        return std::move(ptr);
    }
};

ENABLE_SELECTION(csr_select, call, std::shared_ptr<gko::LinOp>, get_the_type);
constexpr auto csr_list =
    typename span_list<tt_list<double, float>,
                       tt_list<gko::int32, gko::int64>>::type();

template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Csr, gko::LinOp>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::cout << "build_csr" << std::endl;
    // go though the type
    std::string vt{default_valuetype};
    std::string it{default_indextype};
    if (item.HasMember("vt")) {
        vt = item["vt"].GetString();
    }
    if (item.HasMember("it")) {
        it = item["it"].GetString();
    }
    auto ptr = csr_select<gko::matrix::Csr>(
        csr_list, [=](std::string key) { return key == (vt + "+" + it); }, item,
        exec, linop, manager);
    return ptr;
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_MATRIX_CSR_HPP_
