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

#ifndef GKOEXT_RESOURCE_MANAGER_MATRIX_DENSE_HPP_
#define GKOEXT_RESOURCE_MANAGER_MATRIX_DENSE_HPP_


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


template <typename T>
struct Generic<gko::matrix::Dense<T>> {
    using type = std::shared_ptr<gko::matrix::Dense<T>>;
    static type build(rapidjson::Value &item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager *manager)
    {
        std::cout << "is_double?" << std::is_same<T, double>::value
                  << std::endl;
        std::cout << "is_float?" << std::is_same<T, float>::value << std::endl;

        // std::shared_ptr<Executor> exec_ptr;
        auto exec_ptr =
            get_pointer_check<Executor>(manager, item, "exec", exec, linop);
        auto size = get_value_with_default(item, "dim", gko::dim<2>{});
        auto stride = get_value_with_default(item, "stride", size[1]);
        auto ptr = share(gko::matrix::Dense<T>::create(exec_ptr, size, stride));
        if (item.HasMember("read")) {
            std::ifstream mtx_fd(item["read"].GetString());
            auto data = gko::read_raw<T>(mtx_fd);
            ptr->read(data);
        }
        std::cout << ptr->get_size()[0] << " " << ptr->get_size()[1] << " "
                  << ptr->get_stride() << std::endl;
        return std::move(ptr);
    }
};

ENABLE_SELECTION(dense_select, call, std::shared_ptr<gko::LinOp>, get_the_type);
constexpr auto dense_list = type_list<double, float>();

template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Dense, gko::LinOp>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    std::cout << "build_dense" << std::endl;
    // go though the type
    std::string vt{default_valuetype};
    if (item.HasMember("type")) {
        vt = item["type"].GetString();
    }
    auto ptr = dense_select<gko::matrix::Dense>(
        dense_list, [=](std::string key) { return key == vt; }, item, exec,
        linop, manager);
    return ptr;
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_MATRIX_DENSE_HPP_
