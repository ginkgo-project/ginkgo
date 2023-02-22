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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_MATRIX_FFT_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_MATRIX_FFT_HPP_


#include <ginkgo/core/matrix/fft.hpp>


#include "resource_manager/base/generic_constructor.hpp"
#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"
#include "resource_manager/base/type_default.hpp"
#include "resource_manager/base/type_pack.hpp"
#include "resource_manager/base/type_resolving.hpp"
#include "resource_manager/base/type_string.hpp"
#include "resource_manager/base/types.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


template <>
struct Generic<gko::matrix::Fft> {
    using type = std::shared_ptr<gko::matrix::Fft>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto exec_ptr =
            get_pointer_check<Executor>(item, "exec", exec, linop, manager);
        auto size1 = get_value_with_default(item, "size1", size_type{});
        auto inverse = get_value_with_default(item, "inverse", false);
        auto ptr = share(gko::matrix::Fft::create(exec_ptr, size1, inverse));

        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};

template <>
struct Generic<gko::matrix::Fft2> {
    using type = std::shared_ptr<gko::matrix::Fft2>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto exec_ptr =
            get_pointer_check<Executor>(item, "exec", exec, linop, manager);
        auto size1 = get_value_with_default(item, "size1", size_type{});
        auto size2 = get_value_with_default(item, "size2", size1);
        auto inverse = get_value_with_default(item, "inverse", false);
        auto ptr =
            share(gko::matrix::Fft2::create(exec_ptr, size1, size2, inverse));

        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};

template <>
struct Generic<gko::matrix::Fft3> {
    using type = std::shared_ptr<gko::matrix::Fft3>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto exec_ptr =
            get_pointer_check<Executor>(item, "exec", exec, linop, manager);
        auto size1 = get_value_with_default(item, "size1", size_type{});
        auto size2 = get_value_with_default(item, "size2", size1);
        auto size3 = get_value_with_default(item, "size3", size2);
        auto inverse = get_value_with_default(item, "inverse", false);
        auto ptr = share(
            gko::matrix::Fft3::create(exec_ptr, size1, size2, size3, inverse));

        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};


IMPLEMENT_BRIDGE(RM_LinOp, Fft, gko::matrix::Fft);
IMPLEMENT_BRIDGE(RM_LinOp, Fft2, gko::matrix::Fft2);
IMPLEMENT_BRIDGE(RM_LinOp, Fft3, gko::matrix::Fft3);


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_MATRIX_FFT_HPP_
