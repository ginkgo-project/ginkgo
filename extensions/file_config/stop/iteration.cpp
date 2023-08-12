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

#include <ginkgo/core/stop/iteration.hpp>


#include "file_config/base/generic_constructor.hpp"
#include "file_config/base/helper.hpp"
#include "file_config/base/json_helper.hpp"
#include "file_config/base/macro_impl_helper.hpp"
#include "file_config/base/type_default.hpp"
#include "file_config/base/type_pack.hpp"
#include "file_config/base/type_resolving.hpp"
#include "file_config/base/type_string.hpp"
#include "file_config/base/types.hpp"


namespace gko {
namespace extensions {
namespace file_config {


template <>
struct Generic<typename gko::stop::Iteration::Factory, gko::stop::Iteration> {
    using type = std::shared_ptr<typename gko::stop::Iteration::Factory>;
    static type build(const nlohmann::json& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY(gko::stop::Iteration, manager, item, exec, linop);
            SET_VALUE(size_type, max_iters);
            SET_EXECUTOR;
        }();
        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};


IMPLEMENT_BRIDGE(RM_CriterionFactory, IterationFactory,
                 gko::stop::Iteration::Factory);


}  // namespace file_config
}  // namespace extensions
}  // namespace gko
