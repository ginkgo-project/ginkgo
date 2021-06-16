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

#ifndef GKOEXT_RESOURCE_MANAGER_HPP_
#define GKOEXT_RESOURCE_MANAGER_HPP_

#include <map>
#include "resource_manager/base/generic_base_selector.hpp"
#include "resource_manager/base/generic_constructor.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/resource_manager.hpp"
#include "resource_manager/base/types.hpp"
#include "resource_manager/executor/executor.hpp"
#include "resource_manager/matrix/dense.hpp"
#include "resource_manager/solver/cg.hpp"
#include "resource_manager/stop/iteration.hpp"

namespace gko {
namespace extension {
namespace resource_manager {


IMPLEMENT_SELECTION(Executor, RM_Executor, ENUM_EXECUTER);
IMPLEMENT_SELECTION(LinOp, RM_LinOp, ENUM_LINOP);
IMPLEMENT_SELECTION(LinOpFactory, RM_LinOpFactory, ENUM_LINOPFACTORY);
IMPLEMENT_SELECTION(CriterionFactory, RM_CriterionFactory,
                    ENUM_CRITERIONFACTORY);


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKOEXT_RESOURCE_MANAGER_HPP_
