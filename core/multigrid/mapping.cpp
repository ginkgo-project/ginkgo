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

#include <ginkgo/core/multigrid/mapping.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/components/fill_array.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/mapping_kernels.hpp"


namespace gko {
namespace multigrid {
namespace mapping {


GKO_REGISTER_OPERATION(applyadd, mapping::applyadd);
GKO_REGISTER_OPERATION(advanced_applyadd, mapping::advanced_applyadd);


}  // namespace mapping


template <typename ValueType, typename IndexType>
void Mapping<ValueType, IndexType>::apply2_impl(const LinOp *b, LinOp *x) const
{
    auto exec = this->get_executor();
    exec->run(mapping::make_applyadd(this, as<matrix::Dense<ValueType>>(b),
                                     as<matrix::Dense<ValueType>>(x)));
}


template <typename ValueType, typename IndexType>
void Mapping<ValueType, IndexType>::apply2_impl(const LinOp *alpha,
                                                const LinOp *b, LinOp *x) const
{
    auto exec = this->get_executor();
    exec->run(mapping::make_advanced_applyadd(
        this, as<matrix::Dense<ValueType>>(alpha),
        as<matrix::Dense<ValueType>>(b), as<matrix::Dense<ValueType>>(x)));
}


#define GKO_DECLARE_MAPPING(_vtype, _itype) class Mapping<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MAPPING);


}  // namespace multigrid
}  // namespace gko
