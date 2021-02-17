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

#include "core/multigrid/mapping_kernels.hpp"


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/multigrid/mapping.hpp>


#include "core/base/allocator.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The MAPPING solver namespace.
 *
 * @ingroup mapping
 */
namespace mapping {


template <typename ValueType, typename IndexType>
void applyadd(std::shared_ptr<const DefaultExecutor> exec,
              const multigrid::Mapping<ValueType, IndexType> *source,
              const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x)
{
    auto x_vals = x->get_values();
    const auto x_stride = x->get_stride();
    auto x_dim = x->get_size();
    auto mapindex = source->get_const_mapping_index();
    auto num = source->get_num_stored_elements();
    auto is_restrict = source->is_restrict();
    for (size_type i = 0; i < num; i++) {
        const auto x_row = is_restrict ? mapindex[i] : i;
        const auto b_row = is_restrict ? i : mapindex[i];
        for (size_type j = 0; j < x_dim[1]; j++) {
            x->at(x_row, j) += b->at(b_row, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MAPPING_APPLYADD_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_applyadd(std::shared_ptr<const DefaultExecutor> exec,
                       const multigrid::Mapping<ValueType, IndexType> *source,
                       const matrix::Dense<ValueType> *alpha,
                       const matrix::Dense<ValueType> *b,
                       matrix::Dense<ValueType> *x)
{
    auto x_vals = x->get_values();
    const auto x_stride = x->get_stride();
    auto x_dim = x->get_size();
    auto mapindex = source->get_const_mapping_index();
    auto num = source->get_num_stored_elements();
    auto is_restrict = source->is_restrict();
    auto alpha_val = alpha->at(0, 0);
    for (size_type i = 0; i < num; i++) {
        const auto x_row = is_restrict ? mapindex[i] : i;
        const auto b_row = is_restrict ? i : mapindex[i];
        for (size_type j = 0; j < x_dim[1]; j++) {
            x->at(x_row, j) += alpha_val * b->at(b_row, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MAPPING_ADVANCED_APPLYADD_KERNEL);


}  // namespace mapping
}  // namespace reference
}  // namespace kernels
}  // namespace gko
