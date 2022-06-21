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

#include "core/multigrid/uniform_coarsening_kernels.hpp"


#include <memory>
#include <tuple>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/multigrid/uniform_coarsening.hpp>


#include "core/base/allocator.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The UNIFORM_COARSENING solver namespace.
 *
 * @ingroup uniform_coarsening
 */
namespace uniform_coarsening {


template <typename ValueType, typename IndexType>
void fill_restrict_op(std::shared_ptr<const DefaultExecutor> exec,
                      const Array<IndexType>* coarse_rows,
                      matrix::Csr<ValueType, IndexType>* restrict_op)
{
    auto num_rows = restrict_op->get_size()[0];
    auto num_cols = restrict_op->get_size()[1];
    GKO_ASSERT(num_cols == coarse_rows->get_num_elems());
    GKO_ASSERT(num_cols >= num_rows);
    auto coarse_data = coarse_rows->get_const_data();

    for (IndexType j = 0; j < num_cols; ++j) {
        if (coarse_data[j] >= 0) {
            restrict_op->get_col_idxs()[coarse_data[j]] = j;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UNIFORM_COARSENING_FILL_RESTRICT_OP);


template <typename IndexType>
void fill_incremental_indices(std::shared_ptr<const DefaultExecutor> exec,
                              size_type num_jumps,
                              Array<IndexType>* coarse_rows)
{
    for (IndexType i = 0; i < coarse_rows->get_num_elems(); i += num_jumps) {
        coarse_rows->get_data()[i] = i / num_jumps;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_UNIFORM_COARSENING_FILL_INCREMENTAL_INDICES);


}  // namespace uniform_coarsening
}  // namespace reference
}  // namespace kernels
}  // namespace gko
