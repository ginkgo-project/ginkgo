/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/reorder/metis_fill_reduce_kernels.hpp"


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#if GKO_HAVE_METIS
#include <metis.h>
#endif


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The metis fill reduce namespace.
 *
 * @ingroup reorder
 */
namespace metis_fill_reduce {


template <typename IndexType>
void get_permutation(std::shared_ptr<const ReferenceExecutor> exec,
                     const gko::size_type num_vertices,
                     const IndexType *mat_row_ptrs,
                     const IndexType *mat_col_idxs,
                     const IndexType *vertex_weights, IndexType *permutation,
                     IndexType *inv_permutation)
{}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_METIS_FILL_REDUCE_GET_PERMUTATION_KERNEL);


template <typename ValueType, typename IndexType>
void construct_inverse_permutation_matrix(
    std::shared_ptr<const ReferenceExecutor> exec,
    const IndexType *inv_permutation,
    gko::matrix::Csr<ValueType, IndexType> *inverse_permutation_matrix)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_METIS_FILL_REDUCE_CONSTRUCT_INVERSE_PERMUTATION_KERNEL);


template <typename ValueType, typename IndexType>
void construct_permutation_matrix(
    std::shared_ptr<const ReferenceExecutor> exec, const IndexType *permutation,
    gko::matrix::Csr<ValueType, IndexType> *permutation_matrix)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_METIS_FILL_REDUCE_CONSTRUCT_PERMUTATION_KERNEL);


template <typename ValueType, typename IndexType>
void permute(std::shared_ptr<const ReferenceExecutor> exec,
             gko::matrix::Csr<ValueType, IndexType> *permutation_matrix,
             gko::LinOp *to_permute)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_METIS_FILL_REDUCE_PERMUTE_KERNEL);


}  // namespace metis_fill_reduce
}  // namespace reference
}  // namespace kernels
}  // namespace gko
