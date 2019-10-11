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


#include <memory>
#include <vector>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/metis_types.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#if GKO_HAVE_METIS
#include <metis.h>
#endif


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The parallel ILU factorization namespace.
 *
 * @ingroup factor
 */
namespace metis_fill_reduce {


template <typename ValueType, typename IndexType>
void get_permutation(
    std::shared_ptr<const OmpExecutor> exec, size_type num_vertices,
    std::shared_ptr<matrix::SparsityCsr<ValueType, IndexType>> adjacency_matrix,
    std::shared_ptr<Array<IndexType>> vertex_weights,
    std::shared_ptr<matrix::Permutation<IndexType>> permutation_mat,
    std::shared_ptr<matrix::Permutation<IndexType>> inv_permutation_mat)
#if GKO_HAVE_METIS
{
    IndexType num_vtxs = static_cast<IndexType>(num_vertices);
    auto adj_ptrs = adjacency_matrix->get_row_ptrs();
    auto adj_idxs = adjacency_matrix->get_col_idxs();
    auto vtx_weights = vertex_weights->get_data();
    auto permutation_arr = permutation_mat->get_permutation();
    auto inv_permutation_arr = inv_permutation_mat->get_permutation();
    idx_t options[METIS_NOPTIONS];
    GKO_ASSERT_NO_METIS_ERRORS(METIS_SetDefaultOptions(options));
    GKO_ASSERT_NO_METIS_ERRORS(
        METIS_NodeND(&num_vtxs, adj_ptrs, adj_idxs, vtx_weights, options,
                     permutation_arr, inv_permutation_arr));
}
#else
{
    std::vector<IndexType> tmp(num_vertices, 0);
    std::iota(tmp.begin(), tmp.end(), 0);
    for (auto i = 0; i < num_vertices; ++i) {
        permutation_mat->get_permutation()[i] = tmp[i];
    }
}
#endif

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_METIS_INDEX_TYPE(
    GKO_DECLARE_METIS_FILL_REDUCE_GET_PERMUTATION_KERNEL);


}  // namespace metis_fill_reduce
}  // namespace omp
}  // namespace kernels
}  // namespace gko
