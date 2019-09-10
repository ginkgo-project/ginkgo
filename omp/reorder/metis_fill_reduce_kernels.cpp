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
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/metis_types.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


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
void remove_diagonal_elements(std::shared_ptr<const OmpExecutor> exec,
                              bool remove_diagonal_elements,
                              gko::matrix::Csr<ValueType, IndexType> *matrix,
                              IndexType *adj_ptrs, IndexType *adj_idxs)
#if GKO_HAVE_METIS
{
    auto num_rows = matrix->get_size()[0];
    auto zero = gko::zero<ValueType>();
    if (remove_diagonal_elements && matrix->get_values()[0] != zero) {
        for (auto i = 0; i <= num_rows; ++i) {
            adj_ptrs[i] = matrix->get_row_ptrs()[i] - i;
        }
        std::vector<IndexType> temp_idxs;
        auto row_ptrs = matrix->get_row_ptrs();
        auto col_idxs = matrix->get_col_idxs();
        for (auto i = 0; i < num_rows; ++i) {
            for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
                if (col_idxs[j] != i) {
                    temp_idxs.push_back(col_idxs[j]);
                }
            }
        }
        for (auto i = 0; i < temp_idxs.size(); ++i) {
            adj_idxs[i] = temp_idxs[i];
        }
    } else {
        for (auto i = 0; i <= num_rows; ++i) {
            adj_ptrs[i] = matrix->get_row_ptrs()[i];
        }
        for (auto i = 0; i < matrix->get_num_stored_elements(); ++i) {
            adj_idxs[i] = matrix->get_col_idxs()[i];
        }
    }
}
#else
{}
#endif

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_METIS_INDEX_TYPE(
    GKO_DECLARE_METIS_FILL_REDUCE_REMOVE_DIAGONAL_ELEMENTS_KERNEL);

template <typename IndexType>
void get_permutation(std::shared_ptr<const OmpExecutor> exec,
                     IndexType num_vertices, IndexType *adj_ptrs,
                     IndexType *adj_idxs, IndexType *vertex_weights,
                     IndexType *permutation, IndexType *inv_permutation)
#if GKO_HAVE_METIS
{
    idx_t options[METIS_NOPTIONS];
    GKO_ASSERT_NO_METIS_ERRORS(METIS_SetDefaultOptions(options));
    GKO_ASSERT_NO_METIS_ERRORS(METIS_NodeND(&num_vertices, adj_ptrs, adj_idxs,
                                            vertex_weights, options,
                                            permutation, inv_permutation));
}
#else
{}
#endif

GKO_INSTANTIATE_FOR_EACH_METIS_INDEX_TYPE(
    GKO_DECLARE_METIS_FILL_REDUCE_GET_PERMUTATION_KERNEL);


template <typename IndexType>
void permute(std::shared_ptr<const OmpExecutor> exec,
             gko::Array<IndexType> *permutation, gko::LinOp *to_permute)
#if GKO_HAVE_METIS
{}
#else
{}
#endif

GKO_INSTANTIATE_FOR_EACH_METIS_INDEX_TYPE(
    GKO_DECLARE_METIS_FILL_REDUCE_PERMUTE_KERNEL);


}  // namespace metis_fill_reduce
}  // namespace omp
}  // namespace kernels
}  // namespace gko
