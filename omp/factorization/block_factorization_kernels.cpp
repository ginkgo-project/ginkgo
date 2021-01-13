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

#include "core/factorization/block_factorization_kernels.hpp"


#include <algorithm>
#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "core/components/fixed_block.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/fbcsr_builder.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


template <typename IndexType>
IndexType count_missing_elements(const IndexType num_rows,
                                 const IndexType num_cols,
                                 const IndexType *const col_idxs,
                                 const IndexType *const row_ptrs)
{
    IndexType missing_elements{};
    // if row >= num_cols, diagonal elements no longer exist
#pragma omp parallel for reduction(+ : missing_elements)
    for (IndexType row = 0; row < num_rows && row < num_cols; ++row) {
        bool was_diagonal_found{false};
        for (IndexType idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            const auto col = col_idxs[idx];
            if (col == row) {
                was_diagonal_found = true;
                break;
            }
        }
        if (!was_diagonal_found) {
            ++missing_elements;
        }
    }
    return missing_elements;
}


template <typename ValueType, typename IndexType>
void add_diagonal_blocks(const std::shared_ptr<const OmpExecutor> exec,
                         matrix::Fbcsr<ValueType, IndexType> *const mtx,
                         bool /*is_sorted*/) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_BLOCKS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_BLU(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const system_matrix,
    IndexType *const l_row_ptrs,
    IndexType *const u_row_ptrs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_BLU_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_BLU(
    const std::shared_ptr<const OmpExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const system_matrix,
    matrix::Fbcsr<ValueType, IndexType> *const fb_l,
    matrix::Fbcsr<ValueType, IndexType> *const fb_u)
{
    using Dbv = blockutils::DenseBlocksView<ValueType, IndexType>;
    using CDbv = blockutils::DenseBlocksView<const ValueType, IndexType>;

    const int bs = system_matrix->get_block_size();
    const auto row_ptrs = system_matrix->get_const_row_ptrs();
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const CDbv vals(system_matrix->get_const_values(), bs, bs);

    const auto row_ptrs_l = fb_l->get_const_row_ptrs();
    const auto col_idxs_l = fb_l->get_col_idxs();
    Dbv vals_l(fb_l->get_values(), bs, bs);

    const auto row_ptrs_u = fb_u->get_const_row_ptrs();
    const auto col_idxs_u = fb_u->get_col_idxs();
    Dbv vals_u(fb_u->get_values(), bs, bs);

#pragma omp parallel for
    for (IndexType row = 0; row < system_matrix->get_num_block_rows(); ++row) {
        IndexType current_index_l = row_ptrs_l[row];
        IndexType current_index_u =
            row_ptrs_u[row] + 1;  // we treat the diagonal separately
        const auto l_diag_idx = row_ptrs_l[row + 1] - 1;
        const auto u_diag_idx = row_ptrs_u[row];

        // if there is no diagonal value, set diag blocks to identity
        // diag blocks of L are always identity
        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < bs; j++) {
                vals_u(u_diag_idx, i, j) = zero<ValueType>();
                vals_l(l_diag_idx, i, j) = zero<ValueType>();
            }
            vals_u(u_diag_idx, i, i) = one<ValueType>();
            vals_l(l_diag_idx, i, i) = one<ValueType>();
        }
        col_idxs_l[l_diag_idx] = row;

        for (IndexType el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            const auto col = col_idxs[el];
            if (col < row) {
                col_idxs_l[current_index_l] = col;
                for (int i = 0; i < bs; i++)
                    for (int j = 0; j < bs; j++)
                        vals_l(current_index_l, i, j) = vals(el, i, j);
                ++current_index_l;
            } else {  // col >= row
                col_idxs_u[current_index_u] = col;
                for (int i = 0; i < bs; i++)
                    for (int j = 0; j < bs; j++)
                        vals_u(current_index_u, i, j) = vals(el, i, j);
                ++current_index_u;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_BLU_KERNEL);


}  // namespace factorization
}  // namespace omp
}  // namespace kernels
}  // namespace gko
