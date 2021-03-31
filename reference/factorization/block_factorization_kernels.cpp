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


#include "core/components/prefix_sum.hpp"
#include "core/matrix/fbcsr_builder.hpp"


namespace gko {
namespace kernels {
namespace reference {
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
void add_diagonal_blocks(const std::shared_ptr<const ReferenceExecutor> exec,
                         matrix::Fbcsr<ValueType, IndexType> *const mtx,
                         bool /*is_sorted*/)
{
    const auto col_idxs = mtx->get_const_col_idxs();
    IndexType *const row_ptrs = mtx->get_row_ptrs();
    const int bs = mtx->get_block_size();
    const size_type nbnz = mtx->get_num_stored_blocks();
    range<accessor::block_col_major<const ValueType, 3>> values(
        mtx->get_const_values(), dim<3>(nbnz, bs, bs));
    const auto num_brows = static_cast<IndexType>(mtx->get_num_block_rows());
    const auto num_bcols = static_cast<IndexType>(mtx->get_num_block_cols());

    const auto missing_blocks =
        count_missing_elements(num_brows, num_bcols, col_idxs, row_ptrs);

    if (missing_blocks == 0) {
        return;
    }

    const auto old_nbnz = mtx->get_num_stored_blocks();
    const auto new_nbnz = old_nbnz + missing_blocks;
    Array<ValueType> new_values_array{exec, new_nbnz * bs * bs};
    Array<IndexType> new_col_idxs_array{exec, new_nbnz};
    range<accessor::block_col_major<ValueType, 3>> new_values(
        new_values_array.get_data(), dim<3>(new_nbnz, bs, bs));
    auto new_col_idxs = new_col_idxs_array.get_data();
    IndexType added_blocks{};
    // row_ptrs will be updated in-place

    for (IndexType row = 0; row < num_brows; ++row) {
        bool diagonal_handled{false};
        const IndexType old_row_ptrs_start{row_ptrs[row]};
        const IndexType old_row_ptrs_end{row_ptrs[row + 1]};
        const IndexType new_row_ptrs_start = old_row_ptrs_start + added_blocks;

        row_ptrs[row] = new_row_ptrs_start;
        for (IndexType old_idx = old_row_ptrs_start; old_idx < old_row_ptrs_end;
             ++old_idx) {
            auto new_idx = old_idx + added_blocks;
            const auto col_idx = col_idxs[old_idx];
            if (!diagonal_handled && col_idx > row) {
                const auto start_cols = col_idxs + old_idx;
                const auto end_cols = col_idxs + old_row_ptrs_end;
                // expect row to not be sorted, so search for a diagonal entry
                if (std::find(start_cols, end_cols, row) != end_cols) {
                    // no need to add diagonal since diagonal is already present
                    diagonal_handled = true;
                }
                // if diagonal was not found, add it
                if (!diagonal_handled) {
                    for (int i = 0; i < bs; i++)
                        for (int j = 0; j < bs; j++)
                            new_values(new_idx, i, j) = zero<ValueType>();
                    new_col_idxs[new_idx] = row;
                    ++added_blocks;
                    new_idx = old_idx + added_blocks;
                    diagonal_handled = true;
                }
            }
            if (row >= num_bcols || col_idx == row) {
                diagonal_handled = true;
            }
            for (int i = 0; i < bs; i++)
                for (int j = 0; j < bs; j++)
                    new_values(new_idx, i, j) = values(old_idx, i, j);
            new_col_idxs[new_idx] = col_idx;
        }
        if (row < num_bcols && !diagonal_handled) {
            const auto new_idx = old_row_ptrs_end + added_blocks;
            for (int i = 0; i < bs; i++)
                for (int j = 0; j < bs; j++)
                    new_values(new_idx, i, j) = zero<ValueType>();
            new_col_idxs[new_idx] = row;
            diagonal_handled = true;
            ++added_blocks;
        }
    }
    row_ptrs[num_brows] = new_nbnz;

    matrix::FbcsrBuilder<ValueType, IndexType> mtx_builder{mtx};
    mtx_builder.get_value_array() = std::move(new_values_array);
    mtx_builder.get_col_idx_array() = std::move(new_col_idxs_array);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_BLOCKS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_BLU(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const system_matrix,
    IndexType *const l_row_ptrs, IndexType *const u_row_ptrs)
{
    const auto row_ptrs = system_matrix->get_const_row_ptrs();
    const auto col_idxs = system_matrix->get_const_col_idxs();
    IndexType l_nbnz{};
    IndexType u_nbnz{};

    l_row_ptrs[0] = 0;
    u_row_ptrs[0] = 0;
    for (IndexType row = 0; row < system_matrix->get_num_block_rows(); ++row) {
        for (IndexType el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            const IndexType col = col_idxs[el];
            // don't count diagonal
            if (col < row) l_nbnz++;
            if (col > row) u_nbnz++;
        }
        // add diagonal now
        l_nbnz++;
        u_nbnz++;
        l_row_ptrs[row + 1] = l_nbnz;
        u_row_ptrs[row + 1] = u_nbnz;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_BLU_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_BLU(
    const std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const system_matrix,
    matrix::Fbcsr<ValueType, IndexType> *const fb_l,
    matrix::Fbcsr<ValueType, IndexType> *const fb_u)
{
    using Dbv = range<accessor::block_col_major<ValueType, 3>>;
    using CDbv = range<accessor::block_col_major<const ValueType, 3>>;

    const int bs = system_matrix->get_block_size();
    const auto nbnz = system_matrix->get_num_stored_blocks();
    const auto row_ptrs = system_matrix->get_const_row_ptrs();
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const CDbv vals(system_matrix->get_const_values(), dim<3>(nbnz, bs, bs));

    const auto row_ptrs_l = fb_l->get_const_row_ptrs();
    auto const col_idxs_l = fb_l->get_col_idxs();
    const auto l_nbnz = fb_l->get_num_stored_blocks();
    Dbv vals_l(fb_l->get_values(), dim<3>(l_nbnz, bs, bs));

    const auto row_ptrs_u = fb_u->get_const_row_ptrs();
    auto const col_idxs_u = fb_u->get_col_idxs();
    const auto u_nbnz = fb_u->get_num_stored_blocks();
    Dbv vals_u(fb_u->get_values(), dim<3>(u_nbnz, bs, bs));

    for (IndexType row = 0; row < system_matrix->get_num_block_rows(); ++row) {
        IndexType current_index_l = row_ptrs_l[row];
        IndexType current_index_u = row_ptrs_u[row];
        const auto l_diag_idx = row_ptrs_l[row + 1] - 1;
        // const auto u_diag_idx = row_ptrs_u[row];

        // if there is no diagonal value, set diag blocks of U to identity
        // diag blocks of L are always identity
        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < bs; j++) {
                // vals_u(u_diag_idx, i, j) = zero<ValueType>();
                vals_l(l_diag_idx, i, j) = zero<ValueType>();
            }
            // vals_u(u_diag_idx, i, i) = one<ValueType>();
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
}  // namespace reference
}  // namespace kernels
}  // namespace gko
