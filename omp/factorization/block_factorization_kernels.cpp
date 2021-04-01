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


#include "accessor/block_col_major.hpp"
#include "core/base/utils.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/factorization/factorization_kernels.hpp"
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


namespace kernel {


template <typename IndexType>
IndexType count_missing_elements(const IndexType num_rows,
                                 const IndexType num_cols,
                                 const IndexType *const col_idxs,
                                 const IndexType *const row_ptrs,
                                 IndexType *const brow_diag_add)
{
    IndexType missing_elements{};
    const IndexType min_dim = std::min(num_rows, num_cols);
#pragma omp parallel for reduction(+ : missing_elements)
    for (IndexType row = 0; row < num_rows; ++row) {
        brow_diag_add[row] = 0;
        if (row < min_dim) {
            bool was_diagonal_found{false};
            for (IndexType idx = row_ptrs[row]; idx < row_ptrs[row + 1];
                 ++idx) {
                const auto col = col_idxs[idx];
                if (col == row) {
                    was_diagonal_found = true;
                    break;
                }
            }
            if (!was_diagonal_found) {
                ++missing_elements;
                brow_diag_add[row] = 1;
            }
        }
    }
    return missing_elements;
}


template <int mat_blk_sz, typename ValueType, typename IndexType>
void add_missing_diagonal_blocks(const IndexType num_b_rows,
                                 const ValueType *const old_values,
                                 const IndexType *const old_col_idxs,
                                 const IndexType *const old_row_ptrs,
                                 ValueType *const new_values,
                                 IndexType *const new_col_idxs,
                                 const IndexType *const row_ptrs_addition)
{
    constexpr int mat_blk_sz_2 = mat_blk_sz * mat_blk_sz;
#pragma omp parallel for
    for (IndexType row = 0; row < num_b_rows; ++row) {
        const IndexType old_row_start{old_row_ptrs[row]};
        const IndexType old_row_end{old_row_ptrs[row + 1]};
        const IndexType new_row_start{old_row_start + row_ptrs_addition[row]};
        const IndexType new_row_end{old_row_end + row_ptrs_addition[row + 1]};

        // if no element needs to be added, do a simple copy
        if (new_row_end - new_row_start == old_row_end - old_row_start) {
            for (IndexType i = 0; i < new_row_end - new_row_start; ++i) {
                const IndexType new_idx = new_row_start + i;
                const IndexType old_idx = old_row_start + i;
#pragma omp simd
                for (int ib = 0; ib < mat_blk_sz_2; ib++) {
                    new_values[new_idx * mat_blk_sz_2 + ib] =
                        old_values[old_idx * mat_blk_sz_2 + ib];
                }
                new_col_idxs[new_idx] = old_col_idxs[old_idx];
            }
        } else {
            IndexType new_idx = new_row_start;
            bool diagonal_added{false};
            for (IndexType old_idx = old_row_start; old_idx < old_row_end;
                 ++old_idx) {
                const auto col_idx = old_col_idxs[old_idx];
                if (!diagonal_added && row < col_idx) {
#pragma omp simd
                    for (int ib = 0; ib < mat_blk_sz_2; ib++) {
                        new_values[new_idx * mat_blk_sz_2 + ib] =
                            zero<ValueType>();
                    }
                    new_col_idxs[new_idx] = row;
                    ++new_idx;
                    diagonal_added = true;
                }
#pragma omp simd
                for (int ib = 0; ib < mat_blk_sz_2; ib++) {
                    new_values[new_idx * mat_blk_sz_2 + ib] =
                        old_values[old_idx * mat_blk_sz_2 + ib];
                }
                new_col_idxs[new_idx] = col_idx;
                ++new_idx;
            }
            if (!diagonal_added) {
#pragma omp simd
                for (int ib = 0; ib < mat_blk_sz_2; ib++) {
                    new_values[new_idx * mat_blk_sz_2 + ib] = zero<ValueType>();
                }
                new_col_idxs[new_idx] = row;
                diagonal_added = true;
            }
        }
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void add_diagonal_blocks(const std::shared_ptr<const OmpExecutor> exec,
                         matrix::Fbcsr<ValueType, IndexType> *const mtx,
                         bool /*is_sorted*/)
{
    const auto mtx_size = mtx->get_size();
    const int bs = mtx->get_block_size();
    const auto num_rows = static_cast<IndexType>(mtx_size[0]);
    const auto num_cols = static_cast<IndexType>(mtx_size[1]);
    const IndexType num_brows = mtx->get_num_block_rows();
    const IndexType num_bcols = mtx->get_num_block_cols();
    const IndexType row_ptrs_size = num_brows + 1;

    Array<IndexType> row_ptrs_addition(exec, row_ptrs_size);

    auto old_values = mtx->get_const_values();
    auto old_col_idxs = mtx->get_const_col_idxs();
    auto old_row_ptrs = mtx->get_const_row_ptrs();
    auto row_ptrs_add = row_ptrs_addition.get_data();

    const auto missing_elems = kernel::count_missing_elements(
        num_brows, num_bcols, old_col_idxs, old_row_ptrs, row_ptrs_add);

    if (missing_elems == 0) {
        return;
    }

    components::prefix_sum(exec, row_ptrs_add, row_ptrs_size);
    exec->synchronize();

    const auto total_additions = row_ptrs_add[num_brows];
    const auto new_num_blocks = total_additions + mtx->get_num_stored_blocks();

    Array<ValueType> new_values_arr{exec, new_num_blocks * bs * bs};
    Array<IndexType> new_col_idxs_arr{exec, new_num_blocks};
    auto new_values = new_values_arr.get_data();
    auto new_col_idxs = new_col_idxs_arr.get_data();

    if (bs == 2) {
        kernel::add_missing_diagonal_blocks<2>(
            num_brows, old_values, old_col_idxs, old_row_ptrs, new_values,
            new_col_idxs, row_ptrs_add);
    } else if (bs == 3) {
        kernel::add_missing_diagonal_blocks<3>(
            num_brows, old_values, old_col_idxs, old_row_ptrs, new_values,
            new_col_idxs, row_ptrs_add);
    } else if (bs == 4) {
        kernel::add_missing_diagonal_blocks<4>(
            num_brows, old_values, old_col_idxs, old_row_ptrs, new_values,
            new_col_idxs, row_ptrs_add);
    } else if (bs == 7) {
        kernel::add_missing_diagonal_blocks<7>(
            num_brows, old_values, old_col_idxs, old_row_ptrs, new_values,
            new_col_idxs, row_ptrs_add);
    } else {
        throw ::gko::NotImplemented(__FILE__, __LINE__,
                                    "add_missing_diaginal_blocks bs>4");
    }

    const auto new_row_ptrs = mtx->get_row_ptrs();
#pragma omp parallel for simd
    for (IndexType i = 0; i < num_brows + 1; i++) {
        new_row_ptrs[i] += row_ptrs_add[i];
    }

    matrix::FbcsrBuilder<ValueType, IndexType> mtx_builder{mtx};
    mtx_builder.get_value_array() = std::move(new_values_arr);
    mtx_builder.get_col_idx_array() = std::move(new_col_idxs_arr);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_BLOCKS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_BLU(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const system_matrix,
    IndexType *const l_row_ptrs, IndexType *const u_row_ptrs)
{
    auto num_rows = system_matrix->get_num_block_rows();
    auto row_ptrs = system_matrix->get_const_row_ptrs();
    auto col_idxs = system_matrix->get_const_col_idxs();

    // Calculate the NNZ per row first
#pragma omp parallel for
    for (IndexType row = 0; row < num_rows; ++row) {
        size_type l_nnz{};
        size_type u_nnz{};
        for (IndexType el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            const IndexType col = col_idxs[el];
            // don't count diagonal
            l_nnz += col < row;
            u_nnz += col > row;
        }
        // add diagonal again
        l_row_ptrs[row] = l_nnz + 1;
        u_row_ptrs[row] = u_nnz + 1;
    }

    // Now, compute the prefix-sum, to get proper row_ptrs for L and U
    components::prefix_sum(exec, l_row_ptrs, num_rows + 1);
    components::prefix_sum(exec, u_row_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_BLU_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_BLU(
    const std::shared_ptr<const OmpExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const system_matrix,
    matrix::Fbcsr<ValueType, IndexType> *const fb_l,
    matrix::Fbcsr<ValueType, IndexType> *const fb_u)
{
    using Dbv = acc::range<acc::block_col_major<ValueType, 3>>;

    const int bs = system_matrix->get_block_size();
    const auto row_ptrs = system_matrix->get_const_row_ptrs();
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const ValueType *const sys_v_arr = system_matrix->get_const_values();

    const auto row_ptrs_l = fb_l->get_const_row_ptrs();
    const auto col_idxs_l = fb_l->get_col_idxs();
    ValueType *const l_v_arr = fb_l->get_values();
    const auto l_nbnz = fb_l->get_num_stored_blocks();
    Dbv vals_l(to_array<size_type>(l_nbnz, bs, bs), l_v_arr);

    const auto row_ptrs_u = fb_u->get_const_row_ptrs();
    const auto col_idxs_u = fb_u->get_col_idxs();
    ValueType *const u_v_arr = fb_u->get_values();
    const auto u_nbnz = fb_u->get_num_stored_blocks();
    Dbv vals_u(to_array<size_type>(u_nbnz, bs, bs), u_v_arr);

#pragma omp parallel for
    for (IndexType row = 0; row < system_matrix->get_num_block_rows(); ++row) {
        const auto l_diag_idx = row_ptrs_l[row + 1] - 1;

        // if there is no diagonal value, set diag blocks to identity
        // diag blocks of L are always identity
        for (int j = 0; j < bs; j++) {
            for (int i = 0; i < bs; i++) {
                vals_l(l_diag_idx, i, j) = zero<ValueType>();
            }
            vals_l(l_diag_idx, j, j) = one<ValueType>();
        }
        col_idxs_l[l_diag_idx] = row;

        IndexType current_index_l = row_ptrs_l[row];
        IndexType current_index_u = row_ptrs_u[row];

        for (IndexType el = row_ptrs[row]; el < row_ptrs[row + 1]; ++el) {
            const auto col = col_idxs[el];
            if (col < row) {
                col_idxs_l[current_index_l] = col;
#pragma omp simd
                for (int j = 0; j < bs * bs; j++) {
                    l_v_arr[current_index_l * bs * bs + j] =
                        sys_v_arr[el * bs * bs + j];
                    // vals_l(current_index_l, i, j) = vals(el, i, j);
                }
                ++current_index_l;
            } else {  // col >= row
                col_idxs_u[current_index_u] = col;
#pragma omp simd
                for (int j = 0; j < bs * bs; j++) {
                    // vals_u(current_index_u, i, j) = vals(el, i, j);
                    u_v_arr[current_index_u * bs * bs + j] =
                        sys_v_arr[el * bs * bs + j];
                }
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
