/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/components/fixed_block.hpp"


namespace kernel {


/* SubwarpSize needs to be a power of 2.
 * Each subwarp works on one row.
 * \tparam matBlkSz  Fixed size of the dense blocks the matrix is made up of.
 */
template <int subwarpSize, int matBlkSz, typename ValueType, typename IndexType>
__global__
    __launch_bounds__(default_block_size) void add_missing_diagonal_blocks(
        const IndexType num_b_rows,
        const ValueType *const __restrict__ old_values,
        const IndexType *const __restrict__ old_col_idxs,
        const IndexType *const __restrict__ old_row_ptrs,
        ValueType *const __restrict__ new_values,
        IndexType *const __restrict__ new_col_idxs,
        const IndexType *const __restrict__ row_ptrs_addition)
{
    // Precaution in case not enough threads were created
    const auto total_subwarp_count =
        thread::get_subwarp_num_flat<subwarpSize, IndexType>();
    const auto begin_row =
        thread::get_subwarp_id_flat<subwarpSize, IndexType>();

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<subwarpSize>(thread_block);
    const auto sw_threadidx = subwarp_grp.thread_rank();

    constexpr int matBlkSz_2{matBlkSz * matBlkSz};

    for (auto row = begin_row; row < num_b_rows; row += total_subwarp_count) {
        const IndexType old_row_start{old_row_ptrs[row]};
        const IndexType old_row_end{old_row_ptrs[row + 1]};
        const IndexType new_row_start{old_row_start + row_ptrs_addition[row]};
        const IndexType new_row_end{old_row_end + row_ptrs_addition[row + 1]};

        // if no element needs to be added, do a simple copy of the whole row
        if (new_row_end - new_row_start == old_row_end - old_row_start) {
            // the subwarp handles the entire block-row
            // one thread per subwarp updates the block-column indices
            const IndexType blockRowSz =
                (new_row_end - new_row_start) * matBlkSz_2;
            for (IndexType iz = sw_threadidx; iz < blockRowSz;
                 iz += subwarpSize) {
                const IndexType new_idx = new_row_start + iz / matBlkSz_2;
                const IndexType old_idx = old_row_start + iz / matBlkSz_2;

                new_values[new_idx * matBlkSz_2 + iz] =
                    old_values[old_idx * matBlkSz_2 + iz];

                if (sw_threadidx % matBlkSz_2 == 0)
                    new_col_idxs[new_idx] = old_col_idxs[old_idx];
                subwarp_grp.sync();
            }
        } else {
            const IndexType oldBlockRowSz{(old_row_end - old_row_start) *
                                          matBlkSz_2};
            const IndexType oldBlkRowBase{old_row_start * matBlkSz_2};
            const IndexType newBlkRowBase{new_row_start * matBlkSz_2};
            ValueType *const new_values_brow = new_values + newBlkRowBase;
            const ValueType *const old_values_brow = old_values + oldBlkRowBase;
            bool diagonal_added{false};

            // This works for at least the cases where
            // - the entire ffs block is contained in one subwarp range.
            // - the entire subwarp range ins contained within the ffs block.

            for (IndexType old_iz = sw_threadidx, new_iz = sw_threadidx;
                 old_iz < oldBlockRowSz;
                 old_iz += subwarpSize, new_iz += subwarpSize) {
                const auto old_idx = old_row_start + old_iz / matBlkSz_2;
                const auto new_idx = new_row_start + new_iz / matBlkSz_2;

                const auto col_idx = old_col_idxs[old_idx];
                const bool diagonal_add_required =
                    !diagonal_added && row < col_idx;
                const auto sw_diag_req =
                    subwarp_grp.ballot(diagonal_add_required);

                if (sw_diag_req) {
                    const IndexType start_oldiz = subwarp_grp.shfl(old_iz, 0);

                    const auto first_diagreq_sw_tid = ffs(sw_diag_req) - 1;
                    const IndexType first_diagreq_oldiz =
                        subwarp_grp.shfl(old_iz, first_diagreq_sw_tid);
                    const bool ffsBlockStartNow =
                        (first_diagreq_oldiz % matBlkSz_2 == 0);
                    const IndexType startFfsMatBlock =
                        first_diagreq_oldiz / matBlkSz_2;
                    const IndexType endFfsMatBlock_oldiz =
                        (startFfsMatBlock + 1) * matBlkSz_2;
                    const bool swReachedFfsBlockEnd =
                        (start_oldiz + subwarpSize >= endFfsMatBlock_oldiz);
                    const IndexType end_oldiz = (!swReachedFfsBlockEnd)
                                                    ? start_oldiz + subwarpSize
                                                    : endFfsMatBlock_oldiz;
                    // const IndexType numInserted = end_oldiz -
                    // first_diagreq_oldiz;

                    if (old_iz >= first_diagreq_oldiz && old_iz < end_oldiz) {
                        new_values_brow[new_iz] = zero<ValueType>();
                        // Add new entry to column index array only if
                        //  the whole block is done in this iteration
                        if (ffsBlockStartNow &&
                            sw_threadidx == first_diagreq_sw_tid)
                            new_col_idxs[new_idx] = row;
                    }

                    // if first diagonal entry was inserted in a thread before
                    // this one
                    const bool is_thread_after_firstdiag =
                        (old_iz >= first_diagreq_oldiz);

                    const IndexType writing_new_iz =
                        new_iz + is_thread_after_firstdiag * matBlkSz_2;
                    new_values_brow[writing_new_iz] = old_values_brow[old_iz];

                    if (new_iz % matBlkSz_2 == 0) {
                        const IndexType write_new_idx =
                            new_idx +
                            (is_thread_after_firstdiag && ffsBlockStartNow);
                        new_col_idxs[write_new_idx] = col_idx;
                    }

                    // if first diagonal entry is inserted in a thread after
                    // this one, it needs to be considered after writing the
                    // values
                    // const bool is_thread_before_diag = (sw_threadidx <
                    // first_diagreq_sw_tid);

                    if (swReachedFfsBlockEnd) {
                        diagonal_added = true;
                        new_iz += matBlkSz_2;
                    }
                } else {
                    new_values_brow[new_iz] = old_values_brow[old_iz];
                    if (old_iz % matBlkSz_2 == 0)
                        new_col_idxs[new_idx] = col_idx;
                }
            }
            if (!diagonal_added && sw_threadidx < matBlkSz_2) {
                const IndexType new_idx = new_row_end - 1;
                const IndexType new_iz = new_idx * matBlkSz_2;
                new_values[new_iz + sw_threadidx] = zero<ValueType>();
                if (sw_threadidx == 0) new_col_idxs[new_idx] = row;
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void initialize_BLU(
    const size_type num_rows, const IndexType *const __restrict__ row_ptrs,
    const IndexType *const __restrict__ col_idxs,
    const ValueType *const __restrict__ values,
    const IndexType *const __restrict__ l_row_ptrs,
    IndexType *const __restrict__ l_col_idxs,
    ValueType *const __restrict__ l_values,
    const IndexType *const __restrict__ u_row_ptrs,
    IndexType *const __restrict__ u_col_idxs,
    ValueType *const __restrict__ u_values)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row < num_rows) {
        auto l_idx = l_row_ptrs[row];
        auto u_idx = u_row_ptrs[row] + 1;  // we treat the diagonal separately
        // default diagonal to one
        auto diag_val = one<ValueType>();
        for (size_type i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
            const auto col = col_idxs[i];
            const auto val = values[i];
            // save diagonal entry for later
            if (col == row) {
                diag_val = val;
            }
            if (col < row) {
                l_col_idxs[l_idx] = col;
                l_values[l_idx] = val;
                ++l_idx;
            }
            if (row < col) {
                u_col_idxs[u_idx] = col;
                u_values[u_idx] = val;
                ++u_idx;
            }
        }
        // store diagonal entries
        auto l_diag_idx = l_row_ptrs[row + 1] - 1;
        auto u_diag_idx = u_row_ptrs[row];
        l_col_idxs[l_diag_idx] = row;
        u_col_idxs[u_diag_idx] = row;
        l_values[l_diag_idx] = one<ValueType>();
        u_values[u_diag_idx] = diag_val;
    }
}


}  // namespace kernel
