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

namespace kernel {


/* SubwarpSize needs to be a power of 2.
 * Each subwarp works on one row.
 * \tparam matBlkSz  Fixed size of the dense blocks the matrix is made up of.
 */
template <int matBlkSz, int subwarpSize, typename ValueType, typename IndexType>
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

                new_values[new_row_start * matBlkSz_2 + iz] =
                    old_values[old_row_start * matBlkSz_2 + iz];

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


namespace detail {

/// Extents of positions relevant for some purpose in a block-row
template <typename IndexType>
struct BRowPos {
    IndexType cur_iz;  ///< Current position in block-row
    IndexType end_iz;  ///< One-past-the-last position in block-row
};

/*
 * Compute the positions in 'this' block row that satisfy a condition and
 *  are to be operated on in the 'current' iteration by a subwarp
 *
 * \param[in] sgrp  The subwarp handling the current block row
 * \param[in] pred  The condition to be met by the positions identified
 * \param[in] start_iz  The (common) starting position in the relevant
 *   values array, w.r.t. the current block row, so far
 */
template <int subwarpSize, typename IndexType>
__device__ BRowPos<IndexType> getBRowPos(
    const group::thread_block_tile<subwarpSize> &sgrp, const bool pred,
    const IndexType start_iz)
{
    const auto sw_threadidx = sgrp.thread_rank();
    BRowPos<IndexType> brp{-1, 0};
    const unsigned ballot = sgrp.ballot(pred);
    if (ballot) {
        const int start_low_stid = __ffs(ballot) - 1;
        const int last_low_stid =
            subwarpSize - ((__ffs(__brev(ballot)) - 1) % subwarpSize) - 1;
        brp.cur_iz = start_iz + (sw_threadidx - start_low_stid);
        brp.end_iz = sgrp.shfl(brp.cur_iz, last_low_stid) + 1;
    }
    return brp;
}

}  // namespace detail


template <int matBlkSz, int subwarpSize, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void initialize_BLU(
    const size_type num_b_rows, const IndexType *const __restrict__ row_ptrs,
    const IndexType *const __restrict__ col_idxs,
    const ValueType *const __restrict__ values,
    const IndexType *const __restrict__ l_row_ptrs,
    IndexType *const __restrict__ l_col_idxs,
    ValueType *const __restrict__ l_values,
    const IndexType *const __restrict__ u_row_ptrs,
    IndexType *const __restrict__ u_col_idxs,
    ValueType *const __restrict__ u_values)
{
    static_assert(subwarpSize <= 32, "32-bit integer intrinsics are used");

    constexpr int matBlkSz_2 = matBlkSz * matBlkSz;
    constexpr int warps_in_block{default_block_size / subwarpSize};
    const auto total_subwarp_count =
        thread::get_subwarp_num_flat<subwarpSize, IndexType>();
    const auto begin_row =
        thread::get_subwarp_id_flat<subwarpSize, IndexType>();

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<subwarpSize>(thread_block);
    const auto sw_threadidx = subwarp_grp.thread_rank();
    const unsigned sw_id_in_threadblock{threadIdx.x / subwarpSize};

    for (auto brow = begin_row; brow < num_b_rows;
         brow += total_subwarp_count) {
        // we treat the diagonal separately
        __shared__ UninitializedArray<ValueType, warps_in_block * matBlkSz_2>
            diag_val;
        for (int iz = sw_threadidx; iz < matBlkSz_2; iz += subwarpSize) {
            diag_val[sw_id_in_threadblock * matBlkSz_2 + iz] =
                zero<ValueType>();
        }
        bool encountered_diag{false};
        int first_diag_iz = -1;
        const IndexType blockRowSz =
            (row_ptrs[brow + 1] - row_ptrs[brow]) * matBlkSz_2;
        IndexType iter_start_liz = 0, iter_start_uiz = matBlkSz_2;
        for (IndexType iz = sw_threadidx; iz < blockRowSz; iz += subwarpSize) {
            const IndexType idx = row_ptrs[brow] + iz / matBlkSz_2;
            const IndexType global_iz = row_ptrs[brow] * matBlkSz_2 + iz;

            const auto bcol = col_idxs[idx];
            const auto val = values[global_iz];

            const bool diag = (bcol == brow);
            const auto diagballot = subwarp_grp.ballot(diag);
            if (diagballot) {
                const auto first_diag_tid = __ffs(diagballot) - 1;

                /* If this is the first time this warp has encountered an entry
                 * in the diagonal block, set the starting position
                 * of the diagonal block relative to the start of the block-row
                 */
                if (!encountered_diag) {
                    first_diag_iz =
                        (iz / subwarpSize) * subwarpSize + first_diag_tid;
                }

                encountered_diag = true;
            }

            // const int num_lower_write{__popc(lowerballot)};
            // const int num_upper_write{__popc(upperballot)};
            const detail::BRowPos<IndexType> lowpos =
                detail::getBRowPos<subwarpSize, IndexType>(
                    subwarp_grp, bcol < brow, iter_start_liz);

            const detail::BRowPos<IndexType> uppos =
                detail::getBRowPos<subwarpSize, IndexType>(
                    subwarp_grp, bcol > brow, iter_start_uiz);

            // save diagonal block for later
            if (diag) {
                ValueType *const diag_val_arr = diag_val;
                diag_val_arr[sw_id_in_threadblock * matBlkSz_2 + iz -
                             first_diag_iz] = val;
            } else if (bcol < brow) {
                const IndexType l_idx =
                    l_row_ptrs[brow] + lowpos.cur_iz / matBlkSz_2;
                const IndexType global_l_iz =
                    l_row_ptrs[brow] * matBlkSz_2 + lowpos.cur_iz;
                if (global_l_iz % matBlkSz_2 == 0) {
                    l_col_idxs[l_idx] = bcol;
                }
                l_values[global_l_iz] = values[global_iz];
            } else /*(brow < bcol)*/ {
                const IndexType u_idx =
                    u_row_ptrs[brow] + uppos.cur_iz / matBlkSz_2;
                const IndexType global_u_iz =
                    u_row_ptrs[brow] * matBlkSz_2 + uppos.cur_iz;
                if (global_u_iz % matBlkSz_2 == 0) {
                    u_col_idxs[u_idx] = bcol;
                }
                u_values[global_u_iz] = values[global_iz];
            }

            iter_start_liz =
                (lowpos.end_iz > 0) ? lowpos.end_iz : iter_start_liz;
            iter_start_uiz = (uppos.end_iz > 0) ? uppos.end_iz : iter_start_uiz;
        }

        // store diagonal entries

        const auto l_diag_idx = l_row_ptrs[brow + 1] - 1;
        const auto u_diag_idx = u_row_ptrs[brow];
        if (sw_threadidx == 0) {
            l_col_idxs[l_diag_idx] = brow;
            u_col_idxs[u_diag_idx] = brow;
        }

        subwarp_grp.sync();

        for (int iz = sw_threadidx; iz < matBlkSz_2; iz += subwarpSize) {
            const auto l_giz = l_diag_idx * matBlkSz_2 + iz;
            const auto u_giz = u_diag_idx * matBlkSz_2 + iz;
            const int bcrd0 = iz / matBlkSz;
            const int bcrd1 = iz % matBlkSz;
            l_values[l_giz] =
                static_cast<ValueType>(static_cast<float>(bcrd0 == bcrd1));
            u_values[u_giz] = diag_val[sw_id_in_threadblock * matBlkSz_2 + iz];
        }
    }
}


}  // namespace kernel
