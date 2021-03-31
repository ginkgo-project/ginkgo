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


/*
 * One sweep of iterative parallel block ILU factorization using a simple
 * strategy - one warp per row for each block-row of the original matrix.
 * This has the disadvantage of less parallelism but the advantage is that
 * it should have faster convergence for many types of orderings.
 *
 * The original matrix is in BCSR format, L is in BCSR format, but
 * for the upper factor, we need U^T in BCSC format. Note that individual
 * blocks of U are expected to be transposed too, not just the block sparsity
 * pattern. All blocks are assumed stored column-major.
 */
template <int mat_blk_sz, int subwarp_size, typename ValueType,
          typename IndexType>
__global__
    __launch_bounds__(default_block_size) void compute_bilu_factors_fbcsr_1(
        const IndexType num_b_rows, const IndexType *__restrict__ row_ptrs,
        const IndexType *__restrict__ col_idxs,
        const ValueType *__restrict__ values,
        const IndexType *__restrict__ l_row_ptrs,
        const IndexType *__restrict__ l_col_idxs,
        ValueType *__restrict__ l_values,
        const IndexType *__restrict__ u_col_ptrs,
        const IndexType *__restrict__ u_row_idxs,
        ValueType *__restrict__ u_values)
{
    constexpr int warps_in_block{default_block_size / subwarp_size};
    const auto total_subwarp_count =
        thread::get_subwarp_num_flat<subwarp_size, IndexType>();
    const auto begin_row =
        thread::get_subwarp_id_flat<subwarp_size, IndexType>();

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<subwarp_size>(thread_block);
    const auto sw_threadidx = subwarp_grp.thread_rank();
    const unsigned sw_id_in_threadblock{threadIdx.x / subwarp_size};

    constexpr int mat_blk_sz_2{mat_blk_sz * mat_blk_sz};
    constexpr int entries_per_thread{(mat_blk_sz_2 - 1) / subwarp_size + 1};
    using Blk_t = blockutils::FixedBlock<ValueType, mat_blk_sz, mat_blk_sz>;
    const auto Lvals = reinterpret_cast<const Blk_t *>(l_values);
    const auto Utvals = reinterpret_cast<const Blk_t *>(u_values);

    for (auto brow = begin_row; brow < num_b_rows;
         brow += total_subwarp_count) {
        for (auto ibz = row_ptrs[brow]; ibz < row_ptrs[brow + 1]; ibz++) {
            const auto bcol = col_idxs[ibz];
            const bool lower = brow > bcol;
            // Output location of L is known, but that for U will be determined
            //  while computing the sparse block dot product below.
            const IndexType l_out_ibz = l_row_ptrs[brow] + ibz - row_ptrs[brow];

            ValueType sum[entries_per_thread];
            for (int it = 0; it < entries_per_thread; it++) {
                const IndexType apos = it * subwarp_size + sw_threadidx;
                if (apos < mat_blk_sz_2) {
                    sum[it] = values[ibz * mat_blk_sz_2 + apos];
                }
            }

            IndexType u_out_ibz{-1};
            auto l_idx = l_row_ptrs[brow];
            auto u_idx = u_col_ptrs[bcol];
            while (l_idx < l_row_ptrs[brow + 1] &&
                   u_idx < u_col_ptrs[bcol + 1]) {
                if (u_row_idxs[u_idx] == brow) {
                    u_out_ibz = u_idx;
                    break;
                }
                if (l_col_idxs[l_idx] >= bcol && lower) {
                    break;
                }
                if (l_col_idxs[l_idx] == u_row_idxs[u_idx]) {
                    // GEMM
                    for (int it = 0; it < entries_per_thread; it++) {
                        const int flat_pos = it * subwarp_size + sw_threadidx;
                        if (flat_pos < mat_blk_sz_2) {
                            const int j = flat_pos / mat_blk_sz;
                            const int i = flat_pos % mat_blk_sz;
                            for (int k = 0; k < mat_blk_sz; k++) {
                                sum[it] -=
                                    Lvals[l_idx](i, k) * Utvals[u_idx](j, k);
                            }
                        }
                    }
                    // then move on
                    l_idx++;
                    u_idx++;
                } else if (l_col_idxs[l_idx] < u_row_idxs[u_idx]) {
                    l_idx++;
                } else {
                    u_idx++;
                }
            }

            // Inverse matrix of U^T diag block in shared memory
            // See if it's faster to do this in registers with shuffles
            __shared__ Blk_t uinv_t[warps_in_block];
            if (lower) {
                // Load a row of the U diag block
                ValueType urow[mat_blk_sz];
                if (sw_threadidx < mat_blk_sz) {
                    const IndexType u_diag_ibz = u_col_ptrs[bcol + 1] - 1;
                    for (int i = 0; i < mat_blk_sz; i++) {
                        urow[i] = Utvals[u_diag_ibz](sw_threadidx, i);
                    }
                }
                subwarp_grp.sync();
                uint32 perm = subwarp_grp.thread_rank();
                uint32 trans_perm = subwarp_grp.thread_rank();
                invert_block<mat_blk_sz>(subwarp_grp,
                                         static_cast<uint32>(mat_blk_sz), urow,
                                         perm, trans_perm);
                copy_matrix<mat_blk_sz, and_return>(
                    subwarp_grp, mat_blk_sz, urow, 1, perm, trans_perm,
                    &uinv_t[sw_id_in_threadblock](0, 0), mat_blk_sz);

                for (int it = 0; it < entries_per_thread; it++) {
                    const int flat_pos = it * subwarp_size + sw_threadidx;
                    if (flat_pos < mat_blk_sz_2) {
                        ValueType l_out = 0;
                        const int j = flat_pos / mat_blk_sz;
                        const int i = flat_pos % mat_blk_sz;
                        for (int k = 0; k < mat_blk_sz; k++) {
                            // Fetch sum_block(i,k)
                            const int other_pos = i + k * mat_blk_sz;
                            const int other_it = other_pos / subwarp_size;
                            const int other_rank = other_pos % subwarp_size;
                            const ValueType sum_val =
                                subwarp_grp.shfl(sum[other_it], other_rank);
                            // sum_block(i,k) U^inv(k,j)
                            l_out +=
                                sum_val * uinv_t[sw_id_in_threadblock](j, k);
                        }

                        l_values[l_out_ibz * mat_blk_sz_2 + flat_pos] = l_out;
                    }
                }
            } else {
                const IndexType u_out_start_iz = u_out_ibz * mat_blk_sz_2;
                for (int it = 0; it < entries_per_thread; it++) {
                    const int flat_pos = it * subwarp_size + sw_threadidx;
                    if (flat_pos < mat_blk_sz_2) {
                        // store transpose of sum
                        const int col = flat_pos / mat_blk_sz;
                        const int row = flat_pos % mat_blk_sz;
                        const int out_pos = row * mat_blk_sz + col;
                        u_values[u_out_start_iz + out_pos] = sum[it];
                    }
                }
            }
        }  // end non-zero blocks in current row
    }      // end rows
}


}  // namespace kernel
