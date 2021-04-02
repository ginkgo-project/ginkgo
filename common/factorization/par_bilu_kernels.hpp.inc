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


/// Column-major square block
template <typename ValueType, int bs>
class FixedBlock {
public:
    __device__ const ValueType &operator()(const int i, const int j) const
    {
        return reinterpret_cast<const ValueType *>(vals_)[i + j * bs];
    }

    __device__ ValueType &operator()(const int i, const int j)
    {
        return reinterpret_cast<ValueType *>(vals_)[i + j * bs];
    }

private:
    unsigned char vals_[sizeof(ValueType) / sizeof(unsigned char) * bs * bs];
};


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
 *
 * Stores the intermediate sum in registers - this only works for block sizes
 * up to 5. Use the other shared memory variant for larger blocks.
 */
template <int mat_blk_sz, int subwarp_size, bool jacobi, typename ValueType,
          typename IndexType>
__global__
    __launch_bounds__(default_block_size) void compute_bilu_factors_fbcsr_1(
        const IndexType num_b_rows,
        const IndexType *const __restrict__ row_ptrs,
        const IndexType *const __restrict__ col_idxs,
        const ValueType *const __restrict__ values,
        const IndexType *const __restrict__ l_row_ptrs,
        const IndexType *const __restrict__ l_col_idxs,
        ValueType *const __restrict__ l_values,
        const IndexType *const __restrict__ u_col_ptrs,
        const IndexType *const __restrict__ u_row_idxs,
        ValueType *const __restrict__ u_values,
        ValueType *const __restrict__ n_l_values = nullptr,
        ValueType *const __restrict__ n_u_values = nullptr)
{
    static_assert(mat_blk_sz <= subwarp_size, "Block is too large!");
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
    if (mat_blk_sz_2 > subwarp_size) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("Block is too large for this variant!\n");
        }
        return;
    }
    constexpr int entries_per_thread{(mat_blk_sz_2 - 1) / subwarp_size + 1};
    using Blk_t = FixedBlock<ValueType, mat_blk_sz>;
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

            if (lower) {
                // Load a row of the U^T diag block
                ValueType utrow[mat_blk_sz];
                if (sw_threadidx < mat_blk_sz) {
                    const IndexType u_diag_ibz = u_col_ptrs[bcol + 1] - 1;
                    for (int j = 0; j < mat_blk_sz; j++) {
                        utrow[j] = Utvals[u_diag_ibz](sw_threadidx, j);
                    }
                }
                uint32 perm = subwarp_grp.thread_rank();
                uint32 trans_perm = subwarp_grp.thread_rank();
                invert_block<mat_blk_sz>(subwarp_grp,
                                         static_cast<uint32>(mat_blk_sz), utrow,
                                         perm, trans_perm);
                // Inverse matrix of U^T diag block in shared memory
                // See if it's faster to do this in registers with shuffles,
                //  but that will need a different copy_matrix (after
                //  inversion).
                __shared__ Blk_t utinv[warps_in_block];
                copy_matrix<mat_blk_sz, and_return>(
                    subwarp_grp, mat_blk_sz, utrow, 1, perm, trans_perm,
                    &utinv[sw_id_in_threadblock](0, 0), mat_blk_sz);

                subwarp_grp.sync();

                for (int it = 0; it < entries_per_thread; it++) {
                    const int flat_pos = it * subwarp_size + sw_threadidx;
                    ValueType l_out = 0;
                    const int j = flat_pos / mat_blk_sz;
                    const int i = flat_pos % mat_blk_sz;
                    for (int k = 0; k < mat_blk_sz; k++) {
                        // To fetch sum(i,k)
                        const int other_pos = i + k * mat_blk_sz;
                        const int other_it = other_pos / subwarp_size;
                        const int other_rank = other_pos % subwarp_size;

                        const ValueType sum_val =
                            subwarp_grp.shfl(sum[other_it], other_rank);
                        if (flat_pos < mat_blk_sz_2) {
                            // sum_block(i,k) U^inv(k,j)
                            l_out +=
                                sum_val * utinv[sw_id_in_threadblock](j, k);
                        }
                    }

                    if (flat_pos < mat_blk_sz_2) {
                        if (jacobi) {
                            n_l_values[l_out_ibz * mat_blk_sz_2 + flat_pos] =
                                l_out;
                        } else {
                            l_values[l_out_ibz * mat_blk_sz_2 + flat_pos] =
                                l_out;
                        }
                    }
                }
            } else {
                const IndexType u_out_start_iz = u_out_ibz * mat_blk_sz_2;
                if (mat_blk_sz_2 > subwarp_size) {
                    // too large, write un-coalesced
                    for (int it = 0; it < entries_per_thread; it++) {
                        const int flat_pos = it * subwarp_size + sw_threadidx;
                        if (flat_pos < mat_blk_sz_2) {
                            // store sum to transpose of U
                            const int col =
                                flat_pos % mat_blk_sz;  // col idx of U^t
                            const int row =
                                flat_pos / mat_blk_sz;  // row idx of U^t
                            const int st_pos = row + col * mat_blk_sz;
                            if (jacobi) {
                                n_u_values[u_out_start_iz + st_pos] = sum[it];
                            } else {
                                u_values[u_out_start_iz + st_pos] = sum[it];
                            }
                        }
                    }
                } else {
                    // write coalesced
                    for (int it = 0; it < entries_per_thread; it++) {
                        const int flat_pos = it * subwarp_size + sw_threadidx;
                        // store transpose of sum
                        const int col =
                            flat_pos / mat_blk_sz;  // col idx of U^t
                        const int row =
                            flat_pos % mat_blk_sz;  // row idx of U^t
                        const int sum_trans_pos =
                            col + row * mat_blk_sz;  // pos of sum(col,row)
                        const int sum_it = sum_trans_pos / subwarp_size;
                        const int sum_rank = sum_trans_pos % subwarp_size;
                        const ValueType sum_val =
                            subwarp_grp.shfl(sum[sum_it], sum_rank);
                        if (flat_pos < mat_blk_sz_2) {
                            if (jacobi) {
                                n_u_values[u_out_start_iz + flat_pos] = sum_val;
                            } else {
                                u_values[u_out_start_iz + flat_pos] = sum_val;
                            }
                        }
                    }
                }
            }
        }  // end non-zero blocks in current row
    }      // end rows
}

template <int mat_blk_sz, int subwarp_size, bool jacobi, typename ValueType,
          typename IndexType>
__global__
    __launch_bounds__(default_block_size) void compute_bilu_factors_fbcsr_1_shr(
        const IndexType num_b_rows,
        const IndexType *const __restrict__ row_ptrs,
        const IndexType *const __restrict__ col_idxs,
        const ValueType *const __restrict__ values,
        const IndexType *const __restrict__ l_row_ptrs,
        const IndexType *const __restrict__ l_col_idxs,
        ValueType *const __restrict__ l_values,
        const IndexType *const __restrict__ u_col_ptrs,
        const IndexType *const __restrict__ u_row_idxs,
        ValueType *const __restrict__ u_values,
        ValueType *const __restrict__ n_l_values = nullptr,
        ValueType *const __restrict__ n_u_values = nullptr)
{
    static_assert(mat_blk_sz <= subwarp_size, "Block is too large!");
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
    // constexpr int entries_per_thread{(mat_blk_sz_2 - 1) / subwarp_size + 1};
    using Blk_t = FixedBlock<ValueType, mat_blk_sz>;
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

            __shared__
                UninitializedArray<ValueType, mat_blk_sz_2 * warps_in_block>
                    sum;
            for (int ipos = sw_threadidx; ipos < mat_blk_sz_2;
                 ipos += subwarp_size) {
                sum[sw_id_in_threadblock * mat_blk_sz_2 + ipos] =
                    values[ibz * mat_blk_sz_2 + ipos];
            }
            subwarp_grp.sync();

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
                    for (int ipos = sw_threadidx; ipos < mat_blk_sz_2;
                         ipos += subwarp_size) {
                        const int j = ipos / mat_blk_sz;
                        const int i = ipos % mat_blk_sz;
                        for (int k = 0; k < mat_blk_sz; k++) {
                            sum[sw_id_in_threadblock * mat_blk_sz_2 + ipos] -=
                                Lvals[l_idx](i, k) * Utvals[u_idx](j, k);
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

            if (lower) {
                // Load a row of the U^T diag block
                ValueType utrow[mat_blk_sz];
                if (sw_threadidx < mat_blk_sz) {
                    const IndexType u_diag_ibz = u_col_ptrs[bcol + 1] - 1;
                    for (int j = 0; j < mat_blk_sz; j++) {
                        utrow[j] = Utvals[u_diag_ibz](sw_threadidx, j);
                    }
                }
                uint32 perm = subwarp_grp.thread_rank();
                uint32 trans_perm = subwarp_grp.thread_rank();
                invert_block<mat_blk_sz>(subwarp_grp,
                                         static_cast<uint32>(mat_blk_sz), utrow,
                                         perm, trans_perm);
                // Inverse matrix of U^T diag block in shared memory
                // See if it's faster to do this in registers with shuffles,
                //  but that will need a different copy_matrix (after
                //  inversion).
                __shared__ Blk_t utinv[warps_in_block];
                copy_matrix<mat_blk_sz, and_return>(
                    subwarp_grp, mat_blk_sz, utrow, 1, perm, trans_perm,
                    &utinv[sw_id_in_threadblock](0, 0), mat_blk_sz);

                subwarp_grp.sync();

                for (int flat_pos = sw_threadidx; flat_pos < mat_blk_sz_2;
                     flat_pos += subwarp_size) {
                    ValueType l_out = 0;
                    const int j = flat_pos / mat_blk_sz;
                    const int i = flat_pos % mat_blk_sz;
                    for (int k = 0; k < mat_blk_sz; k++) {
                        // Fetch sum(i,k)
                        const int other_pos = i + k * mat_blk_sz;
                        const ValueType sum_val =
                            sum[sw_id_in_threadblock * mat_blk_sz_2 +
                                other_pos];
                        // sum_block(i,k) U^inv(k,j)
                        l_out += sum_val * utinv[sw_id_in_threadblock](j, k);
                    }

                    if (jacobi) {
                        n_l_values[l_out_ibz * mat_blk_sz_2 + flat_pos] = l_out;
                    } else {
                        l_values[l_out_ibz * mat_blk_sz_2 + flat_pos] = l_out;
                    }
                }
            } else {
                const IndexType u_out_start_iz = u_out_ibz * mat_blk_sz_2;

                for (int ipos = sw_threadidx; ipos < mat_blk_sz_2;
                     ipos += subwarp_size) {
                    const int i = ipos % mat_blk_sz;
                    const int j = ipos / mat_blk_sz;
                    const int sum_idx = j + i * mat_blk_sz;
                    if (jacobi) {
                        n_u_values[u_out_start_iz + ipos] =
                            sum[sw_id_in_threadblock * mat_blk_sz_2 + sum_idx];
                    } else {
                        u_values[u_out_start_iz + ipos] =
                            sum[sw_id_in_threadblock * mat_blk_sz_2 + sum_idx];
                    }
                }
            }
        }  // end non-zero blocks in current row
    }      // end rows
}

}  // namespace kernel
