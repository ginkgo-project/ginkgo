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

namespace kernel {


/**
 * @internal
 *
 * This kernel supports at most `subwarp_size` (< `warp_size`) elements per row.
 * If there are more elements, they are simply ignored. Only the first
 * `subwarp_size` elements are considered both for the values and for the
 * sparsity pattern.
 */
template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType, typename Callable>
__forceinline__ __device__ void generic_generate(
    IndexType num_rows, const IndexType *__restrict__ m_row_ptrs,
    const IndexType *__restrict__ m_col_idxs,
    const ValueType *__restrict__ m_values, IndexType *__restrict__ i_row_ptrs,
    IndexType *__restrict__ i_col_idxs, ValueType *__restrict__ i_values,
    Callable trs_solve)
{
    static_assert((subwarp_size & (subwarp_size - 1)) == 0,
                  "subwarp_size must be a power of 2");

    const IndexType gidx = thread::get_thread_id_flat<IndexType>();
    const IndexType row = gidx / subwarp_size;
    const int tidx = threadIdx.x;

    auto this_block = group::this_thread_block();
    auto subwarp = group::tiled_partition<subwarp_size>(this_block);
    const int subwarp_id = subwarp.thread_rank();

    __shared__ UninitializedArray<ValueType, subwarp_size * subwarp_size *
                                                 subwarps_per_block>
        trisystem;

    const int end_trisystem = blockDim.x * subwarp_size;
    const int trisystem_start =
        (tidx / subwarp_size) * subwarp_size * subwarp_size;

    // Allow at most subwarp_size * subwarp_size elements in a subwarp trisystem
    for (int i = 0; i < subwarp_size; ++i) {
        trisystem[trisystem_start + subwarp_id + i * subwarp_size] =
            zero<ValueType>();
    }
    this_block.sync();

    if (row >= num_rows) {
        return;
    }

    const auto i_row_begin = i_row_ptrs[row];
    const auto i_row_end = i_row_ptrs[row + 1] - i_row_begin < subwarp_size
                               ? i_row_ptrs[row + 1]
                               : i_row_begin + subwarp_size;
    const auto num_row_elems = i_row_end - i_row_begin;

    if (subwarp_id < num_row_elems) {
        const auto col = i_col_idxs[i_row_begin + subwarp_id];
        const auto m_row_end = m_row_ptrs[col + 1];
        auto m_row_ptr = m_row_ptrs[col];
        auto i_row_ptr = i_row_begin;
        int idx{trisystem_start +
                subwarp_id};  // write to trisystem in row major

        while (m_row_ptr < m_row_end && i_row_ptr < i_row_end &&
               idx < end_trisystem) {
            const auto sparsity_col = i_col_idxs[i_row_ptr];
            const auto m_col = m_col_idxs[m_row_ptr];
            if (sparsity_col == m_col) {
                trisystem[idx] = m_values[m_row_ptr];
                idx += subwarp_size;
                ++m_row_ptr;
                ++i_row_ptr;
            } else if (m_col < sparsity_col) {
                ++m_row_ptr;
            } else {
                ++i_row_ptr;
                idx += subwarp_size;
            }
        }
    }

    this_block.sync();

    // Now, read a full col of `trisystem` into local registers, which will be
    // row elements after this (implicit) transpose
    ValueType local_row[subwarp_size];
    const auto trisystem_col_start =
        trisystem_start + subwarp_id * subwarp_size;
    for (int i = 0; i < subwarp_size; ++i) {
        const auto intermediate_idx = subwarp_id + i;
        const auto coalesced_idx = intermediate_idx % subwarp_size;
        local_row[coalesced_idx] =
            trisystem[trisystem_col_start + coalesced_idx];
    }

    const auto rhs = trs_solve(num_row_elems, local_row, subwarp);

    // Write back:
    if (subwarp_id < num_row_elems) {
        const auto idx = i_row_begin + subwarp_id;
        if (gko::isfinite(rhs)) {
            i_values[idx] = rhs;
        } else {
            i_values[idx] =
                i_col_idxs[idx] == row ? one<ValueType>() : zero<ValueType>();
        }
    }
}


template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType>
__global__
__launch_bounds__(subwarp_size *subwarps_per_block) void generate_l_inverse(
    IndexType num_rows, const IndexType *__restrict__ m_row_ptrs,
    const IndexType *__restrict__ m_col_idxs,
    const ValueType *__restrict__ m_values, IndexType *__restrict__ i_row_ptrs,
    IndexType *__restrict__ i_col_idxs, ValueType *__restrict__ i_values)
{
    auto trs_solve = [](IndexType num_elems,
                        const ValueType *__restrict__ local_row,
                        group::thread_block_tile<subwarp_size> &subwarp) {
        const int subwarp_id = subwarp.thread_rank();
        ValueType rhs =
            subwarp_id == num_elems - 1 ? one<ValueType>() : zero<ValueType>();
        // Solve Triangular system
        for (int d_col = num_elems - 1; d_col >= 0; --d_col) {
            const auto elem = local_row[d_col];
            if (d_col == subwarp_id) {
                rhs /= elem;
            }

            const ValueType bot = subwarp.shfl(rhs, d_col);
            if (subwarp_id < d_col) {
                rhs -= bot * elem;
            }
        }

        return rhs;
    };
    generic_generate<subwarp_size, subwarps_per_block>(
        num_rows, m_row_ptrs, m_col_idxs, m_values, i_row_ptrs, i_col_idxs,
        i_values, trs_solve);
}


template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType>
__global__
__launch_bounds__(subwarp_size *subwarps_per_block) void generate_u_inverse(
    IndexType num_rows, const IndexType *__restrict__ m_row_ptrs,
    const IndexType *__restrict__ m_col_idxs,
    const ValueType *__restrict__ m_values, IndexType *__restrict__ i_row_ptrs,
    IndexType *__restrict__ i_col_idxs, ValueType *__restrict__ i_values)
{
    auto trs_solve = [](IndexType num_elems,
                        const ValueType *__restrict__ local_row,
                        group::thread_block_tile<subwarp_size> &subwarp) {
        const int subwarp_id = subwarp.thread_rank();
        ValueType rhs = subwarp_id == 0 ? one<ValueType>() : zero<ValueType>();
        // Solve Triangular system
        for (int d_col = 0; d_col < num_elems; ++d_col) {
            const auto elem = local_row[d_col];
            if (d_col == subwarp_id) {
                rhs /= elem;
            }

            const ValueType top = subwarp.shfl(rhs, d_col);
            if (d_col < subwarp_id) {
                rhs -= top * elem;
            }
        }

        return rhs;
    };
    generic_generate<subwarp_size, subwarps_per_block>(
        num_rows, m_row_ptrs, m_col_idxs, m_values, i_row_ptrs, i_col_idxs,
        i_values, trs_solve);
}


}  // namespace kernel
