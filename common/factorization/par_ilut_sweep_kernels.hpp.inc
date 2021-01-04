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


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void sweep(
    const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_col_idxs,
    const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ l_row_ptrs,
    const IndexType *__restrict__ l_row_idxs,
    const IndexType *__restrict__ l_col_idxs, ValueType *__restrict__ l_vals,
    IndexType l_nnz, const IndexType *__restrict__ u_row_idxs,
    const IndexType *__restrict__ u_col_idxs, ValueType *__restrict__ u_vals,
    const IndexType *__restrict__ ut_col_ptrs,
    const IndexType *__restrict__ ut_row_idxs, ValueType *__restrict__ ut_vals,
    IndexType u_nnz)
{
    auto tidx = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    if (tidx >= l_nnz + u_nnz) {
        return;
    }
    // split the subwarps into two halves for lower and upper triangle
    auto l_nz = tidx;
    auto u_nz = l_nz - l_nnz;
    auto lower = u_nz < 0;
    auto row = lower ? l_row_idxs[l_nz] : u_row_idxs[u_nz];
    auto col = lower ? l_col_idxs[l_nz] : u_col_idxs[u_nz];
    if (lower && row == col) {
        // don't update the diagonal twice
        return;
    }
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    // find entry of A at (row, col)
    auto a_row_begin = a_row_ptrs[row];
    auto a_row_end = a_row_ptrs[row + 1];
    auto a_row_size = a_row_end - a_row_begin;
    auto a_idx =
        group_wide_search(a_row_begin, a_row_size, subwarp,
                          [&](IndexType i) { return a_col_idxs[i] >= col; });
    bool has_a = a_idx < a_row_end && a_col_idxs[a_idx] == col;
    auto a_val = has_a ? a_vals[a_idx] : zero<ValueType>();
    auto l_row_begin = l_row_ptrs[row];
    auto l_row_size = l_row_ptrs[row + 1] - l_row_begin;
    auto ut_col_begin = ut_col_ptrs[col];
    auto ut_col_size = ut_col_ptrs[col + 1] - ut_col_begin;
    ValueType sum{};
    IndexType ut_nz{};
    auto last_entry = min(row, col);
    group_merge<subwarp_size>(
        l_col_idxs + l_row_begin, l_row_size, ut_row_idxs + ut_col_begin,
        ut_col_size, subwarp,
        [&](IndexType l_idx, IndexType l_col, IndexType ut_idx,
            IndexType ut_row, IndexType, bool) {
            // we don't need to use the `bool valid` because last_entry is
            // already a smaller sentinel value than the one used in group_merge
            if (l_col == ut_row && l_col < last_entry) {
                sum += l_vals[l_idx + l_row_begin] *
                       ut_vals[ut_idx + ut_col_begin];
            }
            // remember the transposed element
            auto found_transp = subwarp.ballot(ut_row == row);
            if (found_transp) {
                ut_nz =
                    subwarp.shfl(ut_idx + ut_col_begin, ffs(found_transp) - 1);
            }
            return true;
        });
    // accumulate result from all threads
    sum = reduce(subwarp, sum, [](ValueType a, ValueType b) { return a + b; });

    if (subwarp.thread_rank() == 0) {
        if (lower) {
            auto to_write = (a_val - sum) / ut_vals[ut_col_ptrs[col + 1] - 1];
            if (is_finite(to_write)) {
                l_vals[l_nz] = to_write;
            }
        } else {
            auto to_write = a_val - sum;
            if (is_finite(to_write)) {
                u_vals[u_nz] = to_write;
                ut_vals[ut_nz] = to_write;
            }
        }
    }
}


}  // namespace kernel