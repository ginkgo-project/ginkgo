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


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void ic_init(
    const IndexType *__restrict__ l_row_ptrs, ValueType *__restrict__ l_vals,
    size_type num_rows)
{
    auto row = thread::get_thread_id_flat();
    if (row >= num_rows) {
        return;
    }
    auto l_nz = l_row_ptrs[row + 1] - 1;
    auto diag = sqrt(l_vals[l_nz]);
    if (is_finite(diag)) {
        l_vals[l_nz] = diag;
    } else {
        l_vals[l_nz] = one<ValueType>();
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void ic_sweep(
    const IndexType *__restrict__ a_row_idxs,
    const IndexType *__restrict__ a_col_idxs,
    const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ l_row_ptrs,
    const IndexType *__restrict__ l_col_idxs, ValueType *__restrict__ l_vals,
    IndexType l_nnz)
{
    auto l_nz = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    if (l_nz >= l_nnz) {
        return;
    }
    auto row = a_row_idxs[l_nz];
    auto col = l_col_idxs[l_nz];
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto a_val = a_vals[l_nz];
    auto l_row_begin = l_row_ptrs[row];
    auto l_row_size = l_row_ptrs[row + 1] - l_row_begin;
    auto lh_col_begin = l_row_ptrs[col];
    auto lh_col_size = l_row_ptrs[col + 1] - lh_col_begin;
    ValueType sum{};
    IndexType lh_nz{};
    auto last_entry = col;
    group_merge<subwarp_size>(
        l_col_idxs + l_row_begin, l_row_size, l_col_idxs + lh_col_begin,
        lh_col_size, subwarp,
        [&](IndexType l_idx, IndexType l_col, IndexType lh_idx,
            IndexType lh_row, IndexType, bool) {
            // we don't need to use the `bool valid` because last_entry is
            // already a smaller sentinel value than the one used in group_merge
            if (l_col == lh_row && l_col < last_entry) {
                sum += l_vals[l_idx + l_row_begin] *
                       conj(l_vals[lh_idx + lh_col_begin]);
            }
            // remember the transposed element
            auto found_transp = subwarp.ballot(lh_row == row);
            if (found_transp) {
                lh_nz =
                    subwarp.shfl(lh_idx + lh_col_begin, ffs(found_transp) - 1);
            }
            return true;
        });
    // accumulate result from all threads
    sum = reduce(subwarp, sum, [](ValueType a, ValueType b) { return a + b; });

    if (subwarp.thread_rank() == 0) {
        auto to_write = row == col
                            ? sqrt(a_val - sum)
                            : (a_val - sum) / l_vals[l_row_ptrs[col + 1] - 1];
        if (is_finite(to_write)) {
            l_vals[l_nz] = to_write;
        }
    }
}


}  // namespace kernel
