// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace kernel {


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void ict_sweep(
    const IndexType* __restrict__ a_row_ptrs,
    const IndexType* __restrict__ a_col_idxs,
    const ValueType* __restrict__ a_vals,
    const IndexType* __restrict__ l_row_ptrs,
    const IndexType* __restrict__ l_row_idxs,
    const IndexType* __restrict__ l_col_idxs, ValueType* __restrict__ l_vals,
    IndexType l_nnz)
{
    auto l_nz = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    if (l_nz >= l_nnz) {
        return;
    }
    auto row = l_row_idxs[l_nz];
    auto col = l_col_idxs[l_nz];
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
                sum += load_relaxed(l_vals + (l_idx + l_row_begin)) *
                       conj(load_relaxed(l_vals + (lh_idx + lh_col_begin)));
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
        auto to_write =
            row == col ? sqrt(a_val - sum)
                       : (a_val - sum) /
                             load_relaxed(l_vals + (l_row_ptrs[col + 1] - 1));
        if (is_finite(to_write)) {
            store_relaxed(l_vals + l_nz, to_write);
        }
    }
}


}  // namespace kernel
