// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace kernel {


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void check_unsorted(
    const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs, IndexType num_rows, bool* flag)
{
    __shared__ bool sh_flag;
    auto block = group::this_thread_block();
    if (block.thread_rank() == 0) {
        sh_flag = *flag;
    }
    block.sync();

    auto row = thread::get_thread_id_flat<IndexType>();
    if (row >= num_rows) {
        return;
    }

    // fail early
    if (sh_flag) {
        for (auto nz = row_ptrs[row]; nz < row_ptrs[row + 1] - 1; ++nz) {
            if (col_idxs[nz] > col_idxs[nz + 1]) {
                *flag = false;
                sh_flag = false;
                return;
            }
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void conjugate(
    size_type num_nonzeros, ValueType* __restrict__ val)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx < num_nonzeros) {
        val[tidx] = conj(val[tidx]);
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void check_diagonal_entries(
    const IndexType num_min_rows_cols,
    const IndexType* const __restrict__ row_ptrs,
    const IndexType* const __restrict__ col_idxs,
    bool* const __restrict__ has_all_diags)
{
    constexpr int warp_size = config::warp_size;
    auto tile_grp =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const auto row = thread::get_subwarp_id_flat<warp_size, IndexType>();
    if (row < num_min_rows_cols) {
        const auto tid_in_warp = tile_grp.thread_rank();
        const IndexType row_start = row_ptrs[row];
        const IndexType num_nz = row_ptrs[row + 1] - row_start;
        bool row_has_diag_local{false};
        for (IndexType iz = tid_in_warp; iz < num_nz; iz += warp_size) {
            if (col_idxs[iz + row_start] == row) {
                row_has_diag_local = true;
                break;
            }
        }
        auto row_has_diag = static_cast<bool>(tile_grp.any(row_has_diag_local));
        if (!row_has_diag) {
            if (tile_grp.thread_rank() == 0) {
                *has_all_diags = false;
            }
        }
    }
}


}  // namespace kernel
