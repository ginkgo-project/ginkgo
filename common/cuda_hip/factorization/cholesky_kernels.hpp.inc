/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void build_postorder_cols(
    IndexType num_rows, const IndexType* cols, const IndexType* row_ptrs,
    const IndexType* inv_postorder, IndexType* postorder_cols,
    IndexType* lower_ends)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row >= num_rows) {
        return;
    }
    const auto row_begin = row_ptrs[row];
    const auto row_end = row_ptrs[row + 1];
    auto lower_end = row_begin;
    for (auto nz = row_begin; nz < row_end; nz++) {
        const auto col = cols[nz];
        if (col < row) {
            postorder_cols[lower_end] = inv_postorder[cols[nz]];
            lower_end++;
        }
    }
    // fill the rest with sentinels
    for (auto nz = lower_end; nz < row_end; nz++) {
        postorder_cols[nz] = num_rows - 1;
    }
    lower_ends[row] = lower_end;
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void pointer_doubling_kernel(
    IndexType num_rows, const IndexType* parents, IndexType* doubled_parents)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row >= num_rows) {
        return;
    }
    const auto parent = parents[row];
    doubled_parents[row] = parent < num_rows ? parents[parent] : num_rows;
}


template <int subwarp_size, typename IndexType>
__global__
__launch_bounds__(default_block_size) void cholesky_symbolic_count_kernel(
    IndexType num_rows, const IndexType* row_ptrs, const IndexType* lower_ends,
    const IndexType* inv_postorder, const IndexType* postorder_cols,
    const IndexType* postorder_parent, IndexType* row_nnz)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    if (row >= num_rows) {
        return;
    }
    const auto row_begin = row_ptrs[row];
    // instead of relying on the input containing a diagonal, we artificially
    // introduce the diagonal entry (in postorder indexing) as a sentinel after
    // the last lower triangular entry.
    const auto diag_postorder = inv_postorder[row];
    const auto lower_end = lower_ends[row];
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto lane = subwarp.thread_rank();
    IndexType count{};
    for (auto nz = row_begin + lane; nz < lower_end; nz += subwarp_size) {
        auto node = postorder_cols[nz];
        const auto next_node =
            nz < lower_end - 1 ? postorder_cols[nz + 1] : diag_postorder;
        while (node < next_node) {
            count++;
            node = postorder_parent[node];
        }
    }
    count =
        reduce(subwarp, count, [](IndexType a, IndexType b) { return a + b; }) +
        1;  // lower entries plus diagonal
    if (lane == 0) {
        row_nnz[row] = count;
    }
}


template <int subwarp_size, typename IndexType>
__global__
__launch_bounds__(default_block_size) void cholesky_symbolic_factorize_kernel(
    IndexType num_rows, const IndexType* row_ptrs, const IndexType* lower_ends,
    const IndexType* inv_postorder, const IndexType* postorder_cols,
    const IndexType* postorder, const IndexType* postorder_parent,
    const IndexType* out_row_ptrs, IndexType* out_cols)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    if (row >= num_rows) {
        return;
    }
    const auto row_begin = row_ptrs[row];
    // instead of relying on the input containing a diagonal, we artificially
    // introduce the diagonal entry (in postorder indexing) as a sentinel after
    // the last lower triangular entry.
    const auto diag_postorder = inv_postorder[row];
    const auto lower_end = lower_ends[row];
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto lane = subwarp.thread_rank();
    const auto prefix_mask = (config::lane_mask_type(1) << lane) - 1;
    auto out_base = out_row_ptrs[row];
    for (auto base = row_begin; base < lower_end; base += subwarp_size) {
        auto nz = base + lane;
        auto node = nz < lower_end ? postorder_cols[nz] : diag_postorder;
        const auto next_node =
            nz < lower_end - 1 ? postorder_cols[nz + 1] : diag_postorder;
        bool pred = node < next_node;
        auto mask = subwarp.ballot(pred);
        while (mask) {
            if (pred) {
                const auto out_nz = out_base + popcnt(mask & prefix_mask);
                out_cols[out_nz] = postorder[node];
                node = postorder_parent[node];
                pred = node < next_node;
            }
            out_base += popcnt(mask);
            mask = subwarp.ballot(pred);
        }
    }
    // add diagonal entry
    if (lane == 0) {
        out_cols[out_base] = row;
    }
}
