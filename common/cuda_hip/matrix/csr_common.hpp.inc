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
            return;
        }
    }
}


}  // namespace kernel
