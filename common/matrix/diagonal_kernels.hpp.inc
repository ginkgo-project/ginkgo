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


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void apply_to_dense(
    size_type num_rows, size_type num_cols, const ValueType *__restrict__ diag,
    size_type source_stride, const ValueType *__restrict__ source_values,
    size_type result_stride, ValueType *__restrict__ result_values)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto row = tidx / num_cols;
    const auto col = tidx % num_cols;

    if (row < num_rows) {
        result_values[row * result_stride + col] =
            source_values[row * source_stride + col] * diag[row];
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void right_apply_to_dense(
    size_type num_rows, size_type num_cols, const ValueType *__restrict__ diag,
    size_type source_stride, const ValueType *__restrict__ source_values,
    size_type result_stride, ValueType *__restrict__ result_values)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto row = tidx / num_cols;
    const auto col = tidx % num_cols;

    if (row < num_rows) {
        result_values[row * result_stride + col] =
            source_values[row * source_stride + col] * diag[col];
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void apply_to_csr(
    size_type num_rows, const ValueType *__restrict__ diag,
    const IndexType *__restrict__ row_ptrs,
    ValueType *__restrict__ result_values)
{
    constexpr auto warp_size = config::warp_size;
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const auto row = thread::get_subwarp_id_flat<warp_size>();
    const auto tid_in_warp = warp_tile.thread_rank();

    if (row >= num_rows) {
        return;
    }

    const auto diag_val = diag[row];

    for (size_type idx = row_ptrs[row] + tid_in_warp; idx < row_ptrs[row + 1];
         idx += warp_size) {
        result_values[idx] *= diag_val;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void right_apply_to_csr(
    size_type num_nnz, const ValueType *__restrict__ diag,
    const IndexType *__restrict__ col_idxs,
    ValueType *__restrict__ result_values)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx >= num_nnz) {
        return;
    }

    result_values[tidx] *= diag[col_idxs[tidx]];
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void convert_to_csr(
    size_type size, const ValueType *__restrict__ diag_values,
    IndexType *__restrict__ row_ptrs, IndexType *__restrict__ col_idxs,
    ValueType *__restrict__ csr_values)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx >= size) {
        return;
    }
    if (tidx == 0) {
        row_ptrs[size] = size;
    }

    row_ptrs[tidx] = tidx;
    col_idxs[tidx] = tidx;
    csr_values[tidx] = diag_values[tidx];
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void conj_transpose(
    size_type size, const ValueType *__restrict__ orig_values,
    ValueType *__restrict__ trans_values)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx >= size) {
        return;
    }

    trans_values[tidx] = conj(orig_values[tidx]);
}


}  // namespace kernel
