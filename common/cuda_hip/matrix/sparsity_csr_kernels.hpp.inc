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


template <size_type subwarp_size, typename MatrixValueType,
          typename input_accessor, typename OutputValueType, typename IndexType,
          typename Closure>
__device__ void device_classical_spmv(const size_type num_rows,
                                      const MatrixValueType* __restrict__ val,
                                      const IndexType* __restrict__ col_idxs,
                                      const IndexType* __restrict__ row_ptrs,
                                      acc::range<input_accessor> b,
                                      OutputValueType* __restrict__ c,
                                      const size_type c_stride, Closure scale)
{
    auto subwarp_tile =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto subrow = thread::get_subwarp_num_flat<subwarp_size>();
    const auto subid = subwarp_tile.thread_rank();
    const IndexType column_id = blockIdx.y;
    const OutputValueType value = val[0];
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    for (; row < num_rows; row += subrow) {
        const auto ind_end = row_ptrs[row + 1];
        OutputValueType temp_val = zero<OutputValueType>();
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subwarp_size) {
            temp_val += value * b(col_idxs[ind], column_id);
        }
        auto subwarp_result =
            reduce(subwarp_tile, temp_val,
                   [](const OutputValueType& a, const OutputValueType& b) {
                       return a + b;
                   });
        if (subid == 0) {
            c[row * c_stride + column_id] =
                scale(subwarp_result, c[row * c_stride + column_id]);
        }
    }
}


template <size_type subwarp_size, typename MatrixValueType,
          typename input_accessor, typename OutputValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_classical_spmv(
    const size_type num_rows, const MatrixValueType* __restrict__ val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, acc::range<input_accessor> b,
    OutputValueType* __restrict__ c, const size_type c_stride)
{
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, c, c_stride,
        [](const OutputValueType& x, const OutputValueType& y) { return x; });
}


template <size_type subwarp_size, typename MatrixValueType,
          typename input_accessor, typename OutputValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_classical_spmv(
    const size_type num_rows, const MatrixValueType* __restrict__ alpha,
    const MatrixValueType* __restrict__ val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, acc::range<input_accessor> b,
    const OutputValueType* __restrict__ beta, OutputValueType* __restrict__ c,
    const size_type c_stride)
{
    const OutputValueType alpha_val = alpha[0];
    const auto beta_val = beta[0];
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, c, c_stride,
        [&alpha_val, &beta_val](const OutputValueType& x,
                                const OutputValueType& y) {
            return alpha_val * x + beta_val * y;
        });
}


}  // namespace kernel
