/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

template <size_type subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void subwarp_apply(
    const int iteration, const size_type num_rows,
    const ValueType* __restrict__ val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ b, const size_type b_stride,
    ValueType* __restrict__ c, const size_type c_stride)
{
    auto subwarp_tile =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto subid = subwarp_tile.thread_rank();
    const auto column_id = blockIdx.y;
    auto row = thread::get_subwarp_id_flat<subwarp_size>();
    const auto alpha_val = alpha[0];
    if (row < num_rows) {
        const auto rhs = b[row * b_stride + column_id];
        for (int i = 0; i < iteration; i++) {
            const auto ind_end = row_ptrs[row + 1];
            ValueType temp_val = zero<ValueType>();
            for (auto ind = row_ptrs[row] + subid; ind < ind_end;
                 ind += subwarp_size) {
                temp_val += val[ind] * c[col_idxs[ind] * c_stride + column_id];
            }
            auto subwarp_result = reduce(
                subwarp_tile, temp_val,
                [](const ValueType& a, const ValueType& b) { return a + b; });
            if (subid == 0) {
                c[row * c_stride + column_id] +=
                    alpha_val * (rhs - subwarp_result);
            }
            // __threadfence();
        }
    }
}
