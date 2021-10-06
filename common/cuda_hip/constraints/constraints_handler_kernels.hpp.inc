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


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_subset(
    size_type n, const IndexType* __restrict__ map,
    ValueType* __restrict__ array, ValueType val)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < n) {
        array[map[tidx]] = val;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void copy_subset(
    size_type n, const IndexType* __restrict__ map,
    const ValueType* __restrict__ src, ValueType* __restrict__ dst)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < n) {
        const auto mapped_idx = map[tidx];
        dst[mapped_idx] = src[mapped_idx];
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void set_unit_rows(
    size_type n, const IndexType* __restrict__ map,
    const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs, ValueType* __restrict__ values)
{
    constexpr auto warp_size = config::warp_size;
    const auto subset_idx = thread::get_subwarp_id_flat<warp_size>();
    const auto local_tidx = threadIdx.x % warp_size;

    if (subset_idx < n) {
        const auto row = map[subset_idx];
        for (size_type i = local_tidx; i < row_ptrs[row + 1] - row_ptrs[row];
             i += warp_size) {
            const auto orig_idx = i + row_ptrs[row];
            values[orig_idx] = col_idxs[orig_idx] == row;
        }
    }
}


}  // namespace kernel
