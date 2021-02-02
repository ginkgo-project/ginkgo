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

template <int mat_blk_sz, int subwarp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void transpose_blocks(
    const IndexType nbnz, ValueType *const values)
{
    constexpr int warps_in_block{default_block_size / subwarp_size};
    const auto total_subwarp_count =
        thread::get_subwarp_num_flat<subwarp_size, IndexType>();
    const IndexType begin_blk =
        thread::get_subwarp_id_flat<subwarp_size, IndexType>();

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<subwarp_size>(thread_block);
    const auto sw_threadidx = subwarp_grp.thread_rank();
    const unsigned sw_id_in_threadblock{threadIdx.x / subwarp_size};

    constexpr int mat_blk_sz_2{mat_blk_sz * mat_blk_sz};
    // constexpr int entries_per_thread {mat_blk_sz_2/subwarp_size + 1};

    for (auto ibz = begin_blk; ibz < nbnz; ibz += total_subwarp_count) {
        //__shared__ ValueType origblocks[warps_in_block][mat_blk_sz_2];
        __shared__ UninitializedArray<ValueType, warps_in_block * mat_blk_sz_2>
            origblocks;
        for (int i = sw_threadidx; i < mat_blk_sz_2; i += subwarp_size) {
            // origblocks[sw_id_in_threadblock][i] = values[ibz*mat_blk_sz_2 +
            // i];
            origblocks[sw_id_in_threadblock * mat_blk_sz_2 + i] =
                values[ibz * mat_blk_sz_2 + i];
        }
        subwarp_grp.sync();

        for (int i = sw_threadidx; i < mat_blk_sz_2; i += subwarp_size) {
            const int row = sw_threadidx % mat_blk_sz;
            const int col = sw_threadidx / mat_blk_sz;
            const int in_pos = row * mat_blk_sz + col;
            // values[i] = origblocks[sw_id_in_threadblock][in_pos];
            values[ibz * mat_blk_sz_2 + i] =
                origblocks[sw_id_in_threadblock * mat_blk_sz_2 + in_pos];
        }
    }
}

}  // namespace kernel
