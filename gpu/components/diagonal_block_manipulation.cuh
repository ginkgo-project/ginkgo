/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_GPU_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_CUH_
#define GKO_GPU_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_CUH_


#include "gpu/components/shuffle.cuh"
#include "gpu/components/synchronization.cuh"


namespace gko {
namespace kernels {
namespace gpu {
namespace device {
namespace csr {


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__device__ __forceinline__ void extract_transposed_diag_blocks(
    const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values,
    const IndexType *__restrict__ block_ptrs, size_type num_blocks,
    ValueType *__restrict__ block_row, int increment,
    ValueType *__restrict__ workspace)
{
    const int blocks_per_warp = warp_size / subwarp_size;
    const int tid = threadIdx.y * subwarp_size + threadIdx.x;
    IndexType bid =
        (blockIdx.x * warps_per_block + threadIdx.z) * blocks_per_warp;
    auto bstart = block_ptrs[bid];
    IndexType bsize = 0;
#pragma unroll
    for (int b = 0; b < blocks_per_warp; ++b, ++bid) {
        if (bid >= num_blocks) {
            break;
        }
        bstart += bsize;
        bsize = block_ptrs[bid + 1] - bstart;
#pragma unroll
        for (int i = 0; i < max_block_size; ++i) {
            if (i >= bsize) {
                break;
            }
            if (threadIdx.y == b && threadIdx.x < max_block_size) {
                workspace[threadIdx.x] = zero<ValueType>();
            }
            const auto row = bstart + i;
            const auto rstart = row_ptrs[row] + tid;
            const auto rend = row_ptrs[row + 1];
            // use the entire warp to ensure coalesced memory access
            for (auto j = rstart; j < rend; j += warp_size) {
                const auto col = col_idxs[j] - bstart;
                if (col >= bsize) {
                    break;
                }
                if (col >= 0) {
                    workspace[col] = values[j];
                }
            }
            warp::synchronize();
            if (threadIdx.y == b && threadIdx.x < bsize) {
                block_row[i * increment] = workspace[threadIdx.x];
            }
        }
    }
}


template <int mbs, int ws, int wpb, typename ValueType, typename IndexType>
__device__ __forceinline__ void insert_diag_blocks_trans(
    int rperm, int cperm, const ValueType *__restrict__ B, int binc,
    const IndexType *__restrict__ block_ptrs,
    ValueType *__restrict__ block_data, size_type padding, size_type num_blocks)
{
    const int bpw = warp_size / ws;
    const size_type bid =
        blockIdx.x * wpb * bpw + threadIdx.z * bpw + threadIdx.y;
    const IndexType bstart = bid < num_blocks ? block_ptrs[bid] : 0;
    const IndexType bsize = bid < num_blocks ? block_ptrs[bid + 1] - bstart : 0;
#pragma unroll
    for (int i = 0; i < mbs; ++i) {
        if (i >= bsize) {
            break;
        }
        const auto idx = gpu::warp::shuffle(cperm, i, ws);
        const auto rstart = (bstart + idx) * padding;
        if (bid < num_blocks && threadIdx.x < bsize) {
            block_data[rstart + rperm] = B[i * binc];
        }
    }
}


}  // namespace csr
}  // namespace device
}  // namespace gpu
}  // namespace kernels
}  // namespace gko


#endif  // GKO_GPU_COMPONENTS_DIAGONAL_BLOCK_MANIPULATION_CUH_
