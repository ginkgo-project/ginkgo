/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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

#ifndef GKO_CUDA_COMPONENTS_FORMAT_CONVERSION_CUH_
#define GKO_CUDA_COMPONENTS_FORMAT_CONVERSION_CUH_


#include <ginkgo/core/base/std_extensions.hpp>


#include "cuda/base/cublas_bindings.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {


constexpr auto default_block_size = 512;


/*
 * Calculates the prefix sum of `elements` inside `default_block_size`
 * blocks in-place.
 * `default_block_size` must be a power of 2
 */
template <int block_size = default_block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void start_prefix_sum(
    size_type num_elements, ValueType *__restrict__ elements,
    ValueType *__restrict__ block_sum)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ size_type prefix_helper[block_size];
    prefix_helper[threadIdx.x] =
        (tidx < num_elements) ? elements[tidx] : zero<ValueType>();
    __syncthreads();

    // Do a normal reduction
    for (int i = 1; i < block_size; i <<= 1) {
        const auto ai = i * (2 * threadIdx.x + 1) - 1;
        const auto bi = i * (2 * threadIdx.x + 2) - 1;
        if (bi < block_size) {
            prefix_helper[bi] += prefix_helper[ai];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Store the total sum
        block_sum[blockIdx.x] = prefix_helper[block_size - 1];
        prefix_helper[block_size - 1] = zero<ValueType>();
    }

    __syncthreads();

    // Perform the down-sweep phase to get the true prefix sum
    for (int i = block_size >> 1; i > 0; i >>= 1) {
        const auto ai = i * (2 * threadIdx.x + 1) - 1;
        const auto bi = i * (2 * threadIdx.x + 2) - 1;
        if (bi < block_size) {
            auto tmp = prefix_helper[ai];
            prefix_helper[ai] = prefix_helper[bi];
            prefix_helper[bi] += tmp;
        }
        __syncthreads();
    }
    if (tidx < num_elements) {
        elements[tidx] = prefix_helper[threadIdx.x];
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void finalize_prefix_sum(
    size_type num_elements, ValueType *__restrict__ elements,
    const ValueType *__restrict__ block_sum)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx < num_elements) {
        ValueType prefix_block_sum = zero<ValueType>();
        for (size_type i = 0; i < blockIdx.x; i++) {
            prefix_block_sum += block_sum[i];
        }
        elements[tidx] += prefix_block_sum;
    }
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_REDUCTION_CUH_
