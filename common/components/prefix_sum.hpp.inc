/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

/**
 * @internal
 * First step of the calculation of a prefix sum. Calculates the prefix sum
 * in-place on parts of the array `elements`.
 *
 * @param block_size  thread block size for this kernel, also size of blocks on
 * which this kernel calculates the prefix sum in-place
 * @param elements  array on which the prefix sum is to be calculated
 * @param block_sum  array which stores the total sum of each block, requires at
 * least `ceildiv(num_elements, block_size)` elements
 * @param num_elements  total number of entries in `elements`
 *
 * @note To calculate the prefix sum over an array of size bigger than
 * `block_size`, `finalize_prefix_sum` has to be used as well.
 */
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void start_prefix_sum(
    size_type num_elements, ValueType *__restrict__ elements,
    ValueType *__restrict__ block_sum)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const auto element_id = threadIdx.x;
    __shared__ size_type prefix_helper[block_size];
    prefix_helper[element_id] =
        (tidx < num_elements) ? elements[tidx] : zero<ValueType>();
    auto this_block = group::this_thread_block();
    this_block.sync();

    // Do a normal reduction
#pragma unroll
    for (int i = 1; i < block_size; i <<= 1) {
        const auto ai = i * (2 * element_id + 1) - 1;
        const auto bi = i * (2 * element_id + 2) - 1;
        if (bi < block_size) {
            prefix_helper[bi] += prefix_helper[ai];
        }
        this_block.sync();
    }

    if (element_id == 0) {
        // Store the total sum
        block_sum[blockIdx.x] = prefix_helper[block_size - 1];
        prefix_helper[block_size - 1] = zero<ValueType>();
    }

    this_block.sync();

    // Perform the down-sweep phase to get the true prefix sum
#pragma unroll
    for (int i = block_size >> 1; i > 0; i >>= 1) {
        const auto ai = i * (2 * element_id + 1) - 1;
        const auto bi = i * (2 * element_id + 2) - 1;
        if (bi < block_size) {
            auto tmp = prefix_helper[ai];
            prefix_helper[ai] = prefix_helper[bi];
            prefix_helper[bi] += tmp;
        }
        this_block.sync();
    }
    if (tidx < num_elements) {
        elements[tidx] = prefix_helper[element_id];
    }
}


/**
 * @internal
 * Second step of the calculation of a prefix sum. Increases the value of each
 * entry of `elements` by the total sum of all preceding blocks.
 *
 * @param block_size  thread block size for this kernel, has to be the same as
 * for `start_prefix_sum`
 * @param elements  array on which the prefix sum is to be calculated
 * @param block_sum  array storing the total sum of each block
 * @param num_elements  total number of entries in `elements`
 *
 * @note To calculate a prefix sum, first `start_prefix_sum` has to be called.
 */
template <int block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void finalize_prefix_sum(
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