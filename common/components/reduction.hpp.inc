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
 *
 * Computes a reduction using the binary operation `reduce_op` on a group
 * `group`. Each thread contributes with one element `local_data`. The local
 * thread element is always passed as the first parameter to the `reduce_op`.
 * The function returns the result of the reduction on all threads.
 *
 * @note The function is guaranteed to return the correct value on all threads
 *       only if `reduce_op` is commutative (in addition to being associative).
 *       Otherwise, the correct value is returned only to the thread with
 *       subwarp index 0.
 */
template <
    typename Group, typename ValueType, typename Operator,
    typename = xstd::enable_if_t<group::is_communicator_group<Group>::value>>
__device__ __forceinline__ ValueType reduce(const Group &group,
                                            ValueType local_data,
                                            Operator reduce_op = Operator{})
{
#pragma unroll
    for (int32 bitmask = 1; bitmask < group.size(); bitmask <<= 1) {
        const auto remote_data = group.shfl_xor(local_data, bitmask);
        local_data = reduce_op(local_data, remote_data);
    }
    return local_data;
}


/**
 * @internal
 *
 * Returns the index of the thread that has the element with the largest
 * magnitude among all the threads in the group.
 * Only the values from threads which set `is_pivoted` to `false` will be
 * considered.
 */
template <
    typename Group, typename ValueType,
    typename = xstd::enable_if_t<group::is_communicator_group<Group>::value>>
__device__ __forceinline__ int choose_pivot(const Group &group,
                                            ValueType local_data,
                                            bool is_pivoted)
{
    using real = remove_complex<ValueType>;
    real lmag = is_pivoted ? -one<real>() : abs(local_data);
    const auto pivot =
        reduce(group, group.thread_rank(), [&](int lidx, int ridx) {
            const auto rmag = group.shfl(lmag, ridx);
            if (rmag > lmag) {
                lmag = rmag;
                lidx = ridx;
            }
            return lidx;
        });
    // pivot operator not commutative, make sure everyone has the same pivot
    return group.shfl(pivot, 0);
}


/**
 * @internal
 *
 * Computes a reduction using the binary operation `reduce_op` on entire block.
 * The data for the reduction is taken from the `data` array which has to be of
 * size `block_size` and accessible from all threads. The `data` array is also
 * used as work space (so its content will be destroyed in the process), as well
 * as to store the return value - which is stored in the 0-th position of the
 * array.
 */
template <
    typename Group, typename ValueType, typename Operator,
    typename = xstd::enable_if_t<group::is_synchronizable_group<Group>::value>>
__device__ void reduce(const Group &__restrict__ group,
                       ValueType *__restrict__ data,
                       Operator reduce_op = Operator{})
{
    const auto local_id = group.thread_rank();

#pragma unroll
    for (int k = group.size() / 2; k >= config::warp_size; k /= 2) {
        group.sync();
        if (local_id < k) {
            data[local_id] = reduce_op(data[local_id], data[local_id + k]);
        }
    }

    const auto warp = group::tiled_partition<config::warp_size>(group);
    const auto warp_id = group.thread_rank() / warp.size();
    if (warp_id > 0) {
        return;
    }
    auto result = reduce(warp, data[warp.thread_rank()], reduce_op);
    if (warp.thread_rank() == 0) {
        data[0] = result;
    }
}


/**
 * @internal
 *
 * Computes a reduction using the binary operation `reduce_op` on an array
 * `source` of any size. Has to be called a second time on `result` to reduce
 * an array larger than `block_size`.
 */
template <typename Operator, typename ValueType>
__device__ void reduce_array(size_type size,
                             const ValueType *__restrict__ source,
                             ValueType *__restrict__ result,
                             Operator reduce_op = Operator{})
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    auto thread_result = zero<ValueType>();
    for (auto i = tidx; i < size; i += blockDim.x * gridDim.x) {
        thread_result = reduce_op(thread_result, source[i]);
    }
    result[threadIdx.x] = thread_result;

    group::this_thread_block().sync();

    // Stores the result of the reduction inside `result[0]`
    reduce(group::this_thread_block(), result, reduce_op);
}


/**
 * @internal
 *
 * Computes a reduction using the add operation (+) on an array
 * `source` of any size. Has to be called a second time on `result` to reduce
 * an array larger than `default_block_size`.
 */
template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void reduce_add_array(
    size_type size, const ValueType *__restrict__ source,
    ValueType *__restrict__ result)
{
    __shared__ UninitializedArray<ValueType, default_block_size> block_sum;
    reduce_array(size, source, static_cast<ValueType *>(block_sum),
                 [](const ValueType &x, const ValueType &y) { return x + y; });

    if (threadIdx.x == 0) {
        result[blockIdx.x] = block_sum[0];
    }
}
