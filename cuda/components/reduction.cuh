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

#ifndef GKO_CUDA_COMPONENTS_REDUCTION_CUH_
#define GKO_CUDA_COMPONENTS_REDUCTION_CUH_


#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/shuffle.cuh"
#include "cuda/components/synchronization.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace warp {


/**
 * @internal
 *
 * Computes a reduction using the binary operation `reduce_op` on a sub-warp of
 * `subwarp_size` threads. The `subwarp_size` must be a power of 2, and not
 * larger than cuda_config::warp_size.
 * Each thread contributes with one element `local_data`. The local thread
 * element is always passed as the first parameter to the `reduce_op`.
 * The function returns the result of the reduction on all threads.
 *
 * @note The function is guarantied to return the correct value on all threads
 *       only if `reduce_op` is commutative (in addition to being associative).
 *       Otherwise, the correct value is returned only to the thread with
 *       subwarp index 0. Use warp::shuffle to exchange it with other threads.
 */
template <int32 subwarp_size, typename ValueType, typename Operator>
__device__ __forceinline__ ValueType reduce(ValueType local_data,
                                            Operator reduce_op = Operator{})
{
#pragma unroll
    for (int32 bitmask = 1; bitmask < subwarp_size; bitmask <<= 1) {
        const auto remote_data = shuffle_xor(local_data, bitmask, subwarp_size);
        local_data = reduce_op(local_data, remote_data);
    }
    return local_data;
}


/**
 * @internal
 *
 * Returns the index of the thread that has the element with the largest
 * magnitude among all the threads in the sub-warp of `subwarp_size` threads.
 * Restrictions on subwarp_size from warp::reduce apply.
 * Only the values from threads which set `is_pivoted` to `false` will be
 * considered.
 */
template <int32 subwarp_size, typename ValueType>
__device__ __forceinline__ int choose_pivot(ValueType local_data,
                                            bool is_pivoted)
{
    using real = remove_complex<ValueType>;
    real lmag = is_pivoted ? -one<real>() : abs(local_data);
    const auto pivot =
        reduce<subwarp_size>(threadIdx.x, [&](int lidx, int ridx) {
            const auto rmag = shuffle(lmag, ridx, subwarp_size);
            if (rmag > lmag) {
                lmag = rmag;
                lidx = ridx;
            }
            return lidx;
        });
    // pivot operator not commutative, make sure everyone has the same pivot
    return shuffle(pivot, 0, subwarp_size);
}


}  // namespace warp


namespace block {


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
template <size_type block_size, int32 subwarp_size, typename ValueType,
          typename Operator>
__device__ void reduce(ValueType *data, Operator reduce_op = Operator{})
{
    const auto local_id = thread::get_local_thread_id<subwarp_size>();

#pragma unroll
    for (int k = block_size / 2; k >= cuda_config::warp_size; k /= 2) {
        block::synchronize();
        if (local_id < k) {
            data[local_id] = reduce_op(data[local_id], data[local_id + k]);
        }
    }

    if (local_id >= cuda_config::warp_size) {
        return;
    }
    auto result = data[local_id];
    // TODO: implement function-level launch configuration to use warp::reduce
#pragma unroll
    for (int bitmask = 1; bitmask < cuda_config::warp_size; bitmask <<= 1) {
        result = reduce_op(result, warp::shuffle_xor(result, bitmask));
    }

    if (local_id == 0) {
        data[0] = result;
    }
}


}  // namespace block
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_REDUCTION_CUH_
