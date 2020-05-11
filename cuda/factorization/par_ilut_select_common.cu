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

#include "cuda/factorization/par_ilut_select_common.cuh"


#include "cuda/base/math.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/prefix_sum.cuh"
#include "cuda/components/searching.cuh"
#include "cuda/components/sorting.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/factorization/par_ilut_select_common.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


#include "common/factorization/par_ilut_select_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void ssss_count(const ValueType *values, IndexType size,
                remove_complex<ValueType> *tree, unsigned char *oracles,
                IndexType *partial_counts, IndexType *total_counts)
{
    constexpr auto bucket_count = kernel::searchtree_width;
    auto num_threads_total = ceildiv(size, items_per_thread);
    auto num_blocks =
        static_cast<IndexType>(ceildiv(num_threads_total, default_block_size));
    // pick sample, build searchtree
    kernel::build_searchtree<<<1, bucket_count>>>(as_cuda_type(values), size,
                                                  tree);
    // determine bucket sizes
    kernel::count_buckets<<<num_blocks, default_block_size>>>(
        as_cuda_type(values), size, tree, partial_counts, oracles,
        items_per_thread);
    // compute prefix sum and total sum over block-local values
    kernel::block_prefix_sum<<<bucket_count, default_block_size>>>(
        partial_counts, total_counts, num_blocks);
    // compute prefix sum over bucket counts
    start_prefix_sum<bucket_count><<<1, bucket_count>>>(
        bucket_count, total_counts, total_counts + bucket_count);
}


#define DECLARE_SSSS_COUNT(ValueType, IndexType)                             \
    void ssss_count(const ValueType *values, IndexType size,                 \
                    remove_complex<ValueType> *tree, unsigned char *oracles, \
                    IndexType *partial_counts, IndexType *total_counts)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SSSS_COUNT);


template <typename IndexType>
ssss_bucket<IndexType> ssss_find_bucket(
    std::shared_ptr<const DefaultExecutor> exec, IndexType *prefix_sum,
    IndexType rank)
{
    kernel::find_bucket<<<1, config::warp_size>>>(prefix_sum, rank);
    IndexType values[3]{};
    exec->get_master()->copy_from(exec.get(), 3, prefix_sum, values);
    return {values[0], values[1], values[2]};
}


#define DECLARE_SSSS_FIND_BUCKET(IndexType)                                 \
    ssss_bucket<IndexType> ssss_find_bucket(                                \
        std::shared_ptr<const DefaultExecutor> exec, IndexType *prefix_sum, \
        IndexType rank)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(DECLARE_SSSS_FIND_BUCKET);


}  // namespace par_ilut_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko