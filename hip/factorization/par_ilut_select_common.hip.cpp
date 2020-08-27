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

// force-top: on
// prevent compilation failure related to disappearing assert(...) statements
#include <hip/hip_runtime.h>
// force-top: off


#include "hip/factorization/par_ilut_select_common.hip.hpp"


#include "core/components/prefix_sum.hpp"
#include "core/factorization/par_ilut_kernels.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/prefix_sum.hip.hpp"
#include "hip/components/searching.hip.hpp"
#include "hip/components/sorting.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


#include "common/factorization/par_ilut_select_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void sampleselect_count(std::shared_ptr<const DefaultExecutor> exec,
                        const ValueType *values, IndexType size,
                        remove_complex<ValueType> *tree, unsigned char *oracles,
                        IndexType *partial_counts, IndexType *total_counts)
{
    constexpr auto bucket_count = kernel::searchtree_width;
    auto num_threads_total = ceildiv(size, items_per_thread);
    auto num_blocks =
        static_cast<IndexType>(ceildiv(num_threads_total, default_block_size));
    // pick sample, build searchtree
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::build_searchtree), dim3(1),
                       dim3(bucket_count), 0, 0, as_hip_type(values), size,
                       tree);
    // determine bucket sizes
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::count_buckets), dim3(num_blocks),
                       dim3(default_block_size), 0, 0, as_hip_type(values),
                       size, tree, partial_counts, oracles, items_per_thread);
    // compute prefix sum and total sum over block-local values
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::block_prefix_sum),
                       dim3(bucket_count), dim3(default_block_size), 0, 0,
                       partial_counts, total_counts, num_blocks);
    // compute prefix sum over bucket counts
    components::prefix_sum(exec, total_counts, bucket_count + 1);
}


#define DECLARE_SSSS_COUNT(ValueType, IndexType)                               \
    void sampleselect_count(std::shared_ptr<const DefaultExecutor> exec,       \
                            const ValueType *values, IndexType size,           \
                            remove_complex<ValueType> *tree,                   \
                            unsigned char *oracles, IndexType *partial_counts, \
                            IndexType *total_counts)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SSSS_COUNT);


template <typename IndexType>
sampleselect_bucket<IndexType> sampleselect_find_bucket(
    std::shared_ptr<const DefaultExecutor> exec, IndexType *prefix_sum,
    IndexType rank)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::find_bucket), dim3(1),
                       dim3(config::warp_size), 0, 0, prefix_sum, rank);
    IndexType values[3]{};
    exec->get_master()->copy_from(exec.get(), 3, prefix_sum, values);
    return {values[0], values[1], values[2]};
}


#define DECLARE_SSSS_FIND_BUCKET(IndexType)                                 \
    sampleselect_bucket<IndexType> sampleselect_find_bucket(                \
        std::shared_ptr<const DefaultExecutor> exec, IndexType *prefix_sum, \
        IndexType rank)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(DECLARE_SSSS_FIND_BUCKET);


}  // namespace par_ilut_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko
