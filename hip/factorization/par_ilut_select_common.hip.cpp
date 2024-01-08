// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// force-top: on
// prevent compilation failure related to disappearing assert(...) statements
#include <hip/hip_runtime.h>
// force-top: off


#include "hip/factorization/par_ilut_select_common.hip.hpp"


#include "core/components/prefix_sum_kernels.hpp"
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


#include "common/cuda_hip/factorization/par_ilut_select_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void sampleselect_count(std::shared_ptr<const DefaultExecutor> exec,
                        const ValueType* values, IndexType size,
                        remove_complex<ValueType>* tree, unsigned char* oracles,
                        IndexType* partial_counts, IndexType* total_counts)
{
    constexpr auto bucket_count = kernel::searchtree_width;
    auto num_threads_total = ceildiv(size, items_per_thread);
    auto num_blocks =
        static_cast<IndexType>(ceildiv(num_threads_total, default_block_size));
    // pick sample, build searchtree
    kernel::build_searchtree<<<1, bucket_count, 0, exec->get_stream()>>>(
        as_device_type(values), size, as_device_type(tree));
    // determine bucket sizes
    if (num_blocks > 0) {
        kernel::count_buckets<<<num_blocks, default_block_size, 0,
                                exec->get_stream()>>>(
            as_device_type(values), size, as_device_type(tree), partial_counts,
            oracles, items_per_thread);
    }
    // compute prefix sum and total sum over block-local values
    kernel::block_prefix_sum<<<bucket_count, default_block_size, 0,
                               exec->get_stream()>>>(partial_counts,
                                                     total_counts, num_blocks);
    // compute prefix sum over bucket counts
    components::prefix_sum_nonnegative(exec, total_counts, bucket_count + 1);
}


#define DECLARE_SSSS_COUNT(ValueType, IndexType)                               \
    void sampleselect_count(std::shared_ptr<const DefaultExecutor> exec,       \
                            const ValueType* values, IndexType size,           \
                            remove_complex<ValueType>* tree,                   \
                            unsigned char* oracles, IndexType* partial_counts, \
                            IndexType* total_counts)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SSSS_COUNT);


template <typename IndexType>
sampleselect_bucket<IndexType> sampleselect_find_bucket(
    std::shared_ptr<const DefaultExecutor> exec, IndexType* prefix_sum,
    IndexType rank)
{
    kernel::find_bucket<<<1, config::warp_size, 0, exec->get_stream()>>>(
        prefix_sum, rank);
    IndexType values[3]{};
    exec->get_master()->copy_from(exec, 3, prefix_sum, values);
    return {values[0], values[1], values[2]};
}


#define DECLARE_SSSS_FIND_BUCKET(IndexType)                                 \
    sampleselect_bucket<IndexType> sampleselect_find_bucket(                \
        std::shared_ptr<const DefaultExecutor> exec, IndexType* prefix_sum, \
        IndexType rank)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(DECLARE_SSSS_FIND_BUCKET);


}  // namespace par_ilut_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko
