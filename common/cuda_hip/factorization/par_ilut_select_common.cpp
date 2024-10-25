// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// clang-format off
// prevent compilation failure related to disappearing assert(...) statements
#include "common/cuda_hip/base/runtime.hpp"
// clang-format on


#include "common/cuda_hip/factorization/par_ilut_select_common.hpp"

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/components/atomic.hpp"
#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/prefix_sum.hpp"
#include "common/cuda_hip/components/searching.hpp"
#include "common/cuda_hip/components/sorting.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/factorization/par_ilut_select_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/par_ilut_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


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
#ifdef GKO_COMPILING_HIP
    if constexpr (std::is_same<remove_complex<ValueType>, half>::value) {
        // HIP does not support 16bit atomic operation
        GKO_NOT_SUPPORTED(values);
    } else
#endif
    {
        // pick sample, build searchtree
        kernel::build_searchtree<<<1, bucket_count, 0, exec->get_stream()>>>(
            as_device_type(values), size, as_device_type(tree));
    }
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

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_WITH_HALF(DECLARE_SSSS_COUNT);


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
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
