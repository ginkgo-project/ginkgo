// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilut_kernels.hpp"


#include <limits>


#include <CL/sycl.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"
#include "dpcpp/components/prefix_sum.dp.hpp"
#include "dpcpp/components/searching.dp.hpp"
#include "dpcpp/components/sorting.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/factorization/par_ilut_select_common.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


#include "dpcpp/factorization/par_ilut_select_kernels.hpp.inc"


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
    kernel::build_searchtree(1, bucket_count, 0, exec->get_queue(), values,
                             size, tree);
    // determine bucket sizes
    kernel::count_buckets(num_blocks, default_block_size, 0, exec->get_queue(),
                          values, size, tree, partial_counts, oracles,
                          items_per_thread);
    // compute prefix sum and total sum over block-local values
    kernel::block_prefix_sum(bucket_count, default_block_size, 0,
                             exec->get_queue(), partial_counts, total_counts,
                             num_blocks);
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
    kernel::find_bucket(1, config::warp_size, 0, exec->get_queue(), prefix_sum,
                        rank);
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
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
