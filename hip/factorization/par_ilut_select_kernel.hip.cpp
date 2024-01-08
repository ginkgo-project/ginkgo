// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilut_kernels.hpp"


#include <algorithm>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/prefix_sum.hip.hpp"
#include "hip/components/searching.hip.hpp"
#include "hip/components/sorting.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/factorization/par_ilut_select_common.hip.hpp"


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
void sampleselect_filter(std::shared_ptr<const DefaultExecutor> exec,
                         const ValueType* values, IndexType size,
                         const unsigned char* oracles,
                         const IndexType* partial_counts, IndexType bucket,
                         remove_complex<ValueType>* out)
{
    auto num_threads_total = ceildiv(size, items_per_thread);
    auto num_blocks =
        static_cast<IndexType>(ceildiv(num_threads_total, default_block_size));
    if (num_blocks > 0) {
        kernel::filter_bucket<<<num_blocks, default_block_size, 0,
                                exec->get_stream()>>>(
            as_device_type(values), size, bucket, oracles, partial_counts,
            as_device_type(out), items_per_thread);
    }
}


template <typename ValueType, typename IndexType>
void threshold_select(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* m,
                      IndexType rank, array<ValueType>& tmp1,
                      array<remove_complex<ValueType>>& tmp2,
                      remove_complex<ValueType>& threshold)
{
    auto values = m->get_const_values();
    IndexType size = m->get_num_stored_elements();
    using AbsType = remove_complex<ValueType>;
    constexpr auto bucket_count = kernel::searchtree_width;
    auto max_num_threads = ceildiv(size, items_per_thread);
    auto max_num_blocks = ceildiv(max_num_threads, default_block_size);

    size_type tmp_size_totals =
        ceildiv((bucket_count + 1) * sizeof(IndexType), sizeof(ValueType));
    size_type tmp_size_partials = ceildiv(
        bucket_count * max_num_blocks * sizeof(IndexType), sizeof(ValueType));
    size_type tmp_size_oracles =
        ceildiv(size * sizeof(unsigned char), sizeof(ValueType));
    size_type tmp_size_tree =
        ceildiv(kernel::searchtree_size * sizeof(AbsType), sizeof(ValueType));
    size_type tmp_size_vals =
        size / bucket_count * 4;  // pessimistic estimate for temporary storage
    size_type tmp_size =
        tmp_size_totals + tmp_size_partials + tmp_size_oracles + tmp_size_tree;
    tmp1.resize_and_reset(tmp_size);
    tmp2.resize_and_reset(tmp_size_vals);

    auto total_counts = reinterpret_cast<IndexType*>(tmp1.get_data());
    auto partial_counts =
        reinterpret_cast<IndexType*>(tmp1.get_data() + tmp_size_totals);
    auto oracles = reinterpret_cast<unsigned char*>(
        tmp1.get_data() + tmp_size_totals + tmp_size_partials);
    auto tree =
        reinterpret_cast<AbsType*>(tmp1.get_data() + tmp_size_totals +
                                   tmp_size_partials + tmp_size_oracles);

    sampleselect_count(exec, values, size, tree, oracles, partial_counts,
                       total_counts);

    // determine bucket with correct rank, use bucket-local rank
    auto bucket = sampleselect_find_bucket(exec, total_counts, rank);
    rank -= bucket.begin;

    if (bucket.size * 2 > tmp_size_vals) {
        // we need to reallocate tmp2
        tmp2.resize_and_reset(bucket.size * 2);
    }
    auto tmp21 = tmp2.get_data();
    auto tmp22 = tmp2.get_data() + bucket.size;
    // extract target bucket
    sampleselect_filter(exec, values, size, oracles, partial_counts, bucket.idx,
                        tmp22);

    // recursively select from smaller buckets
    int step{};
    while (bucket.size > kernel::basecase_size) {
        std::swap(tmp21, tmp22);
        const auto* tmp_in = tmp21;
        auto tmp_out = tmp22;

        sampleselect_count(exec, tmp_in, bucket.size, tree, oracles,
                           partial_counts, total_counts);
        auto new_bucket = sampleselect_find_bucket(exec, total_counts, rank);
        sampleselect_filter(exec, tmp_in, bucket.size, oracles, partial_counts,
                            bucket.idx, tmp_out);

        rank -= new_bucket.begin;
        bucket.size = new_bucket.size;
        // we should never need more than 5 recursion steps, this would mean
        // 256^5 = 2^40. fall back to standard library algorithm in that case.
        ++step;
        if (step > 5) {
            array<AbsType> cpu_out_array{
                exec->get_master(),
                make_array_view(exec, bucket.size, tmp_out)};
            auto begin = cpu_out_array.get_data();
            auto end = begin + bucket.size;
            auto middle = begin + rank;
            std::nth_element(begin, middle, end);
            threshold = *middle;
            return;
        }
    }

    // base case
    auto out_ptr = reinterpret_cast<AbsType*>(tmp1.get_data());
    kernel::basecase_select<<<1, kernel::basecase_block_size, 0,
                              exec->get_stream()>>>(
        as_device_type(tmp22), bucket.size, rank, as_device_type(out_ptr));
    threshold = exec->copy_val_to_host(out_ptr);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko
