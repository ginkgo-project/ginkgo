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

#include "core/factorization/par_ilut_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/prefix_sum.hpp"
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
void sampleselect_filter(const ValueType *values, IndexType size,
                         const unsigned char *oracles,
                         const IndexType *partial_counts, IndexType bucket,
                         remove_complex<ValueType> *out)
{
    auto num_threads_total = ceildiv(size, items_per_thread);
    auto num_blocks =
        static_cast<IndexType>(ceildiv(num_threads_total, default_block_size));
    kernel::filter_bucket<<<num_blocks, default_block_size>>>(
        as_cuda_type(values), size, bucket, oracles, partial_counts, out,
        items_per_thread);
}


template <typename ValueType, typename IndexType>
void threshold_select(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *m,
                      IndexType rank, Array<ValueType> &tmp1,
                      Array<remove_complex<ValueType>> &tmp2,
                      remove_complex<ValueType> &threshold)
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

    auto total_counts = reinterpret_cast<IndexType *>(tmp1.get_data());
    auto partial_counts =
        reinterpret_cast<IndexType *>(tmp1.get_data() + tmp_size_totals);
    auto oracles = reinterpret_cast<unsigned char *>(
        tmp1.get_data() + tmp_size_totals + tmp_size_partials);
    auto tree =
        reinterpret_cast<AbsType *>(tmp1.get_data() + tmp_size_totals +
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
    sampleselect_filter(values, size, oracles, partial_counts, bucket.idx,
                        tmp22);

    // recursively select from smaller buckets
    int step{};
    while (bucket.size > kernel::basecase_size) {
        std::swap(tmp21, tmp22);
        const auto *tmp_in = tmp21;
        auto tmp_out = tmp22;

        sampleselect_count(exec, tmp_in, bucket.size, tree, oracles,
                           partial_counts, total_counts);
        auto new_bucket = sampleselect_find_bucket(exec, total_counts, rank);
        sampleselect_filter(tmp_in, bucket.size, oracles, partial_counts,
                            bucket.idx, tmp_out);

        rank -= new_bucket.begin;
        bucket.size = new_bucket.size;
        // we should never need more than 5 recursion steps, this would mean
        // 256^5 = 2^40. fall back to standard library algorithm in that case.
        ++step;
        if (step > 5) {
            Array<AbsType> cpu_out_array{
                exec->get_master(),
                Array<AbsType>::view(exec, bucket.size, tmp_out)};
            auto begin = cpu_out_array.get_data();
            auto end = begin + bucket.size;
            auto middle = begin + rank;
            std::nth_element(begin, middle, end);
            threshold = *middle;
            return;
        }
    }

    // base case
    auto out_ptr = reinterpret_cast<AbsType *>(tmp1.get_data());
    kernel::basecase_select<<<1, kernel::basecase_block_size>>>(
        tmp22, bucket.size, rank, out_ptr);
    threshold = exec->copy_val_to_host(out_ptr);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
