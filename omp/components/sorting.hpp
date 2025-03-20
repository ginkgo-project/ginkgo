// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <numeric>

#include <omp.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>

namespace gko {
namespace kernels {
namespace omp {


template <int num_buckets, typename Iterator, typename BucketIndexOp>
std::array<int64, num_buckets + 1> bucket_sort(Iterator begin, Iterator end,
                                               Iterator out_begin,
                                               BucketIndexOp bucket_op,
                                               gko::array<int64>& tmp)
{
    using index_type = typename std::iterator_traits<Iterator>::difference_type;
    const auto size = end - begin;
    const auto num_threads = omp_get_max_threads();
    const auto tmp_size = num_threads * num_buckets;
    if (tmp.get_size() < tmp_size) {
        tmp.resize_and_reset(tmp_size);
    }
    const auto sums = tmp.get_data();
    std::fill_n(sums, tmp_size, 0);
    std::array<int64, num_buckets + 1> global_offsets{};
#pragma omp parallel
    {
        const auto tid = omp_get_thread_num();
        auto counts = sums + tid * num_buckets;
#pragma omp for
        for (index_type i = 0; i < size; i++) {
            const auto value = *(begin + i);
            const auto bucket = bucket_op(value);
            assert(bucket >= 0);
            assert(bucket < num_buckets);
            counts[bucket]++;
        }
#pragma omp barrier
#pragma omp single
        {
            std::array<int64, num_buckets> offsets{};
            for (int tid = 0; tid < num_threads; tid++) {
                for (int i = 0; i < num_buckets; i++) {
                    const auto value = sums[tid * num_buckets + i];
                    sums[tid * num_buckets + i] = offsets[i];
                    offsets[i] += value;
                }
            }
            std::copy_n(offsets.begin(), num_buckets, global_offsets.begin());
            std::exclusive_scan(global_offsets.begin(), global_offsets.end(),
                                global_offsets.begin(), index_type{});
        }
#pragma omp for
        for (index_type i = 0; i < size; i++) {
            const auto value = *(begin + i);
            const auto bucket = bucket_op(value);
            assert(bucket >= 0);
            assert(bucket < num_buckets);
            const auto output_pos = counts[bucket]++ + global_offsets[bucket];
            *(out_begin + output_pos) = value;
        }
    }
    return global_offsets;
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
