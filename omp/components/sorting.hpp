// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <numeric>

#include <omp.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/base/index_range.hpp"

namespace gko {
namespace kernels {
namespace omp {


template <int num_buckets>
size_type bucket_sort_workspace_size(size_type num_elements)
{
    const auto num_threads = static_cast<size_type>(omp_get_max_threads());
    return num_threads * num_buckets;
}


template <int num_buckets, typename IndexType, typename InputIterator,
          typename OutputIterator, typename BucketIndexOp>
std::array<IndexType, num_buckets + 1> bucket_sort(InputIterator begin,
                                                   InputIterator end,
                                                   OutputIterator out_begin,
                                                   BucketIndexOp bucket_op,
                                                   gko::array<IndexType>& tmp)
{
    const auto size = static_cast<IndexType>(end - begin);
    const auto tmp_size =
        bucket_sort_workspace_size<num_buckets>(static_cast<size_type>(size));
    if (tmp.get_size() < tmp_size) {
        tmp.resize_and_reset(tmp_size);
    }
    const auto counts = tmp.get_data();
    std::fill_n(counts, tmp_size, 0);
    std::array<IndexType, num_buckets + 1> global_offsets{};
#pragma omp parallel
    {
        const auto tid = static_cast<IndexType>(omp_get_thread_num());
        const auto num_threads = omp_get_num_threads();
        const auto work_per_thread =
            static_cast<IndexType>(ceildiv(size, num_threads));
        const auto local_begin = std::min(tid * work_per_thread, size);
        const auto local_end = std::min(local_begin + work_per_thread, size);
        auto local_counts = counts + tid * num_buckets;
        for (auto i : irange{local_begin, local_end}) {
            const auto value = *(begin + i);
            const auto bucket = bucket_op(value);
            assert(bucket >= 0);
            assert(bucket < num_buckets);
            local_counts[bucket]++;
        }
#pragma omp barrier
#pragma omp single
        {
            std::array<IndexType, num_buckets> offsets{};
            for (int tid = 0; tid < num_threads; tid++) {
                for (int i = 0; i < num_buckets; i++) {
                    const auto value = counts[tid * num_buckets + i];
                    counts[tid * num_buckets + i] = offsets[i];
                    offsets[i] += value;
                }
            }
            std::copy_n(offsets.begin(), num_buckets, global_offsets.begin());
            std::exclusive_scan(global_offsets.begin(), global_offsets.end(),
                                global_offsets.begin(), IndexType{});
        }
        for (auto i : irange{local_begin, local_end}) {
            const auto value = *(begin + i);
            const auto bucket = bucket_op(value);
            assert(bucket >= 0);
            assert(bucket < num_buckets);
            const auto out_pos =
                local_counts[bucket]++ + global_offsets[bucket];
            *(out_begin + out_pos) = value;
        }
    }
    return global_offsets;
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
