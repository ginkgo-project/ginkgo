// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/partition_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/math.hpp>


#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace partition {


template <typename LocalIndexType, typename GlobalIndexType>
void build_starting_indices(std::shared_ptr<const DefaultExecutor> exec,
                            const GlobalIndexType* range_offsets,
                            const int* range_parts, size_type num_ranges,
                            int num_parts, int& num_empty_parts,
                            LocalIndexType* ranks, LocalIndexType* sizes)
{
    std::fill_n(sizes, num_parts, 0);
    auto num_threads = static_cast<size_type>(omp_get_max_threads());
    auto size_per_thread =
        static_cast<size_type>(ceildiv(num_ranges, num_threads));
    vector<LocalIndexType> local_sizes(num_parts * num_threads, 0, {exec});
    int tmp = num_empty_parts;
#pragma omp parallel
    {
        auto thread_id = static_cast<size_type>(omp_get_thread_num());
        auto thread_begin = size_per_thread * thread_id;
        auto thread_end = std::min(num_ranges, thread_begin + size_per_thread);
        auto base = num_parts * thread_id;
        // local exclusive prefix sum
        for (auto range = thread_begin; range < thread_end; range++) {
            auto begin = range_offsets[range];
            auto end = range_offsets[range + 1];
            auto part = range_parts[range];
            ranks[range] = local_sizes[part + base];
            local_sizes[part + base] += end - begin;
        }
#pragma omp barrier
        // exclusive prefix sum over local sizes
        // FIXME: PGI/NVHPC(22.7) doesn't like reduction with references
#pragma omp for reduction(+ : tmp)
        for (comm_index_type part = 0; part < num_parts; ++part) {
            LocalIndexType size{};
            for (size_type thread = 0; thread < num_threads; ++thread) {
                auto idx = num_parts * thread + part;
                auto local_size = local_sizes[idx];
                local_sizes[idx] = size;
                size += local_size;
            }
            sizes[part] = size;
            tmp += size == 0 ? 1 : 0;
        }
        // add global baselines to local ranks
        for (auto range = thread_begin; range < thread_end; range++) {
            auto part = range_parts[range];
            ranks[range] += local_sizes[part + base];
        }
    }
    num_empty_parts = tmp;
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PARTITION_BUILD_STARTING_INDICES);


}  // namespace partition
}  // namespace omp
}  // namespace kernels
}  // namespace gko
