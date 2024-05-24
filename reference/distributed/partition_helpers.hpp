// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_DISTRIBUTED_PARTITION_HELPERS_HPP_
#define GKO_REFERENCE_DISTRIBUTED_PARTITION_HELPERS_HPP_


#include <algorithm>

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/partition.hpp>


namespace gko {


template <typename LocalIndexType, typename GlobalIndexType>
size_type find_range(
    GlobalIndexType idx,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    const size_type range_id_hint = 0)
{
    auto range_bounds = partition->get_range_bounds();
    auto num_ranges = partition->get_num_ranges();
    if (range_bounds[range_id_hint] <= idx &&
        idx < range_bounds[range_id_hint + 1]) {
        return range_id_hint;
    }
    auto it =
        std::upper_bound(range_bounds + 1, range_bounds + num_ranges + 1, idx);
    return static_cast<size_type>(std::distance(range_bounds + 1, it));
}


template <typename LocalIndexType, typename GlobalIndexType>
LocalIndexType map_to_local(
    GlobalIndexType idx,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    size_type range_id)
{
    auto range_bounds = partition->get_range_bounds();
    auto range_starting_indices = partition->get_range_starting_indices();
    return static_cast<LocalIndexType>(idx - range_bounds[range_id]) +
           range_starting_indices[range_id];
}


template <typename LocalIndexType, typename GlobalIndexType>
std::vector<size_type> create_part_local_ranges(
    experimental::distributed::comm_index_type part_id,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition)
{
    auto part_ids = partition->get_part_ids();
    std::vector<size_type> local_ranges;
    for (size_type range_id = 0; range_id < partition->get_num_ranges();
         ++range_id) {
        if (part_ids[range_id] == part_id) {
            local_ranges.push_back(range_id);
        }
    }
    return local_ranges;
}


template <typename LocalIndexType, typename GlobalIndexType>
size_type map_local_to_range(
    LocalIndexType idx, const std::vector<size_type>& local_ranges,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    const size_type local_range_id_hint = 0)
{
    auto range_starting_indices = partition->get_range_starting_indices();
    if (range_starting_indices[local_ranges[local_range_id_hint]] <= idx &&
        (local_range_id_hint == local_ranges.size() - 1 ||
         range_starting_indices[local_ranges[local_range_id_hint + 1]] > idx)) {
        return local_range_id_hint;
    }

    auto it = std::distance(
        local_ranges.begin(),
        std::lower_bound(
            local_ranges.begin(), local_ranges.end(), idx,
            [range_starting_indices](const auto rid, const auto idx) {
                return range_starting_indices[rid] < idx;
            }));
    return it;
}


template <typename LocalIndexType, typename GlobalIndexType>
GlobalIndexType map_to_global(
    LocalIndexType idx,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    size_type range_id)
{
    auto range_bounds = partition->get_range_bounds();
    auto starting_indices = partition->get_range_starting_indices();
    return static_cast<GlobalIndexType>(idx - starting_indices[range_id]) +
           range_bounds[range_id];
}


}  // namespace gko


#endif  // GKO_REFERENCE_DISTRIBUTED_PARTITION_HELPERS_HPP_
