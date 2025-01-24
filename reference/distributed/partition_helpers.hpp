// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_DISTRIBUTED_PARTITION_HELPERS_HPP_
#define GKO_REFERENCE_DISTRIBUTED_PARTITION_HELPERS_HPP_


#include <algorithm>

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/partition.hpp>

#include "core/base/segmented_array.hpp"
#include "core/distributed/device_partition.hpp"


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
size_type find_local_range(
    LocalIndexType idx, size_type part_id,
    device_partition<const LocalIndexType, const GlobalIndexType> partition,
    const size_type local_range_id_hint = 0)
{
    const auto& ranges_by_part = partition.ranges_by_part;
    auto local_ranges = ranges_by_part.get_segment(part_id);
    auto local_range_size =
        static_cast<size_type>(local_ranges.end - local_ranges.begin);

    auto range_starting_indices = partition.starting_indices_begin;
    if (range_starting_indices[local_ranges.begin[local_range_id_hint]] <=
            idx &&
        (local_range_id_hint == local_range_size - 1 ||
         range_starting_indices[local_ranges.begin[local_range_id_hint + 1]] >
             idx)) {
        return local_range_id_hint;
    }

    auto it = std::upper_bound(
        local_ranges.begin, local_ranges.end, idx,
        [range_starting_indices](const auto value, const auto rid) {
            return value < range_starting_indices[rid];
        });
    auto local_range_id = std::distance(local_ranges.begin, it) - 1;
    return local_range_id;
}


template <typename LocalIndexType, typename GlobalIndexType>
GlobalIndexType map_to_global(
    LocalIndexType idx,
    device_partition<const LocalIndexType, const GlobalIndexType> partition,
    size_type range_id)
{
    assert(range_id <
           std::distance(partition.offsets_begin, partition.offsets_end) - 1);
    auto range_bounds = partition.offsets_begin;
    auto starting_indices = partition.starting_indices_begin;
    return static_cast<GlobalIndexType>(idx - starting_indices[range_id]) +
           range_bounds[range_id];
}


}  // namespace gko


#endif  // GKO_REFERENCE_DISTRIBUTED_PARTITION_HELPERS_HPP_
