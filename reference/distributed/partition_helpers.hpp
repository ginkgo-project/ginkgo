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


}  // namespace gko


#endif  // GKO_REFERENCE_DISTRIBUTED_PARTITION_HELPERS_HPP_
