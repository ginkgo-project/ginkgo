// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_PARTITION_HPP
#define GINKGO_PARTITION_HPP

#include <ginkgo/core/distributed/partition.hpp>

#include "core/base/segmented_array.hpp"

namespace gko {


template <typename LocalIndexType, typename GlobalIndexType>
struct device_partition {
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;
    using comm_index_type = experimental::distributed::comm_index_type;

    comm_index_type num_parts;
    comm_index_type num_empty_parts;
    size_type size;
    global_index_type* offsets_begin;
    global_index_type* offsets_end;
    local_index_type* starting_indices_begin;
    local_index_type* starting_indices_end;
    local_index_type* part_sizes_begin;
    local_index_type* part_sizes_end;
    const comm_index_type* part_ids_begin;
    const comm_index_type* part_ids_end;
    device_segmented_array<const size_type> ranges_by_part;
};


/**
 * Create device_segmented_array from a segmented_array.
 */
template <typename LocalIndexType, typename GlobalIndexType>
constexpr device_partition<const LocalIndexType, const GlobalIndexType>
to_device(
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition)
{
    auto num_ranges = partition->get_num_ranges();
    auto num_parts = partition->get_num_parts();
    return {num_parts,
            partition->get_num_empty_parts(),
            partition->get_size(),
            partition->get_range_bounds(),
            partition->get_range_bounds() + num_ranges + 1,
            partition->get_range_starting_indices(),
            partition->get_range_starting_indices() + num_ranges,
            partition->get_part_sizes(),
            partition->get_part_sizes() + num_parts,
            partition->get_part_ids(),
            partition->get_part_ids() + num_parts,
            to_device(partition->get_ranges_by_part())};
}

/**
 * Explicitly create a const version of device_segmented_array.
 *
 * This is mostly relevant for tests.
 */
template <typename LocalIndexType, typename GlobalIndexType>
constexpr device_partition<const LocalIndexType, const GlobalIndexType>
to_device_const(
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition)
{
    auto num_ranges = partition->get_num_ranges();
    auto num_parts = partition->get_num_parts();
    return {num_parts,
            partition->get_num_empty_parts(),
            partition->get_size(),
            partition->get_range_bounds(),
            partition->get_range_bounds() + num_ranges + 1,
            partition->get_range_starting_indices(),
            partition->get_range_starting_indices() + num_ranges,
            partition->get_part_sizes(),
            partition->get_part_sizes() + num_parts,
            partition->get_part_ids(),
            partition->get_part_ids() + num_parts,
            to_device(partition->get_ranges_by_part())};
}


}  // namespace gko


#endif  // GINKGO_PARTITION_HPP
