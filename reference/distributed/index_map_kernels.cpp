// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"

#include <ginkgo/core/distributed/index_map.hpp>

#include "core/base/allocator.hpp"
#include "core/base/segmented_array.hpp"
#include "reference/distributed/partition_helpers.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace index_map {


template <typename LocalIndexType, typename GlobalIndexType>
void build_mapping(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        part,
    const array<GlobalIndexType>& recv_connections,
    array<experimental::distributed::comm_index_type>& remote_part_ids,
    array<LocalIndexType>& remote_local_idxs,
    array<GlobalIndexType>& remote_global_idxs, array<int64>& remote_sizes)
{
    using experimental::distributed::comm_index_type;
    auto part_ids = part->get_part_ids();

    vector<GlobalIndexType> unique_indices(recv_connections.get_size(), {exec});
    std::copy_n(recv_connections.get_const_data(), recv_connections.get_size(),
                unique_indices.begin());

    auto find_part = [&](GlobalIndexType idx) {
        auto range_id = find_range(idx, part, 0);
        return part_ids[range_id];
    };

    // sort by (part-id, global-id)
    std::sort(unique_indices.begin(), unique_indices.end(),
              [&](const auto a, const auto b) {
                  auto part_a = find_part(a);
                  auto part_b = find_part(b);
                  return std::tie(part_a, a) < std::tie(part_b, b);
              });

    // make unique by (part-id, global-id)
    auto unique_indices_end =
        std::unique(unique_indices.begin(), unique_indices.end(),
                    [&](const auto a, const auto b) {
                        auto part_a = find_part(a);
                        auto part_b = find_part(b);
                        return std::tie(part_a, a) == std::tie(part_b, b);
                    });
    auto unique_size =
        std::distance(unique_indices.begin(), unique_indices_end);

    remote_global_idxs.resize_and_reset(unique_size);
    remote_local_idxs.resize_and_reset(unique_size);

    // store unique global indices
    std::copy_n(unique_indices.begin(), unique_size,
                remote_global_idxs.get_data());

    // store and transform
    std::transform(unique_indices.begin(), unique_indices_end,
                   remote_local_idxs.get_data(), [&](const auto idx) {
                       auto range = find_range(idx, part, 0);
                       return map_to_local(idx, part, range);
                   });

    // get part-ids
    vector<comm_index_type> full_part_ids(unique_size, exec);
    std::transform(unique_indices.begin(), unique_indices_end,
                   full_part_ids.begin(),
                   [&](const auto idx) { return find_part(idx); });

    vector<comm_index_type> unique_part_ids(full_part_ids, exec);
    auto unique_part_ids_end =
        std::unique(unique_part_ids.begin(), unique_part_ids.end());

    auto unique_part_ids_size =
        std::distance(unique_part_ids.begin(), unique_part_ids_end);
    remote_part_ids.resize_and_reset(unique_part_ids_size);
    std::copy(unique_part_ids.begin(), unique_part_ids_end,
              remote_part_ids.get_data());

    // get recv size per part
    vector<size_type> full_remote_sizes(part->get_num_parts(), exec);
    for (size_type i = 0; i < full_part_ids.size(); ++i) {
        full_remote_sizes[full_part_ids[i]]++;
    }
    remote_sizes.resize_and_reset(unique_part_ids_size);
    std::copy_if(full_remote_sizes.begin(), full_remote_sizes.end(),
                 remote_sizes.get_data(), [](const auto s) { return s > 0; });
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_BUILD_MAPPING);


template <typename LocalIndexType, typename GlobalIndexType>
void map_to_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    const array<experimental::distributed::comm_index_type>& remote_target_ids,
    device_segmented_array<const GlobalIndexType> remote_global_idxs,
    experimental::distributed::comm_index_type rank,
    const array<GlobalIndexType>& global_ids,
    experimental::distributed::index_space is, array<LocalIndexType>& local_ids)
{
    auto part_ids = partition->get_part_ids();
    auto range_bounds = partition->get_range_bounds();
    auto range_starting_idxs = partition->get_range_starting_indices();

    local_ids.resize_and_reset(global_ids.get_size());

    auto map_local = [&](const auto gid) {
        auto range_id = find_range(gid, partition);
        auto part_id = part_ids[range_id];

        return part_id == rank
                   ? static_cast<LocalIndexType>(gid - range_bounds[range_id]) +
                         range_starting_idxs[range_id]
                   : invalid_index<LocalIndexType>();
    };

    auto map_non_local = [&](const auto gid) {
        auto range_id = find_range(gid, partition);
        auto part_id = part_ids[range_id];

        // can't do binary search on whole remote_target_idxs array,
        // since the array is first sorted by part-id and then by
        // global index. As a result, the array is not sorted wrt.
        // the global indexing. So find the part-id that corresponds
        // to the global index first
        auto set_id =
            std::distance(remote_target_ids.get_const_data(),
                          std::lower_bound(remote_target_ids.get_const_data(),
                                           remote_target_ids.get_const_data() +
                                               remote_target_ids.get_size(),
                                           part_id));

        if (set_id == remote_target_ids.get_size()) {
            return invalid_index<LocalIndexType>();
        }

        auto segment = remote_global_idxs.get_segment(set_id);

        // need to check if *it is actually the current global-id
        // since the global-id might not be registered as connected
        // to this rank
        auto it = std::lower_bound(segment.begin, segment.end, gid);
        return it != segment.end && *it == gid
                   ? static_cast<LocalIndexType>(
                         std::distance(remote_global_idxs.flat_begin, it))
                   : invalid_index<LocalIndexType>();
    };

    auto map_combined =
        [&, offset = partition->get_part_sizes()[rank]](const auto gid) {
            auto range_id = find_range(gid, partition);
            auto part_id = part_ids[range_id];

            if (part_id == rank) {
                return map_local(gid);
            } else {
                auto id = map_non_local(gid);
                return id == invalid_index<LocalIndexType>() ? id : id + offset;
            }
        };

    if (is == experimental::distributed::index_space::local) {
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];
            local_ids.get_data()[i] = gid == invalid_index<GlobalIndexType>()
                                          ? invalid_index<LocalIndexType>()
                                          : map_local(gid);
        }
    }
    if (is == experimental::distributed::index_space::non_local) {
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];
            local_ids.get_data()[i] = gid == invalid_index<GlobalIndexType>()
                                          ? invalid_index<LocalIndexType>()
                                          : map_non_local(gid);
        }
    }
    if (is == experimental::distributed::index_space::combined) {
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];
            local_ids.get_data()[i] = gid == invalid_index<GlobalIndexType>()
                                          ? invalid_index<LocalIndexType>()
                                          : map_combined(gid);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_MAP_TO_LOCAL);


template <typename LocalIndexType, typename GlobalIndexType>
void map_to_global(
    std::shared_ptr<const DefaultExecutor> exec,
    device_partition<const LocalIndexType, const GlobalIndexType> partition,
    device_segmented_array<const GlobalIndexType> remote_global_idxs,
    experimental::distributed::comm_index_type rank,
    const array<LocalIndexType>& local_idxs,
    experimental::distributed::index_space is,
    array<GlobalIndexType>& global_idxs)
{
    const auto& ranges_by_part = partition.ranges_by_part;
    auto local_ranges = ranges_by_part.get_segment(rank);

    global_idxs.resize_and_reset(local_idxs.get_size());

    auto local_size =
        static_cast<LocalIndexType>(partition.part_sizes_begin[rank]);
    size_type local_range_id = 0;
    auto map_local = [&](auto lid) {
        if (0 <= lid && lid < local_size) {
            local_range_id =
                find_local_range(lid, rank, partition, local_range_id);
            return map_to_global(lid, partition,
                                 local_ranges.begin[local_range_id]);
        } else {
            return invalid_index<GlobalIndexType>();
        }
    };

    auto remote_size = static_cast<LocalIndexType>(
        remote_global_idxs.flat_end - remote_global_idxs.flat_begin);
    auto map_non_local = [&](auto lid) {
        if (0 <= lid && lid < remote_size) {
            return remote_global_idxs.flat_begin[lid];
        } else {
            return invalid_index<GlobalIndexType>();
        }
    };

    auto map_combined = [&](auto lid) {
        if (lid < local_size) {
            return map_local(lid);
        } else {
            return map_non_local(lid - local_size);
        }
    };

    for (size_type i = 0; i < local_idxs.get_size(); ++i) {
        auto lid = local_idxs.get_const_data()[i];
        if (is == experimental::distributed::index_space::local) {
            global_idxs.get_data()[i] = map_local(lid);
        } else if (is == experimental::distributed::index_space::non_local) {
            global_idxs.get_data()[i] = map_non_local(lid);
        } else if (is == experimental::distributed::index_space::combined) {
            global_idxs.get_data()[i] = map_combined(lid);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_MAP_TO_GLOBAL);


}  // namespace index_map
}  // namespace reference
}  // namespace kernels
}  // namespace gko
