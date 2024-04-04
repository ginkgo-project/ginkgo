// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"


#include <ginkgo/core/distributed/index_map.hpp>


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
    using partition_type =
        experimental::distributed::Partition<LocalIndexType, GlobalIndexType>;
    auto part_ids = part->get_part_ids();

    std::vector<GlobalIndexType> unique_indices(recv_connections.get_size());
    std::copy_n(recv_connections.get_const_data(), recv_connections.get_size(),
                unique_indices.begin());

    auto find_range = [](GlobalIndexType idx, const partition_type* partition,
                         size_type hint) {
        auto range_bounds = partition->get_range_bounds();
        auto num_ranges = partition->get_num_ranges();
        if (range_bounds[hint] <= idx && idx < range_bounds[hint + 1]) {
            return hint;
        } else {
            auto it = std::upper_bound(range_bounds + 1,
                                       range_bounds + num_ranges + 1, idx);
            return static_cast<size_type>(std::distance(range_bounds + 1, it));
        }
    };

    auto map_to_local = [](GlobalIndexType idx, const partition_type* partition,
                           size_type range_id) {
        auto range_bounds = partition->get_range_bounds();
        auto range_starting_indices = partition->get_range_starting_indices();
        return static_cast<LocalIndexType>(idx - range_bounds[range_id]) +
               range_starting_indices[range_id];
    };

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
    std::vector<comm_index_type> full_part_ids(unique_size);
    std::transform(unique_indices.begin(), unique_indices_end,
                   full_part_ids.begin(),
                   [&](const auto idx) { return find_part(idx); });

    std::vector<comm_index_type> unique_part_ids(full_part_ids);
    auto unique_part_ids_end =
        std::unique(unique_part_ids.begin(), unique_part_ids.end());

    auto unique_part_ids_size =
        std::distance(unique_part_ids.begin(), unique_part_ids_end);
    remote_part_ids.resize_and_reset(unique_part_ids_size);
    std::copy(unique_part_ids.begin(), unique_part_ids_end,
              remote_part_ids.get_data());

    // get recv size per part
    std::vector<size_type> full_remote_sizes(part->get_num_parts());
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
void get_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    const array<experimental::distributed::comm_index_type>& remote_targed_ids,
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
        auto set_id = std::distance(
            remote_targed_ids.get_const_data(),
            std::lower_bound(remote_targed_ids.get_const_data(),
                             remote_targed_ids.get_const_data() +
                                 remote_targed_ids.get_size(),
                             part_id));

        if (set_id == remote_targed_ids.get_size()) {
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

            local_ids.get_data()[i] = map_local(gid);
        }
    }
    if (is == experimental::distributed::index_space::non_local) {
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];
            local_ids.get_data()[i] = map_non_local(gid);
        }
    }
    if (is == experimental::distributed::index_space::combined) {
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];
            local_ids.get_data()[i] = map_combined(gid);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_GET_LOCAL_FROM_GLOBAL_ARRAY);

}  // namespace index_map
}  // namespace reference
}  // namespace kernels
}  // namespace gko
