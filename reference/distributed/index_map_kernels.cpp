// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"


#include <ginkgo/core/distributed/index_map.hpp>


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
    segmented_array<LocalIndexType>& remote_local_idxs,
    segmented_array<GlobalIndexType>& remote_global_idxs)
{
    using experimental::distributed::comm_index_type;
    auto part_ids = part->get_part_ids();

    std::vector<GlobalIndexType> unique_indices(recv_connections.get_size());
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

    auto flat_remote_global_idxs = array<GlobalIndexType>(exec, unique_size);
    auto flat_remote_local_idxs = array<LocalIndexType>(exec, unique_size);

    // store unique global indices
    std::copy_n(unique_indices.begin(), unique_size,
                flat_remote_global_idxs.get_data());

    // store and transform
    std::transform(unique_indices.begin(), unique_indices_end,
                   flat_remote_local_idxs.get_data(), [&](const auto idx) {
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
    std::vector<int64> full_remote_sizes(part->get_num_parts());
    for (size_type i = 0; i < full_part_ids.size(); ++i) {
        full_remote_sizes[full_part_ids[i]]++;
    }
    std::vector<int64> remote_sizes;
    for (auto size : full_remote_sizes) {
        if (size) {
            remote_sizes.push_back(size);
        }
    }
    GKO_ASSERT(remote_sizes.size() == unique_part_ids_size);

    remote_global_idxs = segmented_array<GlobalIndexType>(
        std::move(flat_remote_global_idxs), remote_sizes);
    remote_local_idxs = segmented_array<LocalIndexType>(
        std::move(flat_remote_local_idxs), remote_sizes);
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_BUILD_MAPPING);


template <typename LocalIndexType, typename GlobalIndexType>
void get_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    const array<experimental::distributed::comm_index_type>& remote_target_ids,
    device::segmented_array<const xstd::type_identity_t<GlobalIndexType>>
        remote_global_idxs,
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
    auto create_map_non_local = [&](const LocalIndexType offset) {
        return [&, offset](const auto gid) {
            auto range_id = find_range(gid, partition);
            auto part_id = part_ids[range_id];

            // can't do binary search on whole remote_target_idxs array,
            // since the array is first sorted by part-id and then by
            // global index. As a result, the array is not sorted wrt.
            // the global indexing. So find the part-id that corresponds
            // to the global index first
            auto set_id = std::distance(
                remote_target_ids.get_const_data(),
                std::lower_bound(remote_target_ids.get_const_data(),
                                 remote_target_ids.get_const_data() +
                                     remote_target_ids.get_size(),
                                 part_id));

            if (set_id == remote_target_ids.get_size()) {
                return invalid_index<LocalIndexType>();
            }

            // need to check if *it is actually the current global-id
            // since the global-id might not be registered as connected
            // to this rank
            auto it = std::lower_bound(
                remote_global_idxs.enumerate(set_id).begin(),
                remote_global_idxs.enumerate(set_id).end(), gid,
                [](const auto& a, const auto& b) { return a.value < b; });
            return it != remote_global_idxs.enumerate(set_id).end() &&
                           it->value == gid
                       ? offset + static_cast<LocalIndexType>(it->index)
                       : invalid_index<LocalIndexType>();
        };
    };
    auto map_non_local = create_map_non_local(0);

    auto combined_map_non_local =
        create_map_non_local(partition->get_part_size(rank));
    auto map_combined = [&](const auto gid) {
        auto range_id = find_range(gid, partition);
        auto part_id = part_ids[range_id];

        if (part_id == rank) {
            return map_local(gid);
        } else {
            return combined_map_non_local(gid);
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
