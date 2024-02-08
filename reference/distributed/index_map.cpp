// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"


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
    collection::array<LocalIndexType>& remote_local_idxs,
    collection::array<GlobalIndexType>& remote_global_idxs)
{
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
    std::vector<experimental::distributed::comm_index_type> full_part_ids(
        unique_size);
    std::transform(unique_indices.begin(), unique_indices_end,
                   full_part_ids.begin(),
                   [&](const auto idx) { return find_part(idx); });

    std::vector unique_part_ids(full_part_ids);
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
    std::vector<size_type> remote_sizes;
    for (auto size : full_remote_sizes) {
        if (size) {
            remote_sizes.push_back(size);
        }
    }
    GKO_ASSERT(remote_sizes.size() == unique_part_ids_size);

    remote_global_idxs = collection::array<GlobalIndexType>(
        std::move(flat_remote_global_idxs), remote_sizes);
    remote_local_idxs = collection::array<LocalIndexType>(
        std::move(flat_remote_local_idxs), remote_sizes);
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_BUILD_MAPPING);


}  // namespace index_map
}  // namespace reference
}  // namespace kernels
}  // namespace gko
