// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "reference/distributed/partition_helpers.hpp"


namespace gko {
namespace kernels {
namespace omp {
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
    using experimental::distributed::comm_index_type;
    using partition_type =
        experimental::distributed::Partition<LocalIndexType, GlobalIndexType>;
    auto part_ids = part->get_part_ids();
    const auto* range_bounds = part->get_range_bounds();
    const auto* range_starting_indices = part->get_range_starting_indices();
    const auto num_ranges = part->get_num_ranges();
    auto input_size = recv_connections.get_size();

    auto recv_connections_copy = recv_connections;
    auto recv_connections_ptr = recv_connections_copy.get_data();

    // precompute the range id and part id of each input element
    vector<size_type> range_ids(input_size, exec);
    vector<comm_index_type> full_remote_part_ids(input_size, exec);
    size_type range_id = 0;
#pragma omp parallel for firstprivate(range_id)
    for (size_type i = 0; i < input_size; ++i) {
        range_id = find_range(recv_connections_ptr[i], part, range_id);
        range_ids[i] = range_id;
        full_remote_part_ids[i] = part_ids[range_ids[i]];
    }

    // sort by part-id and recv_connection
    auto sort_it = detail::make_zip_iterator(
        full_remote_part_ids.begin(), recv_connections_ptr, range_ids.begin());
    std::sort(sort_it, sort_it + input_size, [](const auto& a, const auto& b) {
        return std::tie(std::get<0>(a), std::get<1>(a)) <
               std::tie(std::get<0>(b), std::get<1>(b));
    });

    // get only unique connections
    auto unique_end = std::unique(
        sort_it, sort_it + input_size, [](const auto& a, const auto& b) {
            return std::tie(std::get<0>(a), std::get<1>(a)) ==
                   std::tie(std::get<0>(b), std::get<1>(b));
        });
    auto unique_size = std::distance(sort_it, unique_end);

    auto flat_remote_global_idxs = array<GlobalIndexType>(exec, unique_size);
    auto flat_remote_global_idxs_ptr = flat_remote_global_idxs.get_data();
    auto flat_remote_local_idxs = array<LocalIndexType>(exec, unique_size);
    auto flat_remote_local_idxs_ptr = flat_remote_local_idxs.get_data();

    // store unique connections, also map global indices to local
#pragma omp parallel for
    for (size_type i = 0; i < unique_size; ++i) {
        flat_remote_global_idxs_ptr[i] = recv_connections_ptr[i];
        flat_remote_local_idxs_ptr[i] =
            map_to_local(recv_connections_ptr[i], part, range_ids[i]);
    }

    // compute number of connections per part-id
    vector<unsigned long long int> full_remote_sizes(part->get_num_parts(), 0,
                                                     exec);

#pragma omp parallel for
    for (size_type i = 0; i < unique_size; ++i) {
#pragma omp atomic
        full_remote_sizes[full_remote_part_ids[i]]++;
    }
    auto num_neighbors =
        full_remote_sizes.size() -
        std::count(full_remote_sizes.begin(), full_remote_sizes.end(), 0);

    remote_part_ids.resize_and_reset(num_neighbors);
    std::vector<unsigned long long int> remote_sizes(num_neighbors);
    {
        size_type idx = 0;
        for (size_type i = 0; i < full_remote_sizes.size(); ++i) {
            if (full_remote_sizes[i] > 0) {
                remote_part_ids.get_data()[idx] =
                    static_cast<comm_index_type>(i);
                remote_sizes[idx] = full_remote_sizes[i];
                idx++;
            }
        }
    }

    remote_global_idxs = collection::array<GlobalIndexType>(
        std::move(flat_remote_global_idxs), remote_sizes);
    remote_local_idxs = collection::array<LocalIndexType>(
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
    const collection::array<GlobalIndexType>& remote_global_idxs,
    experimental::distributed::comm_index_type rank,
    const array<GlobalIndexType>& global_ids,
    experimental::distributed::index_space is, array<LocalIndexType>& local_ids)
{
    auto part_ids = partition->get_part_ids();
    auto range_bounds = partition->get_range_bounds();
    auto range_starting_idxs = partition->get_range_starting_indices();

    local_ids.resize_and_reset(global_ids.get_size());

    auto map_local = [&](const auto gid, const auto range_id_hint) {
        auto range_id = find_range(gid, partition, range_id_hint);
        auto part_id = part_ids[range_id];

        return std::make_pair(
            part_id == rank
                ? static_cast<LocalIndexType>(gid - range_bounds[range_id]) +
                      range_starting_idxs[range_id]
                : invalid_index<LocalIndexType>(),
            range_id);
    };

    auto create_map_non_local = [&](const LocalIndexType offset) {
        return [&, offset](const auto gid, const auto range_id_hint) {
            auto range_id = find_range(gid, partition, range_id_hint);
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
                return std::make_pair(invalid_index<LocalIndexType>(),
                                      range_id);
            }

            auto remote_global_begin =
                remote_global_idxs[set_id].get_const_data();
            auto remote_global_end =
                remote_global_begin + remote_global_idxs[set_id].get_size();

            // need to check if *it is actually the current global-id
            // since the global-id might not be registered as connected
            // to this rank
            auto it =
                std::lower_bound(remote_global_begin, remote_global_end, gid);
            return std::make_pair(
                it != remote_global_end && *it == gid
                    ? static_cast<LocalIndexType>(
                          std::distance(
                              remote_global_idxs.get_flat().get_const_data(),
                              it) +
                          offset)
                    : invalid_index<LocalIndexType>(),
                range_id);
        };
    };
    auto map_non_local = create_map_non_local(0);

    auto combined_map_non_local =
        create_map_non_local(partition->get_part_size(rank));
    auto map_combined = [&](const auto gid, const auto range_id_hint) {
        auto range_id = find_range(gid, partition, range_id_hint);
        auto part_id = part_ids[range_id];

        if (part_id == rank) {
            return map_local(gid, range_id_hint);
        } else {
            return combined_map_non_local(gid, range_id_hint);
        }
    };

    size_type range_id = 0;
    if (is == experimental::distributed::index_space::local) {
#pragma omp parallel for firstprivate(range_id)
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];

            auto result = map_local(gid, range_id);
            range_id = result.second;

            local_ids.get_data()[i] = result.first;
        }
    }
    if (is == experimental::distributed::index_space::non_local) {
#pragma omp parallel for firstprivate(range_id)
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];
            auto result = map_non_local(gid, range_id);
            range_id = result.second;

            local_ids.get_data()[i] = result.first;
        }
    }
    if (is == experimental::distributed::index_space::combined) {
#pragma omp parallel for firstprivate(range_id)
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];
            auto result = map_combined(gid, range_id);
            range_id = result.second;

            local_ids.get_data()[i] = result.first;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_GET_LOCAL_FROM_GLOBAL_ARRAY);


}  // namespace index_map
}  // namespace omp
}  // namespace kernels
}  // namespace gko
