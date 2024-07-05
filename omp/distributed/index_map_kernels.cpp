// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"

#include <omp.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/base/segmented_array.hpp"
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
    array<LocalIndexType>& remote_local_idxs,
    array<GlobalIndexType>& remote_global_idxs, array<int64>& remote_sizes)
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
        return std::tie(get<0>(a), get<1>(a)) < std::tie(get<0>(b), get<1>(b));
    });

    // get only unique connections
    auto unique_end = std::unique(sort_it, sort_it + input_size,
                                  [](const auto& a, const auto& b) {
                                      return std::tie(get<0>(a), get<1>(a)) ==
                                             std::tie(get<0>(b), get<1>(b));
                                  });
    auto unique_size = std::distance(sort_it, unique_end);

    remote_global_idxs.resize_and_reset(unique_size);
    auto remote_global_idxs_ptr = remote_global_idxs.get_data();
    remote_local_idxs.resize_and_reset(unique_size);
    auto remote_local_idxs_ptr = remote_local_idxs.get_data();

    // store unique connections, also map global indices to local
#pragma omp parallel for
    for (size_type i = 0; i < unique_size; ++i) {
        remote_global_idxs_ptr[i] = recv_connections_ptr[i];
        remote_local_idxs_ptr[i] =
            map_to_local(recv_connections_ptr[i], part, range_ids[i]);
    }

    // compute number of connections per part-id
    vector<unsigned long long int> full_remote_sizes(part->get_num_parts(), 0,
                                                     exec);

#pragma omp parallel for
    for (size_type i = 0; i < unique_size; ++i) {
        // std::vector access with [] can count as function call, which
        // is not allowed in an atomic expression, thus getting the reference
        // before the atomic update.
        auto& size = full_remote_sizes[full_remote_part_ids[i]];
#pragma omp atomic
        size++;
    }
    auto num_neighbors =
        full_remote_sizes.size() -
        std::count(full_remote_sizes.begin(), full_remote_sizes.end(), 0);

    remote_sizes.resize_and_reset(num_neighbors);
    remote_part_ids.resize_and_reset(num_neighbors);
    {
        size_type idx = 0;
        for (size_type i = 0; i < full_remote_sizes.size(); ++i) {
            if (full_remote_sizes[i] > 0) {
                remote_part_ids.get_data()[idx] =
                    static_cast<comm_index_type>(i);
                remote_sizes.get_data()[idx] =
                    static_cast<int64>(full_remote_sizes[i]);
                idx++;
            }
        }
    }
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

    // can't extract functions to map global indices to local indices as for
    // the reference implementation, because it resulted in internal compiler
    // errors for intel 19.1.3
    size_type range_id = 0;
    if (is == experimental::distributed::index_space::local) {
#pragma omp parallel for firstprivate(range_id)
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];

            range_id = find_range(gid, partition, range_id);
            auto part_id = part_ids[range_id];

            local_ids.get_data()[i] = part_id == rank
                                          ? static_cast<LocalIndexType>(
                                                gid - range_bounds[range_id]) +
                                                range_starting_idxs[range_id]
                                          : invalid_index<LocalIndexType>();
        }
    }
    if (is == experimental::distributed::index_space::non_local) {
#pragma omp parallel for firstprivate(range_id)
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];

            range_id = find_range(gid, partition, range_id);
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
                local_ids.get_data()[i] = invalid_index<LocalIndexType>();
            } else {
                auto segment = remote_global_idxs.get_segment(set_id);

                // need to check if *it is actually the current global-id
                // since the global-id might not be registered as connected
                // to this rank
                auto it = std::lower_bound(segment.begin, segment.end, gid);
                local_ids.get_data()[i] =
                    it != segment.end && *it == gid
                        ? static_cast<LocalIndexType>(
                              std::distance(remote_global_idxs.flat_begin, it))
                        : invalid_index<LocalIndexType>();
            }
        }
    }
    if (is == experimental::distributed::index_space::combined) {
        auto offset = partition->get_part_sizes()[rank];
#pragma omp parallel for firstprivate(range_id) default(shared)
        for (size_type i = 0; i < global_ids.get_size(); ++i) {
            auto gid = global_ids.get_const_data()[i];
            range_id = find_range(gid, partition, range_id);
            auto part_id = part_ids[range_id];

            if (part_id == rank) {
                // same as is local
                local_ids.get_data()[i] =
                    part_id == rank ? static_cast<LocalIndexType>(
                                          gid - range_bounds[range_id]) +
                                          range_starting_idxs[range_id]
                                    : invalid_index<LocalIndexType>();
            } else {
                // same as is non_local, with additional offset
                auto set_id = std::distance(
                    remote_target_ids.get_const_data(),
                    std::lower_bound(remote_target_ids.get_const_data(),
                                     remote_target_ids.get_const_data() +
                                         remote_target_ids.get_size(),
                                     part_id));

                if (set_id == remote_target_ids.get_size()) {
                    local_ids.get_data()[i] = invalid_index<LocalIndexType>();
                } else {
                    auto segment = remote_global_idxs.get_segment(set_id);

                    auto it = std::lower_bound(segment.begin, segment.end, gid);
                    local_ids.get_data()[i] =
                        it != segment.end && *it == gid
                            ? static_cast<LocalIndexType>(
                                  std::distance(remote_global_idxs.flat_begin,
                                                it) +
                                  offset)
                            : invalid_index<LocalIndexType>();
                }
            }
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
    const array<LocalIndexType>& local_ids,
    experimental::distributed::index_space is,
    array<GlobalIndexType>& global_ids)
{
    const auto& ranges_by_part = partition.ranges_by_part;
    auto local_ranges = ranges_by_part.get_segment(rank);

    global_ids.resize_and_reset(local_ids.get_size());

    auto local_size =
        static_cast<LocalIndexType>(partition.part_sizes_begin[rank]);
    auto remote_size = static_cast<LocalIndexType>(
        remote_global_idxs.flat_end - remote_global_idxs.flat_begin);
    size_type local_range_id = 0;
    if (is == experimental::distributed::index_space::local) {
#pragma omp parallel for firstprivate(local_range_id)
        for (size_type i = 0; i < local_ids.get_size(); ++i) {
            auto lid = local_ids.get_const_data()[i];

            if (0 <= lid && lid < local_size) {
                local_range_id =
                    find_local_range(lid, rank, partition, local_range_id);
                global_ids.get_data()[i] = map_to_global(
                    lid, partition, local_ranges.begin[local_range_id]);
            } else {
                global_ids.get_data()[i] = invalid_index<GlobalIndexType>();
            }
        }
    }
    if (is == experimental::distributed::index_space::non_local) {
#pragma omp parallel for
        for (size_type i = 0; i < local_ids.get_size(); ++i) {
            auto lid = local_ids.get_const_data()[i];

            if (0 <= lid && lid < remote_size) {
                global_ids.get_data()[i] = remote_global_idxs.flat_begin[lid];
            } else {
                global_ids.get_data()[i] = invalid_index<GlobalIndexType>();
            }
        }
    }
    if (is == experimental::distributed::index_space::combined) {
#pragma omp parallel for firstprivate(local_range_id)
        for (size_type i = 0; i < local_ids.get_size(); ++i) {
            auto lid = local_ids.get_const_data()[i];

            if (0 <= lid && lid < local_size) {
                local_range_id =
                    find_local_range(lid, rank, partition, local_range_id);
                global_ids.get_data()[i] = map_to_global(
                    lid, partition, local_ranges.begin[local_range_id]);
            } else if (local_size <= lid && lid < local_size + remote_size) {
                global_ids.get_data()[i] =
                    remote_global_idxs.flat_begin[lid - local_size];
            } else {
                global_ids.get_data()[i] = invalid_index<GlobalIndexType>();
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_INDEX_MAP_MAP_TO_GLOBAL);


}  // namespace index_map
}  // namespace omp
}  // namespace kernels
}  // namespace gko
