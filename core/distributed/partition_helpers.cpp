// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/partition_helpers.hpp>


#include <numeric>


#include <ginkgo/core/distributed/partition.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/distributed/partition_helpers_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace components {
namespace {


GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // namespace
}  // namespace components


namespace partition_helpers {
namespace {


GKO_REGISTER_OPERATION(sort_by_range_start,
                       partition_helpers::sort_by_range_start);
GKO_REGISTER_OPERATION(check_consecutive_ranges,
                       partition_helpers::check_consecutive_ranges);
GKO_REGISTER_OPERATION(compress_ranges, partition_helpers::compress_ranges);


}  // namespace
}  // namespace partition_helpers


template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
build_partition_from_local_range(std::shared_ptr<const Executor> exec,
                                 mpi::communicator comm, span local_range)
{
    std::array<GlobalIndexType, 2> range{
        static_cast<GlobalIndexType>(local_range.begin),
        static_cast<GlobalIndexType>(local_range.end)};

    // make all range_start_ends available on each rank
    // note: not all combination of MPI + GPU library seem to support
    // mixing host and device buffers, e.g. OpenMPI 4.0.5 and Rocm 4.0
    auto mpi_exec = exec->get_master();
    array<GlobalIndexType> ranges_start_end(mpi_exec, comm.size() * 2);
    ranges_start_end.fill(invalid_index<GlobalIndexType>());
    comm.all_gather(mpi_exec, range.data(), 2, ranges_start_end.get_data(), 2);
    ranges_start_end.set_executor(exec);

    // make_sort_by_range_start
    array<comm_index_type> part_ids(exec, comm.size());
    exec->run(components::make_fill_seq_array(part_ids.get_data(),
                                              part_ids.get_size()));
    exec->run(partition_helpers::make_sort_by_range_start(ranges_start_end,
                                                          part_ids));

    // check for consistency
    bool consecutive_ranges = false;
    exec->run(partition_helpers::make_check_consecutive_ranges(
        ranges_start_end, consecutive_ranges));
    if (!consecutive_ranges) {
        GKO_INVALID_STATE("The partition contains gaps.");
    }

    // join (now consecutive) starts and ends into combined array
    array<GlobalIndexType> ranges(exec, comm.size() + 1);
    exec->run(
        partition_helpers::make_compress_ranges(ranges_start_end, ranges));

    return Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
        exec, ranges, part_ids);
}

#define GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_RANGE(_local_type,          \
                                                     _global_type)         \
    std::unique_ptr<Partition<_local_type, _global_type>>                  \
    build_partition_from_local_range(std::shared_ptr<const Executor> exec, \
                                     mpi::communicator comm, span local_range)
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_RANGE);


template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
build_partition_from_local_size(std::shared_ptr<const Executor> exec,
                                mpi::communicator comm, size_type local_size)
{
    auto local_size_gi = static_cast<GlobalIndexType>(local_size);
    array<GlobalIndexType> sizes(exec->get_master(), comm.size());
    comm.all_gather(exec, &local_size_gi, 1, sizes.get_data(), 1);

    array<GlobalIndexType> offsets(exec->get_master(), comm.size() + 1);
    offsets.get_data()[0] = 0;
    std::partial_sum(sizes.get_data(), sizes.get_data() + comm.size(),
                     offsets.get_data() + 1);

    return Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
        exec, offsets);
}

#define GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_SIZE(_local_type, _global_type) \
    std::unique_ptr<Partition<_local_type, _global_type>>                      \
    build_partition_from_local_size(std::shared_ptr<const Executor> exec,      \
                                    mpi::communicator comm,                    \
                                    size_type local_range)
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_SIZE);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
