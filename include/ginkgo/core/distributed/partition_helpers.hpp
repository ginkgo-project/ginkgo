// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_PARTITION_HELPERS_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_PARTITION_HELPERS_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/range.hpp>


namespace gko {
namespace experimental {
namespace distributed {

template <typename LocalIndexType, typename GlobalIndexType>
class Partition;


/**
 * Builds a partition from a local range.
 *
 * @param exec  the Executor on which the partition should be built.
 * @param comm  the communicator used to determine the global partition.
 * @param local_range the start and end indices of the local range.
 *
 * @warning  This throws, if the resulting partition would contain gaps.
 *           That means that for a partition of size `n` every local range `r_i
 *           = [s_i, e_i)` either `s_i != 0` and another local range `r_j =
 *           [s_j, e_j = s_i)` exists, or `e_i != n` and another local range
 *           `r_j = [s_j = e_i, e_j)` exists.
 *
 * @return a Partition where each range has the individual local_start
 *         and local_ends.
 */
template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
build_partition_from_local_range(std::shared_ptr<const Executor> exec,
                                 mpi::communicator comm, span local_range);


/**
 * Builds a partition from a local size.
 *
 * @param exec  the Executor on which the partition should be built.
 * @param comm  the communicator used to determine the global partition.
 * @param local_range the number of the locally owned indices
 *
 * @return a Partition where each range has the specified local size. More
 *         specifically, if this is called on process i with local_size `s_i`,
 *         then the range `i` has size `s_i`, and range `r_i = [start, start +
 *         s_i)`, where `start = sum_j^(i-1) s_j`.
 */
template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
build_partition_from_local_size(std::shared_ptr<const Executor> exec,
                                mpi::communicator comm, size_type local_size);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_PARTITION_HELPERS_HPP_
