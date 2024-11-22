// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_ASSEMBLY_HELPERS_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_ASSEMBLY_HELPERS_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/range.hpp>


namespace gko {
namespace experimental {
namespace distributed {

template <typename LocalIndexType, typename GlobalIndexType>
class Partition;


/**
 * Assembles device_matrix_data entries owned by this MPI rank from other ranks
 * and communicates entries located on this MPI rank owned by other ranks to
 * their respective owners. This can be useful e.g. in a finite element code
 * where each rank assembles a local contribution to a global system matrix and
 * the global matrix has to be assembled by summing up the local contributions
 * on rank boundaries. The partition used is only relevant for row ownership.
 *
 * @param comm the communicator used to assemble the global matrix.
 * @param input the device_matrix_data structure.
 * @param partition the partition used to determine row owndership.
 *
 * @return the globally assembled device_matrix_data structure for this MPI
 * rank.
 */
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
device_matrix_data<ValueType, GlobalIndexType> add_non_local_entries(
    mpi::communicator comm,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    ptr_param<const Partition<LocalIndexType, GlobalIndexType>> partition);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_ASSEMBLY_HELPERS_HPP_
