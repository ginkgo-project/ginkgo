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


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
device_matrix_data<ValueType, GlobalIndexType> assemble(
    mpi::communicator comm,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    ptr_param<const Partition<LocalIndexType, GlobalIndexType>> partition);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_ASSEMBLY_HELPERS_HPP_
