// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/collective_communicator.hpp"

#include <mpi.h>

#include <ginkgo/core/distributed/dense_communicator.hpp>


namespace gko {
namespace experimental {
namespace mpi {


CollectiveCommunicator::CollectiveCommunicator(communicator base)
    : base_(std::move(base))
{}


const communicator& CollectiveCommunicator::get_base_communicator() const
{
    return base_;
}


request CollectiveCommunicator::i_all_to_all_v(
    std::shared_ptr<const Executor> exec, const void* send_buffer,
    MPI_Datatype send_type, void* recv_buffer, MPI_Datatype recv_type) const
{
    return this->i_all_to_all_v_impl(std::move(exec), send_buffer, send_type,
                                     recv_buffer, recv_type);
}


std::shared_ptr<CollectiveCommunicator>
detail::create_default_collective_communicator(communicator base)
{
    return std::make_shared<DenseCommunicator>(base);
}


}  // namespace mpi
}  // namespace experimental
}  // namespace gko
