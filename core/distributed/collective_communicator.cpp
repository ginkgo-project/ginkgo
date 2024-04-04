// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/collective_communicator.hpp"


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


}  // namespace mpi
}  // namespace experimental
}  // namespace gko
