// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_BASE_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_BASE_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/mpi.hpp>


namespace gko {
namespace experimental {
namespace distributed {


/**
 * A base class for distributed objects.
 *
 * This class stores and gives access to the used mpi::communicator object.
 *
 * @note The communicator is not changed on assignment.
 *
 * @ingroup distributed
 */
class DistributedBase {
public:
    virtual ~DistributedBase() = default;

    DistributedBase(const DistributedBase& other) = default;

    DistributedBase(DistributedBase&& other) = default;

    /**
     * Copy assignment that doesn't change the used mpi::communicator.
     * @return  unmodified *this
     */
    DistributedBase& operator=(const DistributedBase&) { return *this; }

    /**
     * Move assignment that doesn't change the used mpi::communicator.
     * @return  unmodified *this
     */
    DistributedBase& operator=(DistributedBase&&) noexcept { return *this; }

    /**
     * Access the used mpi::communicator.
     * @return  used mpi::communicator
     */
    mpi::communicator get_communicator() const { return comm_; }

protected:
    /**
     * Creates a new DistributedBase with the specified mpi::communicator.
     * @param comm  used mpi::communicator
     */
    explicit DistributedBase(mpi::communicator comm) : comm_{std::move(comm)} {}

private:
    mpi::communicator comm_;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_BASE_HPP_
