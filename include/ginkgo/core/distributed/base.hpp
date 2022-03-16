/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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

    void set_commuinicator(mpi::communicator new_comm)
    {
        comm_ = std::move(new_comm);
    }

private:
    mpi::communicator comm_;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_BASE_HPP_
