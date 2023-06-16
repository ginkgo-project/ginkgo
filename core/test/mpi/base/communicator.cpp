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

#include <mpi.h>


#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/mpi.hpp>


namespace {


class Communicator : public ::testing::Test {
protected:
    Communicator() : comm(MPI_COMM_WORLD) {}

    void SetUp()
    {
        rank = comm.rank();
        ASSERT_EQ(comm.size(), 8);
    }

    gko::experimental::mpi::communicator comm;
    int rank;
};


TEST_F(Communicator, CommKnowsItsSize)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    EXPECT_EQ(comm.size(), size);
}


TEST_F(Communicator, CommKnowsItsRank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    EXPECT_EQ(comm.rank(), rank);
}


TEST_F(Communicator, CommKnowsItsLocalRank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Expect local rank to be same as rank when on one node
    EXPECT_EQ(comm.node_local_rank(), rank);
}


TEST_F(Communicator, CommunicatorCanBeCopyConstructed)
{
    gko::experimental::mpi::communicator copy(comm);

    EXPECT_TRUE(copy == comm);
}


TEST_F(Communicator, CommunicatorCanBeCopyAssigned)
{
    gko::experimental::mpi::communicator copy = comm;

    EXPECT_TRUE(copy == comm);
}


TEST_F(Communicator, CommunicatorCanBeMoveConstructed)
{
    gko::experimental::mpi::communicator comm2(MPI_COMM_WORLD);
    gko::experimental::mpi::communicator copy(std::move(comm2));

    EXPECT_TRUE(copy == comm);
}


TEST_F(Communicator, CommunicatorCanBeMoveAssigned)
{
    gko::experimental::mpi::communicator comm2(MPI_COMM_WORLD);
    gko::experimental::mpi::communicator copy(MPI_COMM_NULL);
    copy = std::move(comm2);

    EXPECT_TRUE(copy == comm);
}


TEST_F(Communicator, CommunicatorCanBeSynchronized)
{
    ASSERT_NO_THROW(comm.synchronize());
}


TEST_F(Communicator, CanSetCustomCommunicator)
{
    auto world_rank = comm.rank();
    auto world_size = comm.size();
    auto color = world_rank / 4;

    auto row_comm =
        gko::experimental::mpi::communicator(comm.get(), color, world_rank);
    for (auto i = 0; i < world_size; ++i) {
        EXPECT_LT(row_comm.rank(), 4);
    }
}


}  // namespace
