/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
#include <ginkgo/core/base/range.hpp>


namespace {


class Communicator : public ::testing::Test {
protected:
    Communicator() : comm(MPI_COMM_WORLD) {}

    void SetUp()
    {
        rank = comm.rank();
        ASSERT_EQ(comm.size(), 8);
    }

    gko::mpi::communicator comm;
    int rank;
};


TEST_F(Communicator, DefaultCommIsInvalid)
{
    auto comm = gko::mpi::communicator();

    EXPECT_EQ(comm.get(), MPI_COMM_NULL);
}


TEST_F(Communicator, CanCreateWorld)
{
    auto comm = gko::mpi::communicator::create_world();

    EXPECT_EQ(comm->compare(MPI_COMM_WORLD), true);
}


TEST_F(Communicator, KnowsItsCommunicator)
{
    MPI_Comm dup;
    MPI_Comm_dup(MPI_COMM_WORLD, &dup);
    auto comm_world = gko::mpi::communicator(dup);

    EXPECT_EQ(comm_world.compare(dup), true);
}


TEST_F(Communicator, CommunicatorCanBeCopied)
{
    auto copy = comm;

    EXPECT_EQ(comm.compare(MPI_COMM_WORLD), true);
    EXPECT_EQ(copy.compare(MPI_COMM_WORLD), true);
}


TEST_F(Communicator, CommunicatorCanBeCopyConstructed)
{
    auto copy = gko::mpi::communicator(comm);

    EXPECT_EQ(comm.compare(MPI_COMM_WORLD), true);
    EXPECT_EQ(copy.compare(MPI_COMM_WORLD), true);
}


TEST_F(Communicator, CommunicatorCanBeMoved)
{
    int size;
    auto comm_world = gko::mpi::communicator::create_world();
    auto moved = std::move(comm_world);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    EXPECT_EQ(comm_world, nullptr);
    EXPECT_EQ(moved->compare(MPI_COMM_WORLD), true);
    EXPECT_EQ(moved->size(), size);
}


TEST_F(Communicator, CommunicatorCanBeMoveConstructed)
{
    int size;
    auto comm_world = gko::mpi::communicator::create_world();
    auto moved = gko::mpi::communicator(std::move(*comm_world.get()));

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    EXPECT_EQ(comm_world->get(), MPI_COMM_NULL);
    EXPECT_EQ(comm_world->size(), 0);
    EXPECT_EQ(moved.compare(MPI_COMM_WORLD), true);
    EXPECT_EQ(moved.size(), size);
}


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
    EXPECT_EQ(comm.local_rank(), rank);
}


TEST_F(Communicator, KnowsItsRanks)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    EXPECT_EQ(comm.rank(), rank);
}


TEST_F(Communicator, CanSetCustomCommunicator)
{
    auto world_rank = comm.rank();
    auto world_size = comm.size();
    auto color = world_rank / 4;

    auto row_comm = gko::mpi::communicator(comm.get(), color, world_rank);
    for (auto i = 0; i < world_size; ++i) {
        EXPECT_LT(row_comm.rank(), 4);
    }
}


}  // namespace
