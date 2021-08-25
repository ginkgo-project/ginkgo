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

#include "gtest-mpi-listener.hpp"
#include "gtest-mpi-main.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


class Communicator : public ::testing::Test {
protected:
    Communicator() : comm(MPI_COMM_WORLD) {}

    void SetUp()
    {
        rank = gko::mpi::get_my_rank(comm);
        ASSERT_EQ(gko::mpi::get_num_ranks(comm), 8);
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
    auto comm_world = gko::mpi::communicator(MPI_COMM_WORLD);
    auto copy = comm_world;

    EXPECT_EQ(comm_world.compare(MPI_COMM_WORLD), true);
    EXPECT_EQ(copy.compare(MPI_COMM_WORLD), true);
}


TEST_F(Communicator, CommunicatorCanBeCopyConstructed)
{
    auto comm_world = gko::mpi::communicator(MPI_COMM_WORLD);
    auto copy = gko::mpi::communicator(comm_world);

    EXPECT_EQ(comm_world.compare(MPI_COMM_WORLD), true);
    EXPECT_EQ(copy.compare(MPI_COMM_WORLD), true);
}


TEST_F(Communicator, CommunicatorCanBeMoved)
{
    int size;
    auto comm_world = gko::mpi::communicator(MPI_COMM_WORLD);

    auto moved = std::move(comm_world);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    EXPECT_EQ(comm_world.get(), MPI_COMM_NULL);
    EXPECT_EQ(comm_world.size(), 0);
    EXPECT_EQ(moved.compare(MPI_COMM_WORLD), true);
    EXPECT_EQ(moved.size(), size);
}


TEST_F(Communicator, CommunicatorCanBeMoveConstructed)
{
    int size;
    auto comm_world = gko::mpi::communicator(MPI_COMM_WORLD);

    auto moved = gko::mpi::communicator(std::move(comm_world));

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    EXPECT_EQ(comm_world.get(), MPI_COMM_NULL);
    EXPECT_EQ(comm_world.size(), 0);
    EXPECT_EQ(moved.compare(MPI_COMM_WORLD), true);
    EXPECT_EQ(moved.size(), size);
}


TEST_F(Communicator, CommKnowsItsSize)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);

    EXPECT_EQ(comm.size(), size);
}


TEST_F(Communicator, KnowsItsSize)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    EXPECT_EQ(gko::mpi::get_num_ranks(MPI_COMM_WORLD), size);
}


TEST_F(Communicator, CommKnowsItsRank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);

    EXPECT_EQ(comm.rank(), rank);
}


TEST_F(Communicator, CommKnowsItsLocalRank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);

    // Expect local rank to be same as rank when on one node
    EXPECT_EQ(comm.local_rank(), rank);
}


TEST_F(Communicator, KnowsItsRanks)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    EXPECT_EQ(rank, gko::mpi::get_my_rank(MPI_COMM_WORLD));
}


TEST_F(Communicator, KnowsItsDefaultCommunicator)
{
    auto comm_world = gko::mpi::communicator(MPI_COMM_WORLD);
    ASSERT_TRUE(comm_world == comm);
}


TEST_F(Communicator, KnowsNumRanks)
{
    EXPECT_EQ(gko::mpi::get_num_ranks(comm), 8);
}


TEST_F(Communicator, CanSetCustomCommunicator)
{
    auto world_rank = gko::mpi::get_my_rank(comm);
    auto world_size = gko::mpi::get_num_ranks(comm);
    auto color = world_rank / 4;

    auto row_comm = gko::mpi::communicator(comm.get(), color, world_rank);
    for (auto i = 0; i < world_size; ++i) {
        EXPECT_LT(gko::mpi::get_my_rank(row_comm.get()), 4);
    }
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
