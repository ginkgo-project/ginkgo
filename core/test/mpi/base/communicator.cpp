// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
