// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstring>
#include <vector>

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/mpi.hpp>


namespace {


/**
 * Returns the rank among the ranks of comm on the same node, determined from
 * the processor names instead of MPI_Comm_split_type.
 */
int node_local_rank_from_names(MPI_Comm comm)
{
    int rank;
    int size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::vector<char> names(size * MPI_MAX_PROCESSOR_NAME);
    auto name = [&](int i) {
        return names.data() + i * MPI_MAX_PROCESSOR_NAME;
    };
    int length;
    MPI_Get_processor_name(name(rank), &length);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, names.data(),
                  MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm);
    int local_rank = 0;
    for (int i = 0; i < rank; ++i) {
        if (std::strcmp(name(i), name(rank)) == 0) {
            ++local_rank;
        }
    }
    return local_rank;
}


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
    EXPECT_EQ(comm.node_local_rank(),
              node_local_rank_from_names(MPI_COMM_WORLD));
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


TEST_F(Communicator, SplitCommunicatorsSetForceHostBuffer)
{
    gko::experimental::mpi::communicator host_comm(MPI_COMM_WORLD, true);

    gko::experimental::mpi::communicator from_comm(host_comm, rank / 4, rank);
    gko::experimental::mpi::communicator from_mpi_comm(MPI_COMM_WORLD, rank / 4,
                                                       rank);

    EXPECT_TRUE(from_comm.force_host_buffer());
    EXPECT_FALSE(from_mpi_comm.force_host_buffer());
}


}  // namespace
