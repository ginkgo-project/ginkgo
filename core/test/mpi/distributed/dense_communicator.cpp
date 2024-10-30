// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/distributed/dense_communicator.hpp>

#include "core/test/utils/assertions.hpp"

using gko::experimental::mpi::comm_index_type;

class DenseCommunicator : public ::testing::Test {
protected:
    using part_type = gko::experimental::distributed::Partition<int, long>;
    using map_type = gko::experimental::distributed::index_map<int, long>;

    void SetUp() override { ASSERT_EQ(comm.size(), 6); }

    gko::experimental::mpi::DenseCommunicator create_default_comm()
    {
        auto part = gko::share(part_type::build_from_global_size_uniform(
            ref, comm.size(), comm.size() * 3));
        gko::array<long> recv_connections[] = {{ref, {3, 5, 10, 11}},
                                               {ref, {0, 1, 7, 12, 13}},
                                               {ref, {3, 4, 17}},
                                               {ref, {1, 2, 12, 14}},
                                               {ref, {4, 5, 9, 10, 16, 15}},
                                               {ref, {8, 12, 13, 14}}};
        auto imap = map_type{ref, part, comm.rank(), recv_connections[rank]};


        return {comm, imap};
    }

    std::shared_ptr<gko::Executor> ref = gko::ReferenceExecutor::create();
    gko::experimental::mpi::communicator comm = MPI_COMM_WORLD;
    int rank = comm.rank();
};


TEST_F(DenseCommunicator, CanDefaultConstruct)
{
    gko::experimental::mpi::DenseCommunicator dcomm{comm};

    ASSERT_EQ(dcomm.get_base_communicator(), comm);
    ASSERT_EQ(dcomm.get_send_size(), 0);
    ASSERT_EQ(dcomm.get_recv_size(), 0);
}


TEST_F(DenseCommunicator, CanConstructFromIndexMap)
{
    auto part = gko::share(part_type::build_from_global_size_uniform(
        ref, comm.size(), comm.size() * 3));
    gko::array<long> recv_connections[] = {{ref, {3, 5, 10, 11}},
                                           {ref, {0, 1, 7, 12, 13}},
                                           {ref, {3, 4, 17}},
                                           {ref, {1, 2, 12, 14}},
                                           {ref, {4, 5, 9, 10, 16, 15}},
                                           {ref, {8, 12, 13, 14}}};
    auto imap = map_type{ref, part, comm.rank(), recv_connections[rank]};

    gko::experimental::mpi::DenseCommunicator dcomm{comm, imap};

    std::array<gko::size_type, 6> send_sizes = {4, 6, 2, 4, 7, 3};
    ASSERT_EQ(dcomm.get_recv_size(), recv_connections[rank].get_size());
    ASSERT_EQ(dcomm.get_send_size(), send_sizes[rank]);
}


TEST_F(DenseCommunicator, CanConstructFromEmptyIndexMap)
{
    auto imap = map_type{ref};

    gko::experimental::mpi::DenseCommunicator dcomm{comm, imap};

    ASSERT_EQ(dcomm.get_recv_size(), 0);
    ASSERT_EQ(dcomm.get_send_size(), 0);
}


TEST_F(DenseCommunicator, CanConstructFromIndexMapWithoutConnection)
{
    auto part = gko::share(part_type::build_from_global_size_uniform(
        ref, comm.size(), comm.size() * 3));
    auto imap = map_type{ref, part, comm.rank(), {ref, 0}};

    gko::experimental::mpi::DenseCommunicator dcomm{comm, imap};

    ASSERT_EQ(dcomm.get_recv_size(), 0);
    ASSERT_EQ(dcomm.get_send_size(), 0);
}


TEST_F(DenseCommunicator, CanTestEquality)
{
    auto comm_a = create_default_comm();
    auto comm_b = create_default_comm();

    ASSERT_EQ(comm_a, comm_b);
}


TEST_F(DenseCommunicator, CanTestInequality)
{
    auto comm_a = create_default_comm();
    auto comm_b = gko::experimental::mpi::DenseCommunicator(comm);

    ASSERT_NE(comm_a, comm_b);
}


TEST_F(DenseCommunicator, CanCopyConstruct)
{
    auto dcomm = create_default_comm();

    auto copy(dcomm);

    ASSERT_TRUE(copy == dcomm);
}


TEST_F(DenseCommunicator, CanCopyAssign)
{
    auto dcomm = create_default_comm();
    gko::experimental::mpi::DenseCommunicator copy{comm};

    copy = dcomm;

    ASSERT_TRUE(copy == dcomm);
}


TEST_F(DenseCommunicator, CanMoveConstruct)
{
    auto dcomm = create_default_comm();
    auto moved_from = dcomm;
    auto empty_comm = gko::experimental::mpi::DenseCommunicator{MPI_COMM_NULL};

    auto moved(std::move(moved_from));

    ASSERT_TRUE(moved == dcomm);
    ASSERT_TRUE(moved_from == empty_comm);
}


TEST_F(DenseCommunicator, CanMoveAssign)
{
    auto dcomm = create_default_comm();
    auto moved_from = dcomm;
    auto empty_comm = gko::experimental::mpi::DenseCommunicator{MPI_COMM_NULL};
    gko::experimental::mpi::DenseCommunicator moved{comm};

    moved = std::move(moved_from);

    ASSERT_TRUE(moved == dcomm);
    ASSERT_TRUE(moved_from == empty_comm);
}


TEST_F(DenseCommunicator, CanCommunicateIalltoall)
{
    auto part = gko::share(part_type::build_from_global_size_uniform(
        ref, comm.size(), comm.size() * 3));
    gko::array<long> recv_connections[] = {{ref, {3, 5, 10, 11}},
                                           {ref, {0, 1, 7, 12, 13}},
                                           {ref, {3, 4, 17}},
                                           {ref, {1, 2, 12, 14}},
                                           {ref, {4, 5, 9, 10, 16, 15}},
                                           {ref, {8, 12, 13, 14}}};
    auto imap = map_type{ref, part, comm.rank(), recv_connections[rank]};
    gko::experimental::mpi::DenseCommunicator dcomm{comm, imap};
    gko::array<long> recv_buffer{ref, recv_connections[rank].get_size()};
    gko::array<long> send_buffers[] = {{ref, {0, 1, 1, 2}},
                                       {ref, {3, 5, 3, 4, 4, 5}},
                                       {ref, {7, 8}},
                                       {ref, {10, 11, 9, 10}},
                                       {ref, {12, 13, 12, 14, 12, 13, 14}},
                                       {ref, {17, 16, 15}}};

    auto req = dcomm.i_all_to_all_v(ref, send_buffers[rank].get_const_data(),
                                    recv_buffer.get_data());
    req.wait();

    GKO_ASSERT_ARRAY_EQ(recv_buffer, recv_connections[rank]);
}


TEST_F(DenseCommunicator, CanCommunicateIalltoallWhenEmpty)
{
    gko::experimental::mpi::DenseCommunicator dcomm{comm};

    auto req = dcomm.i_all_to_all_v(ref, static_cast<int*>(nullptr),
                                    static_cast<int*>(nullptr));
    req.wait();
}


TEST_F(DenseCommunicator, CanCreateInverse)
{
    auto dcomm = create_default_comm();

    auto inverse = dcomm.create_inverse();

    ASSERT_EQ(inverse->get_recv_size(), dcomm.get_send_size());
    ASSERT_EQ(inverse->get_send_size(), dcomm.get_recv_size());
}


TEST_F(DenseCommunicator, CanCommunicateRoundTrip)
{
    auto part = gko::share(part_type::build_from_global_size_uniform(
        ref, comm.size(), comm.size() * 3));
    gko::array<long> recv_connections[] = {{ref, {3, 5, 10, 11}},
                                           {ref, {0, 1, 7, 12, 13}},
                                           {ref, {3, 4, 17}},
                                           {ref, {1, 2, 12, 14}},
                                           {ref, {4, 5, 9, 10, 16, 15}},
                                           {ref, {8, 12, 13, 14}}};
    auto imap = map_type{ref, part, comm.rank(), recv_connections[rank]};
    gko::experimental::mpi::DenseCommunicator dcomm{comm, imap};
    auto inverse = dcomm.create_inverse();
    gko::array<long> send_buffers[] = {{ref, {1, 2, 3, 4}},
                                       {ref, {5, 6, 7, 8, 9, 10}},
                                       {ref, {11, 12}},
                                       {ref, {13, 14, 15, 16}},
                                       {ref, {17, 18, 19, 20, 21, 22, 23}},
                                       {ref, {24, 25, 26}}};
    gko::array<long> recv_buffer{ref, recv_connections[rank].get_size()};
    gko::array<long> round_trip{ref, send_buffers[rank].get_size()};

    dcomm
        .i_all_to_all_v(ref, send_buffers[rank].get_const_data(),
                        recv_buffer.get_data())
        .wait();
    inverse
        ->i_all_to_all_v(ref, recv_buffer.get_const_data(),
                         round_trip.get_data())
        .wait();

    GKO_ASSERT_ARRAY_EQ(send_buffers[rank], round_trip);
}
