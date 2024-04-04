// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/distributed/neighborhood_communicator.hpp>

#include "core/test/utils/assertions.hpp"

using gko::experimental::mpi::comm_index_type;

class NeighborhoodCommunicator : public ::testing::Test {
protected:
    using part_type = gko::experimental::distributed::Partition<int, long>;
    using map_type = gko::experimental::distributed::index_map<int, long>;

    void SetUp() override { ASSERT_EQ(comm.size(), 6); }

    std::shared_ptr<gko::Executor> ref = gko::ReferenceExecutor::create();
    gko::experimental::mpi::communicator comm = MPI_COMM_WORLD;
    int rank = comm.rank();
};


TEST_F(NeighborhoodCommunicator, CanDefaultConstruct)
{
    gko::experimental::mpi::NeighborhoodCommunicator nhcomm{comm};

    ASSERT_EQ(nhcomm.get_base_communicator(), comm);
    ASSERT_EQ(nhcomm.get_send_size(), 0);
    ASSERT_EQ(nhcomm.get_recv_size(), 0);
}


TEST_F(NeighborhoodCommunicator, CanConstructFromIndexMap)
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

    gko::experimental::mpi::NeighborhoodCommunicator spcomm{comm, imap};

    std::array<gko::size_type, 6> send_sizes = {4, 6, 2, 4, 7, 3};
    ASSERT_EQ(spcomm.get_recv_size(), recv_connections[rank].get_size());
    ASSERT_EQ(spcomm.get_send_size(), send_sizes[rank]);
}


TEST_F(NeighborhoodCommunicator, CanConstructFromEnvelopData)
{
    std::vector<comm_index_type> sources[] = {{1, 2}, {0, 2, 4}, {1, 5},
                                              {0, 4}, {1, 3, 5}, {2, 4}};
    std::vector<comm_index_type> recv_sizes[] = {{2, 2}, {2, 1, 2}, {2, 1},
                                                 {2, 2}, {2, 2, 2}, {1, 3}};
    std::vector<comm_index_type> destinations = sources[rank];
    std::vector<comm_index_type> send_sizes[] = {{2, 2}, {2, 2, 2}, {1, 1},
                                                 {2, 2}, {2, 2, 3}, {1, 2}};
    std::vector<comm_index_type> recv_offsets(recv_sizes[rank].size() + 1);
    std::vector<comm_index_type> send_offsets(send_sizes[rank].size() + 1);
    std::partial_sum(recv_sizes[rank].begin(), recv_sizes[rank].end(),
                     recv_offsets.begin() + 1);
    std::partial_sum(send_sizes[rank].begin(), send_sizes[rank].end(),
                     send_offsets.begin() + 1);

    gko::experimental::mpi::NeighborhoodCommunicator spcomm{
        comm,         sources[rank],    recv_sizes[rank], recv_offsets,
        destinations, send_sizes[rank], send_offsets};

    ASSERT_EQ(spcomm.get_recv_size(), recv_offsets.back());
    ASSERT_EQ(spcomm.get_send_size(), send_offsets.back());
}


TEST_F(NeighborhoodCommunicator, CanConstructFromEmptyIndexMap)
{
    auto imap = map_type{ref};

    gko::experimental::mpi::NeighborhoodCommunicator spcomm{comm, imap};

    ASSERT_EQ(spcomm.get_recv_size(), 0);
    ASSERT_EQ(spcomm.get_send_size(), 0);
}


TEST_F(NeighborhoodCommunicator, CanConstructFromIndexMapWithoutConnection)
{
    auto part = gko::share(part_type::build_from_global_size_uniform(
        ref, comm.size(), comm.size() * 3));
    auto imap = map_type{ref, part, comm.rank(), {ref, 0}};

    gko::experimental::mpi::NeighborhoodCommunicator spcomm{comm, imap};

    ASSERT_EQ(spcomm.get_recv_size(), 0);
    ASSERT_EQ(spcomm.get_send_size(), 0);
}


TEST_F(NeighborhoodCommunicator, CanConstructFromEmptyEnvelopData)
{
    std::vector<comm_index_type> sources;
    std::vector<comm_index_type> recv_sizes;
    std::vector<comm_index_type> destinations;
    std::vector<comm_index_type> send_sizes;
    std::vector<comm_index_type> recv_offsets{0};
    std::vector<comm_index_type> send_offsets{0};

    gko::experimental::mpi::NeighborhoodCommunicator spcomm{
        comm,         sources,    send_sizes,  send_offsets,
        destinations, recv_sizes, recv_offsets};

    ASSERT_EQ(spcomm.get_recv_size(), 0);
    ASSERT_EQ(spcomm.get_send_size(), 0);
}


TEST_F(NeighborhoodCommunicator, CanCommunicateIalltoall)
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
    gko::experimental::mpi::NeighborhoodCommunicator spcomm{comm, imap};
    gko::array<long> recv_buffer{ref, recv_connections[rank].get_size()};
    gko::array<long> send_buffers[] = {{ref, {0, 1, 1, 2}},
                                       {ref, {3, 5, 3, 4, 4, 5}},
                                       {ref, {7, 8}},
                                       {ref, {10, 11, 9, 10}},
                                       {ref, {12, 13, 12, 14, 12, 13, 14}},
                                       {ref, {17, 16, 15}}};

    auto req = spcomm.i_all_to_all_v(ref, send_buffers[rank].get_const_data(),
                                     recv_buffer.get_data());
    req.wait();

    GKO_ASSERT_ARRAY_EQ(recv_buffer, recv_connections[rank]);
}


TEST_F(NeighborhoodCommunicator, CanCommunicateIalltoallWhenEmpty)
{
    gko::experimental::mpi::NeighborhoodCommunicator spcomm{comm};

    auto req = spcomm.i_all_to_all_v(ref, static_cast<int*>(nullptr),
                                     static_cast<int*>(nullptr));
    req.wait();
}


TEST_F(NeighborhoodCommunicator, CanCreateInverse)
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
    gko::experimental::mpi::NeighborhoodCommunicator spcomm{comm, imap};

    auto inverse = spcomm.create_inverse();

    ASSERT_EQ(inverse->get_recv_size(), spcomm.get_send_size());
    ASSERT_EQ(inverse->get_send_size(), spcomm.get_recv_size());
}


TEST_F(NeighborhoodCommunicator, CanCommunicateRoundTrip)
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
    gko::experimental::mpi::NeighborhoodCommunicator spcomm{comm, imap};
    auto inverse = spcomm.create_inverse();
    gko::array<long> send_buffers[] = {{ref, {1, 2, 3, 4}},
                                       {ref, {5, 6, 7, 8, 9, 10}},
                                       {ref, {11, 12}},
                                       {ref, {13, 14, 15, 16}},
                                       {ref, {17, 18, 19, 20, 21, 22, 23}},
                                       {ref, {24, 25, 26}}};
    gko::array<long> recv_buffer{ref, recv_connections[rank].get_size()};
    gko::array<long> round_trip{ref, send_buffers[rank].get_size()};

    spcomm
        .i_all_to_all_v(ref, send_buffers[rank].get_const_data(),
                        recv_buffer.get_data())
        .wait();
    inverse
        ->i_all_to_all_v(ref, recv_buffer.get_const_data(),
                         round_trip.get_data())
        .wait();

    GKO_ASSERT_ARRAY_EQ(send_buffers[rank], round_trip);
}
