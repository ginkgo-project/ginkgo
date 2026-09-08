// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <core/test/utils.hpp>

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>

#include "core/test/utils/assertions.hpp"

using gko::experimental::mpi::comm_index_type;

template <typename CommunicatorType>
class CollectiveCommunicator : public ::testing::Test {
protected:
    using communicator_type = CommunicatorType;
    using part_type =
        gko::experimental::distributed::Partition<int, gko::int64>;
    using map_type = gko::experimental::distributed::index_map<int, gko::int64>;

    void SetUp() override { ASSERT_EQ(comm.size(), 6); }

    communicator_type create_default_comm()
    {
        auto part = gko::share(part_type::build_from_global_size_uniform(
            ref, comm.size(), comm.size() * 3));
        gko::array<gko::int64> recv_connections[] = {
            {ref, {3, 5, 10, 11}},
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
    std::array<gko::array<gko::int64>, 6> recv_connections{
        {{ref, {3, 5, 10, 11}},
         {ref, {0, 1, 7, 12, 13}},
         {ref, {3, 4, 17}},
         {ref, {1, 2, 12, 14}},
         {ref, {4, 5, 9, 10, 16, 15}},
         {ref, {8, 12, 13, 14}}}};
    int rank = comm.rank();
};

using gko::experimental::mpi::DenseCommunicator;
using gko::experimental::mpi::NeighborhoodCommunicator;
using CommunicatorTypes = ::testing::Types<
#if !GINKGO_HAVE_OPENMPI_PRE_4_1_X
    NeighborhoodCommunicator,
#endif
    DenseCommunicator>;
TYPED_TEST_SUITE(CollectiveCommunicator, CommunicatorTypes,
                 TypenameNameGenerator);


TYPED_TEST(CollectiveCommunicator, CanDefaultConstruct)
{
    using communicator_type = typename TestFixture::communicator_type;
    communicator_type nhcomm{this->comm};

    ASSERT_EQ(nhcomm.get_base_communicator(), this->comm);
    ASSERT_EQ(nhcomm.get_send_size(), 0);
    ASSERT_EQ(nhcomm.get_recv_size(), 0);
}


TYPED_TEST(CollectiveCommunicator, CanConstructFromIndexMap)
{
    using communicator_type = typename TestFixture::communicator_type;
    using part_type = typename TestFixture::part_type;
    using map_type = typename TestFixture::map_type;
    auto part = gko::share(part_type::build_from_global_size_uniform(
        this->ref, this->comm.size(), this->comm.size() * 3));
    auto imap = map_type{this->ref, part, this->rank,
                         this->recv_connections[this->rank]};

    communicator_type spcomm{this->comm, imap};

    std::array<gko::size_type, 6> send_sizes = {4, 6, 2, 4, 7, 3};
    ASSERT_EQ(spcomm.get_recv_size(),
              this->recv_connections[this->rank].get_size());
    ASSERT_EQ(spcomm.get_send_size(), send_sizes[this->rank]);
}


TYPED_TEST(CollectiveCommunicator, CanConstructFromEmptyIndexMap)
{
    using communicator_type = typename TestFixture::communicator_type;
    using map_type = typename TestFixture::map_type;
    auto imap = map_type{this->ref};

    communicator_type spcomm{this->comm, imap};

    ASSERT_EQ(spcomm.get_recv_size(), 0);
    ASSERT_EQ(spcomm.get_send_size(), 0);
}


TYPED_TEST(CollectiveCommunicator, CanConstructFromIndexMapWithoutConnection)
{
    using communicator_type = typename TestFixture::communicator_type;
    using part_type = typename TestFixture::part_type;
    using map_type = typename TestFixture::map_type;
    auto part = gko::share(part_type::build_from_global_size_uniform(
        this->ref, this->comm.size(), this->comm.size() * 3));
    auto imap = map_type{this->ref, part, this->rank, {this->ref, 0}};

    communicator_type spcomm{this->comm, imap};

    ASSERT_EQ(spcomm.get_recv_size(), 0);
    ASSERT_EQ(spcomm.get_send_size(), 0);
}


TYPED_TEST(CollectiveCommunicator, CanTestEquality)
{
    auto comm_a = this->create_default_comm();
    auto comm_b = this->create_default_comm();

    ASSERT_EQ(comm_a, comm_b);
}


TYPED_TEST(CollectiveCommunicator, CanTestInequality)
{
    using communicator_type = typename TestFixture::communicator_type;
    auto comm_a = this->create_default_comm();
    auto comm_b = communicator_type(this->comm);

    ASSERT_NE(comm_a, comm_b);
}


TYPED_TEST(CollectiveCommunicator, CanCopyConstruct)
{
    auto spcomm = this->create_default_comm();

    auto copy(spcomm);

    ASSERT_EQ(copy, spcomm);
}


TYPED_TEST(CollectiveCommunicator, CanCopyAssign)
{
    using communicator_type = typename TestFixture::communicator_type;
    auto spcomm = this->create_default_comm();
    communicator_type copy{this->comm};

    copy = spcomm;

    ASSERT_EQ(copy, spcomm);
}


TYPED_TEST(CollectiveCommunicator, CanMoveConstruct)
{
    using communicator_type = typename TestFixture::communicator_type;
    auto spcomm = this->create_default_comm();
    auto moved_from = spcomm;
    auto empty_comm = communicator_type{MPI_COMM_NULL};

    auto moved(std::move(moved_from));

    ASSERT_EQ(moved, spcomm);
    ASSERT_EQ(moved_from, empty_comm);
}


TYPED_TEST(CollectiveCommunicator, CanMoveAssign)
{
    using communicator_type = typename TestFixture::communicator_type;
    auto spcomm = this->create_default_comm();
    auto moved_from = spcomm;
    auto empty_comm = communicator_type{MPI_COMM_NULL};
    communicator_type moved{this->comm};

    moved = std::move(moved_from);

    ASSERT_EQ(moved, spcomm);
    ASSERT_EQ(moved_from, empty_comm);
}


TYPED_TEST(CollectiveCommunicator, CanCommunicateIalltoall)
{
    auto spcomm = this->create_default_comm();
    gko::array<gko::int64> recv_buffer{
        this->ref, this->recv_connections[this->rank].get_size()};
    gko::array<gko::int64> send_buffers[] = {
        {this->ref, {0, 1, 1, 2}},
        {this->ref, {3, 5, 3, 4, 4, 5}},
        {this->ref, {7, 8}},
        {this->ref, {10, 11, 9, 10}},
        {this->ref, {12, 13, 12, 14, 12, 13, 14}},
        {this->ref, {17, 16, 15}}};

    auto req = spcomm.i_all_to_all_v(this->ref,
                                     send_buffers[this->rank].get_const_data(),
                                     recv_buffer.get_data());
    req.wait();

    GKO_ASSERT_ARRAY_EQ(recv_buffer, this->recv_connections[this->rank]);
}


TYPED_TEST(CollectiveCommunicator, CanCommunicateIalltoallWhenEmpty)
{
    using communicator_type = typename TestFixture::communicator_type;
    communicator_type spcomm{this->comm};

    auto req = spcomm.i_all_to_all_v(this->ref, static_cast<int*>(nullptr),
                                     static_cast<int*>(nullptr));

    ASSERT_NO_THROW(req.wait());
}


TYPED_TEST(CollectiveCommunicator, CanCreateInverse)
{
    auto spcomm = this->create_default_comm();

    auto inverse = spcomm.create_inverse();

    ASSERT_EQ(inverse->get_recv_size(), spcomm.get_send_size());
    ASSERT_EQ(inverse->get_send_size(), spcomm.get_recv_size());
}


TYPED_TEST(CollectiveCommunicator, CanCommunicateRoundTrip)
{
    auto spcomm = this->create_default_comm();
    auto inverse = spcomm.create_inverse();
    gko::array<gko::int64> send_buffers[] = {
        {this->ref, {1, 2, 3, 4}},
        {this->ref, {5, 6, 7, 8, 9, 10}},
        {this->ref, {11, 12}},
        {this->ref, {13, 14, 15, 16}},
        {this->ref, {17, 18, 19, 20, 21, 22, 23}},
        {this->ref, {24, 25, 26}}};
    gko::array<gko::int64> recv_buffer{
        this->ref, this->recv_connections[this->rank].get_size()};
    gko::array<gko::int64> round_trip{this->ref,
                                      send_buffers[this->rank].get_size()};

    spcomm
        .i_all_to_all_v(this->ref, send_buffers[this->rank].get_const_data(),
                        recv_buffer.get_data())
        .wait();
    inverse
        ->i_all_to_all_v(this->ref, recv_buffer.get_const_data(),
                         round_trip.get_data())
        .wait();

    GKO_ASSERT_ARRAY_EQ(send_buffers[this->rank], round_trip);
}
