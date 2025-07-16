// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/distributed/row_gatherer.hpp>
#include <ginkgo/core/distributed/vector.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


using CollCommType =
#if GINKGO_HAVE_OPENMPI_PRE_4_1_X
    gko::experimental::mpi::DenseCommunicator;
#else
    gko::experimental::mpi::NeighborhoodCommunicator;
#endif


template <typename IndexType>
class RowGatherer : public ::testing::Test {
protected:
    using index_type = IndexType;
    using part_type =
        gko::experimental::distributed::Partition<index_type, gko::int64>;
    using map_type =
        gko::experimental::distributed::index_map<index_type, gko::int64>;
    using row_gatherer_type =
        gko::experimental::distributed::RowGatherer<index_type>;

    void SetUp() override { ASSERT_EQ(comm.size(), 6); }

    std::array<gko::array<gko::int64>, 6> create_recv_connections()
    {
        return {gko::array<gko::int64>{ref, {3, 5, 10, 11}},
                gko::array<gko::int64>{ref, {0, 1, 7, 12, 13}},
                gko::array<gko::int64>{ref, {3, 4, 17}},
                gko::array<gko::int64>{ref, {1, 2, 12, 14}},
                gko::array<gko::int64>{ref, {4, 5, 9, 10, 15, 16}},
                gko::array<gko::int64>{ref, {8, 12, 13, 14}}};
    }

    gko::size_type recv_connections_size()
    {
        gko::size_type size = 0;
        for (auto& recv_connections : create_recv_connections()) {
            size += recv_connections.get_size();
        }
        return size;
    }

    std::shared_ptr<gko::Executor> ref = gko::ReferenceExecutor::create();
    gko::experimental::mpi::communicator comm = MPI_COMM_WORLD;
    std::shared_ptr<part_type> part = part_type::build_from_global_size_uniform(
        this->ref, this->comm.size(), this->comm.size() * 3);
    map_type imap = map_type{ref, part, comm.rank(),
                             create_recv_connections()[comm.rank()]};
    std::shared_ptr<CollCommType> coll_comm =
        std::make_shared<CollCommType>(this->comm, imap);
};

TYPED_TEST_SUITE(RowGatherer, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(RowGatherer, CanDefaultConstruct)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;

    auto rg = RowGatherer::create(this->ref, this->comm);

    GKO_ASSERT_EQUAL_DIMENSIONS(rg, gko::dim<2>());
}


TYPED_TEST(RowGatherer, CanConstructWithEmptyCollectiveCommAndIndexMap)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    using IndexMap = typename TestFixture::map_type;
    auto coll_comm = std::make_shared<CollCommType>(this->comm);
    auto map = IndexMap{this->ref};

    auto rg = RowGatherer::create(this->ref, coll_comm, map);

    GKO_ASSERT_EQUAL_DIMENSIONS(rg, gko::dim<2>());
}


TYPED_TEST(RowGatherer, CanConstructFromCollectiveCommAndIndexMap)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;

    auto rg = RowGatherer::create(this->ref, this->coll_comm, this->imap);

    gko::dim<2> size{this->recv_connections_size(), 18};
    GKO_ASSERT_EQUAL_DIMENSIONS(rg, size);
}


TYPED_TEST(RowGatherer, CanCopy)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    auto rg = RowGatherer::create(this->ref, this->coll_comm, this->imap);

    auto copy = gko::clone(rg);

    GKO_ASSERT_EQUAL_DIMENSIONS(rg, copy);
    auto copy_coll_comm = std::dynamic_pointer_cast<const CollCommType>(
        copy->get_collective_communicator());
    ASSERT_EQ(*this->coll_comm, *copy_coll_comm);
    auto send_idxs = gko::make_const_array_view(
        rg->get_executor(), rg->get_num_send_idxs(), rg->get_const_send_idxs());
    auto copy_send_idxs = gko::make_const_array_view(
        copy->get_executor(), copy->get_num_send_idxs(),
        copy->get_const_send_idxs());
    GKO_ASSERT_ARRAY_EQ(send_idxs, copy_send_idxs);
}


TYPED_TEST(RowGatherer, CanMove)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    auto rg = RowGatherer::create(this->ref, this->coll_comm, this->imap);
    auto orig_send_idxs = rg->get_const_send_idxs();
    auto orig_coll_comm = rg->get_collective_communicator();
    auto copy = gko::clone(rg);

    auto move = RowGatherer::create(this->ref, this->comm);
    move->move_from(rg);

    GKO_ASSERT_EQUAL_DIMENSIONS(move, copy);
    GKO_ASSERT_EQUAL_DIMENSIONS(rg, gko::dim<2>());
    ASSERT_EQ(orig_send_idxs, move->get_const_send_idxs());
    ASSERT_EQ(orig_coll_comm, move->get_collective_communicator());
    ASSERT_EQ(copy->get_num_send_idxs(), move->get_num_send_idxs());
    ASSERT_EQ(rg->get_const_send_idxs(), nullptr);
    ASSERT_EQ(rg->get_num_send_idxs(), 0);
    ASSERT_NE(rg->get_collective_communicator(), nullptr);
}
