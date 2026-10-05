// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/distributed/row_scatterer.hpp>

#include "core/test/utils.hpp"


using CollCommType =
#if GINKGO_HAVE_OPENMPI_PRE_4_1_X
    gko::experimental::mpi::DenseCommunicator;
#else
    gko::experimental::mpi::NeighborhoodCommunicator;
#endif


template <typename IndexType>
class RowScatterer : public ::testing::Test {
protected:
    using index_type = IndexType;
    using part_type =
        gko::experimental::distributed::Partition<index_type, gko::int64>;
    using map_type =
        gko::experimental::distributed::index_map<index_type, gko::int64>;
    using row_scatterer_type =
        gko::experimental::distributed::RowScatterer<IndexType>;

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

TYPED_TEST_SUITE(RowScatterer, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(RowScatterer, CanDefaultConstructFromMpiCommunicator)
{
    using RowScatterer = typename TestFixture::row_scatterer_type;

    auto rs = RowScatterer::create(this->ref, this->comm);

    GKO_ASSERT_EQUAL_DIMENSIONS(rs, gko::dim<2>());
    auto coll_comm = rs->get_collective_communicator();
    ASSERT_NO_THROW(
        gko::as<gko::experimental::mpi::DenseCommunicator>(coll_comm));
}


TYPED_TEST(RowScatterer, CanConstructFromCollectiveCommAndIndexMap)
{
    using RowScatterer = typename TestFixture::row_scatterer_type;

    auto rs = RowScatterer::create(this->ref, this->coll_comm, this->imap);

    // transpose of the corresponding RowGatherer size
    gko::dim<2> size{18, this->recv_connections_size()};
    GKO_ASSERT_EQUAL_DIMENSIONS(rs, size);
}
