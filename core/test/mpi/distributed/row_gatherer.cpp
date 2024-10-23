// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/distributed/row_gatherer.hpp>
#include <ginkgo/core/distributed/vector.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


template <typename IndexType>
class RowGatherer : public ::testing::Test {
protected:
    using index_type = IndexType;
    using part_type =
        gko::experimental::distributed::Partition<index_type, long>;
    using map_type =
        gko::experimental::distributed::index_map<index_type, long>;
    using row_gatherer_type =
        gko::experimental::distributed::RowGatherer<index_type>;

    void SetUp() override { ASSERT_EQ(comm.size(), 6); }

    template <typename T>
    std::array<gko::array<T>, 6> create_recv_connections()
    {
        return {gko::array<T>{ref, {3, 5, 10, 11}},
                gko::array<T>{ref, {0, 1, 7, 12, 13}},
                gko::array<T>{ref, {3, 4, 17}},
                gko::array<T>{ref, {1, 2, 12, 14}},
                gko::array<T>{ref, {4, 5, 9, 10, 15, 16}},
                gko::array<T>{ref, {8, 12, 13, 14}}};
    }

    std::shared_ptr<gko::Executor> ref = gko::ReferenceExecutor::create();
    gko::experimental::mpi::communicator comm = MPI_COMM_WORLD;
};

TYPED_TEST_SUITE(RowGatherer, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(RowGatherer, CanDefaultConstruct)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;

    auto rg = RowGatherer::create(this->ref, this->comm);

    GKO_ASSERT_EQUAL_DIMENSIONS(rg, gko::dim<2>());
}


TYPED_TEST(RowGatherer, CanConstructWithEmptCollectiveCommAndIndexMap)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    using IndexMap = typename TestFixture::map_type;
    auto coll_comm =
        std::make_shared<gko::experimental::mpi::NeighborhoodCommunicator>(
            this->comm);
    auto map = IndexMap{this->ref};

    auto rg = RowGatherer::create(this->ref, coll_comm, map);

    GKO_ASSERT_EQUAL_DIMENSIONS(rg, gko::dim<2>());
}


TYPED_TEST(RowGatherer, CanConstructFromCollectiveCommAndIndexMap)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    using Part = typename TestFixture::part_type;
    using IndexMap = typename TestFixture::map_type;
    int rank = this->comm.rank();
    auto part = gko::share(Part::build_from_global_size_uniform(
        this->ref, this->comm.size(), this->comm.size() * 3));
    auto recv_connections =
        this->template create_recv_connections<long>()[rank];
    auto imap = IndexMap{this->ref, part, this->comm.rank(), recv_connections};
    auto coll_comm =
        std::make_shared<gko::experimental::mpi::NeighborhoodCommunicator>(
            this->comm, imap);

    auto rg = RowGatherer::create(this->ref, coll_comm, imap);

    gko::dim<2> size{recv_connections.get_size(), 18};
    GKO_ASSERT_EQUAL_DIMENSIONS(rg, size);
}
