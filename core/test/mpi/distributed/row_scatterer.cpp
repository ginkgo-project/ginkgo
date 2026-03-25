// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/distributed/row_gatherer.hpp>
#include <ginkgo/core/distributed/row_scatterer.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


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
    using row_gatherer_type =
        gko::experimental::distributed::RowGatherer<index_type>;
    using row_scatterer_type =
        gko::experimental::distributed::RowScatterer<index_type>;

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


TYPED_TEST(RowScatterer, CanCreateFromGatherer)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    using RowScatterer = typename TestFixture::row_scatterer_type;

    auto rg = RowGatherer::create(this->ref, this->coll_comm, this->imap);
    auto rs = RowScatterer::create_from_gatherer(this->ref, *rg);

    // The size of the scatterer should be the transpose of the gatherer
    auto rg_size = rg->get_size();
    gko::dim<2> expected_size{rg_size[1], rg_size[0]};
    GKO_ASSERT_EQUAL_DIMENSIONS(rs, expected_size);
}


TYPED_TEST(RowScatterer, ScatterIsGatherTranspose)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    using RowScatterer = typename TestFixture::row_scatterer_type;
    using Vector = gko::experimental::distributed::Vector<double>;
    using Dense = gko::matrix::Dense<double>;

    auto rg = RowGatherer::create(this->ref, this->coll_comm, this->imap);
    auto rs = RowScatterer::create_from_gatherer(this->ref, *rg);

    // Create a distributed vector with known values: each rank owns 3 rows
    // Values: rank*3+0, rank*3+1, rank*3+2
    auto num_local_rows = static_cast<gko::int64>(3);
    auto b = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::dim<2>{static_cast<gko::size_type>(num_local_rows), 1});
    auto b_local = const_cast<Dense*>(b->get_local_vector());
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        b_local->at(i, 0) = static_cast<double>(this->comm.rank() * 3 + i);
    }

    // Gather ghost values into a distributed::Vector
    auto recv_size = this->coll_comm->get_recv_size();
    auto ghost_vals = Vector::create(
        this->ref, this->comm,
        gko::dim<2>{static_cast<gko::size_type>(rg->get_size()[0]), 1},
        gko::dim<2>{static_cast<gko::size_type>(recv_size), 1});
    rg->apply_async(b, ghost_vals).wait();

    // Now scatter those ghost values back
    auto target = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::dim<2>{static_cast<gko::size_type>(num_local_rows), 1});
    auto target_local = const_cast<Dense*>(target->get_local_vector());
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        target_local->at(i, 0) = 0.0;
    }

    auto req = rs->apply_async(ghost_vals);
    rs->wait_and_accumulate(req, target);

    // After scattering, each local row should have received contributions
    // from all ranks that had it as a ghost.
    // The value accumulated at position k should be the sum of the original
    // values b[k] over all ranks that needed row k as a ghost.
    //
    // From the recv_connections:
    // rank 0 needs: 3, 5, 10, 11
    // rank 1 needs: 0, 1, 7, 12, 13
    // rank 2 needs: 3, 4, 17
    // rank 3 needs: 1, 2, 12, 14
    // rank 4 needs: 4, 5, 9, 10, 15, 16
    // rank 5 needs: 8, 12, 13, 14
    //
    // Row 0 (rank 0, local 0): needed by rank 1 -> accumulated = 0
    // Row 1 (rank 0, local 1): needed by rank 1, 3 -> accumulated = 1 + 1 = 2
    // Row 2 (rank 0, local 2): needed by rank 3 -> accumulated = 2
    // Row 3 (rank 1, local 0): needed by rank 0, 2 -> accumulated = 3 + 3 = 6
    // Row 4 (rank 1, local 1): needed by rank 2, 4 -> accumulated = 4 + 4 = 8
    // Row 5 (rank 1, local 2): needed by rank 0, 4 -> accumulated = 5 + 5 = 10
    // Row 6 (rank 2, local 0): nobody needs it -> 0
    // Row 7 (rank 2, local 1): needed by rank 1 -> 7
    // Row 8 (rank 2, local 2): needed by rank 5 -> 8
    // Row 9 (rank 3, local 0): needed by rank 4 -> 9
    // Row 10 (rank 3, local 1): needed by rank 0, 4 -> 10 + 10 = 20
    // Row 11 (rank 3, local 2): needed by rank 0 -> 11
    // Row 12 (rank 4, local 0): needed by rank 1, 3, 5 -> 12+12+12 = 36
    // Row 13 (rank 4, local 1): needed by rank 1, 5 -> 13 + 13 = 26
    // Row 14 (rank 4, local 2): needed by rank 3, 5 -> 14 + 14 = 28
    // Row 15 (rank 5, local 0): needed by rank 4 -> 15
    // Row 16 (rank 5, local 1): needed by rank 4 -> 16
    // Row 17 (rank 5, local 2): needed by rank 2 -> 17
    std::array<std::array<double, 3>, 6> expected = {{
        {0, 2, 2},
        {6, 8, 10},
        {0, 7, 8},
        {9, 20, 11},
        {36, 26, 28},
        {15, 16, 17},
    }};

    auto rank = this->comm.rank();
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        EXPECT_DOUBLE_EQ(target_local->at(i, 0), expected[rank][i])
            << "rank=" << rank << " local_row=" << i;
    }
}


TYPED_TEST(RowScatterer, CanOverlapWorkWithScatter)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    using RowScatterer = typename TestFixture::row_scatterer_type;
    using Vector = gko::experimental::distributed::Vector<double>;
    using Dense = gko::matrix::Dense<double>;

    auto rg = RowGatherer::create(this->ref, this->coll_comm, this->imap);
    auto rs = RowScatterer::create_from_gatherer(this->ref, *rg);

    // Create distributed vector with known values
    auto num_local_rows = static_cast<gko::int64>(3);
    auto b = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::dim<2>{static_cast<gko::size_type>(num_local_rows), 1});
    auto b_local = const_cast<Dense*>(b->get_local_vector());
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        b_local->at(i, 0) = static_cast<double>(this->comm.rank() * 3 + i);
    }

    // Gather ghost values
    auto recv_size = this->coll_comm->get_recv_size();
    auto ghost_vals = Vector::create(
        this->ref, this->comm,
        gko::dim<2>{static_cast<gko::size_type>(rg->get_size()[0]), 1},
        gko::dim<2>{static_cast<gko::size_type>(recv_size), 1});
    rg->apply_async(b, ghost_vals).wait();

    auto target = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::dim<2>{static_cast<gko::size_type>(num_local_rows), 1});
    auto target_local = const_cast<Dense*>(target->get_local_vector());
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        target_local->at(i, 0) = 0.0;
    }

    // Two-phase: start async, do other work, then wait and accumulate
    auto req = rs->apply_async(ghost_vals);
    // ... could do other work here while MPI is in flight ...
    rs->wait_and_accumulate(req, target);

    // Same expected values as ScatterIsGatherTranspose
    std::array<std::array<double, 3>, 6> expected = {{
        {0, 2, 2},
        {6, 8, 10},
        {0, 7, 8},
        {9, 20, 11},
        {36, 26, 28},
        {15, 16, 17},
    }};

    auto rank = this->comm.rank();
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        EXPECT_DOUBLE_EQ(target_local->at(i, 0), expected[rank][i])
            << "rank=" << rank << " local_row=" << i;
    }
}


TYPED_TEST(RowScatterer, WeightedScatterAppliesWeightsCorrectly)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    using RowScatterer = typename TestFixture::row_scatterer_type;
    using Vector = gko::experimental::distributed::Vector<double>;
    using Dense = gko::matrix::Dense<double>;

    auto rg = RowGatherer::create(this->ref, this->coll_comm, this->imap);
    auto rs = RowScatterer::create_from_gatherer(this->ref, *rg);

    // Create distributed vector with all 1s
    auto num_local_rows = static_cast<gko::int64>(3);
    auto b = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::dim<2>{static_cast<gko::size_type>(num_local_rows), 1});
    auto b_local = const_cast<Dense*>(b->get_local_vector());
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        b_local->at(i, 0) = 1.0;
    }

    // Gather ghost values (all 1s)
    auto recv_size = this->coll_comm->get_recv_size();
    auto ghost_vals = Vector::create(
        this->ref, this->comm,
        gko::dim<2>{static_cast<gko::size_type>(rg->get_size()[0]), 1},
        gko::dim<2>{static_cast<gko::size_type>(recv_size), 1});
    rg->apply_async(b, ghost_vals).wait();

    // Create weights = 2.0 for all ghost values (as a Dense)
    auto weights = Dense::create(
        this->ref, gko::dim<2>{static_cast<gko::size_type>(recv_size), 1});
    for (gko::size_type i = 0; i < static_cast<gko::size_type>(recv_size);
         ++i) {
        weights->at(i, 0) = 2.0;
    }

    // Scatter with weights
    auto target = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::dim<2>{static_cast<gko::size_type>(num_local_rows), 1});
    auto target_local = const_cast<Dense*>(target->get_local_vector());
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        target_local->at(i, 0) = 0.0;
    }

    auto req = rs->apply_async(weights, ghost_vals);
    rs->wait_and_accumulate(req, target);

    // Since all values are 1 and weights are 2, each contribution is 2.
    // Count of contributions per row determines the result.
    std::array<std::array<double, 3>, 6> expected = {{
        {2, 4, 2},
        {4, 4, 4},
        {0, 2, 2},
        {2, 4, 2},
        {6, 4, 4},
        {2, 2, 2},
    }};

    auto rank = this->comm.rank();
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        EXPECT_DOUBLE_EQ(target_local->at(i, 0), expected[rank][i])
            << "rank=" << rank << " local_row=" << i;
    }
}


TYPED_TEST(RowScatterer, NonUniformWeightsScatterCorrectly)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    using RowScatterer = typename TestFixture::row_scatterer_type;
    using Vector = gko::experimental::distributed::Vector<double>;
    using Dense = gko::matrix::Dense<double>;

    auto rg = RowGatherer::create(this->ref, this->coll_comm, this->imap);
    auto rs = RowScatterer::create_from_gatherer(this->ref, *rg);

    // Create distributed vector with all 1s
    auto num_local_rows = static_cast<gko::int64>(3);
    auto b = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::dim<2>{static_cast<gko::size_type>(num_local_rows), 1});
    auto b_local = const_cast<Dense*>(b->get_local_vector());
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        b_local->at(i, 0) = 1.0;
    }

    // Gather ghost values (all 1s)
    auto recv_size = this->coll_comm->get_recv_size();
    auto ghost_vals = Vector::create(
        this->ref, this->comm,
        gko::dim<2>{static_cast<gko::size_type>(rg->get_size()[0]), 1},
        gko::dim<2>{static_cast<gko::size_type>(recv_size), 1});
    rg->apply_async(b, ghost_vals).wait();

    // Non-uniform weights: each rank uses weight = (rank + 1.0)
    // This exercises the partition-of-unity pattern where different ranks
    // contribute different amounts to the same DOF.
    auto rank = this->comm.rank();
    auto weights = Dense::create(
        this->ref, gko::dim<2>{static_cast<gko::size_type>(recv_size), 1});
    for (gko::size_type i = 0; i < static_cast<gko::size_type>(recv_size);
         ++i) {
        weights->at(i, 0) = static_cast<double>(rank + 1);
    }

    // Scatter with non-uniform weights
    auto target = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::dim<2>{static_cast<gko::size_type>(num_local_rows), 1});
    auto target_local = const_cast<Dense*>(target->get_local_vector());
    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        target_local->at(i, 0) = 0.0;
    }

    auto req = rs->apply_async(weights, ghost_vals);
    rs->wait_and_accumulate(req, target);

    // Ghost connections per rank (global indices):
    //   Rank 0: {3, 5, 10, 11}     -> weight 1.0
    //   Rank 1: {0, 1, 7, 12, 13}  -> weight 2.0
    //   Rank 2: {3, 4, 17}         -> weight 3.0
    //   Rank 3: {1, 2, 12, 14}     -> weight 4.0
    //   Rank 4: {4, 5, 9, 10, 15, 16} -> weight 5.0
    //   Rank 5: {8, 12, 13, 14}    -> weight 6.0
    //
    // Each owning rank accumulates weighted contributions:
    //   Row 0  (rank 0, local 0): rank 1 -> 2.0
    //   Row 1  (rank 0, local 1): rank 1 + rank 3 -> 2.0 + 4.0 = 6.0
    //   Row 2  (rank 0, local 2): rank 3 -> 4.0
    //   Row 3  (rank 1, local 0): rank 0 + rank 2 -> 1.0 + 3.0 = 4.0
    //   Row 4  (rank 1, local 1): rank 2 + rank 4 -> 3.0 + 5.0 = 8.0
    //   Row 5  (rank 1, local 2): rank 0 + rank 4 -> 1.0 + 5.0 = 6.0
    //   Row 6  (rank 2, local 0): no ghost -> 0.0
    //   Row 7  (rank 2, local 1): rank 1 -> 2.0
    //   Row 8  (rank 2, local 2): rank 5 -> 6.0
    //   Row 9  (rank 3, local 0): rank 4 -> 5.0
    //   Row 10 (rank 3, local 1): rank 0 + rank 4 -> 1.0 + 5.0 = 6.0
    //   Row 11 (rank 3, local 2): rank 0 -> 1.0
    //   Row 12 (rank 4, local 0): rank 1+3+5 -> 2.0+4.0+6.0 = 12.0
    //   Row 13 (rank 4, local 1): rank 1 + rank 5 -> 2.0 + 6.0 = 8.0
    //   Row 14 (rank 4, local 2): rank 3 + rank 5 -> 4.0 + 6.0 = 10.0
    //   Row 15 (rank 5, local 0): rank 4 -> 5.0
    //   Row 16 (rank 5, local 1): rank 4 -> 5.0
    //   Row 17 (rank 5, local 2): rank 2 -> 3.0
    std::array<std::array<double, 3>, 6> expected = {{
        {2.0, 6.0, 4.0},    // rank 0
        {4.0, 8.0, 6.0},    // rank 1
        {0.0, 2.0, 6.0},    // rank 2
        {5.0, 6.0, 1.0},    // rank 3
        {12.0, 8.0, 10.0},  // rank 4
        {5.0, 5.0, 3.0},    // rank 5
    }};

    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        EXPECT_DOUBLE_EQ(target_local->at(i, 0), expected[rank][i])
            << "rank=" << rank << " local_row=" << i;
    }
}
