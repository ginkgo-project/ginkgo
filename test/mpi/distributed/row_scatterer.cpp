// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <memory>

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/distributed/row_gatherer.hpp>
#include <ginkgo/core/distributed/row_scatterer.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#if GINKGO_HAVE_OPENMPI_PRE_4_1_X
using CollCommType = gko::experimental::mpi::DenseCommunicator;
#else
using CollCommType = gko::experimental::mpi::NeighborhoodCommunicator;
#endif


template <typename IndexType>
class RowScatterer : public CommonMpiTestFixture {
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

    RowScatterer()
    {
        int rank = comm.rank();
        auto part = gko::share(part_type::build_from_global_size_uniform(
            exec, comm.size(), comm.size() * 3));
        auto recv_connections = create_recv_connections<gko::int64>()[rank];
        auto imap = map_type{exec, part, comm.rank(), recv_connections};
        auto coll_comm = std::make_shared<CollCommType>(comm, imap);
        rg = row_gatherer_type::create(exec, coll_comm, imap);
        rs = row_scatterer_type::create_from_gatherer(exec, rg);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 6); }

    template <typename T>
    std::array<gko::array<T>, 6> create_recv_connections()
    {
        return {gko::array<T>{exec, {3, 5, 10, 11}},
                gko::array<T>{exec, {0, 1, 7, 12, 13}},
                gko::array<T>{exec, {3, 4, 17}},
                gko::array<T>{exec, {1, 2, 12, 14}},
                gko::array<T>{exec, {4, 5, 9, 10, 15, 16}},
                gko::array<T>{exec, {8, 12, 13, 14}}};
    }

    std::shared_ptr<const gko::Executor> host_exec = exec->get_master();
    std::shared_ptr<const gko::Executor> mpi_exec =
        gko::experimental::mpi::requires_host_buffer(exec, comm) ? host_exec
                                                                 : exec;
    std::shared_ptr<row_gatherer_type> rg;
    std::shared_ptr<row_scatterer_type> rs;
};

TYPED_TEST_SUITE(RowScatterer, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(RowScatterer, CanCreateFromGatherer)
{
    // The size of the scatterer should be the transpose of the gatherer
    auto rg_size = this->rg->get_size();
    gko::dim<2> expected_size{rg_size[1], rg_size[0]};
    GKO_ASSERT_EQUAL_DIMENSIONS(this->rs, expected_size);
}


TYPED_TEST(RowScatterer, ScatterIsGatherTranspose)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto num_local_rows = static_cast<gko::int64>(3);

    // Create distributed vector with known values
    auto b = Vector::create(
        this->exec, this->comm, gko::dim<2>{18, 1},
        gko::initialize<Dense>({offset, offset + 1, offset + 2}, this->exec));

    // Gather ghost values
    auto recv_size = this->rg->get_collective_communicator()->get_recv_size();
    auto ghost_vals = Vector::create(
        this->mpi_exec, this->comm, gko::dim<2>{this->rg->get_size()[0], 1},
        gko::dim<2>{static_cast<gko::size_type>(recv_size), 1});
    this->rg->apply_async(b, ghost_vals).wait();

    // Scatter ghost values back
    auto target = Vector::create(
        this->exec, this->comm, gko::dim<2>{18, 1},
        gko::dim<2>{static_cast<gko::size_type>(num_local_rows), 1});
    auto target_local = const_cast<Dense*>(target->get_local_vector());
    target_local->fill(0.0);

    auto scatter_req = this->rs->apply_async(ghost_vals);
    this->rs->wait_and_accumulate(scatter_req, target);

    // Copy result to host for verification
    auto host_target = gko::clone(this->ref, target);
    auto host_target_local = host_target->get_local_vector();

    // Expected accumulated values (from
    // core/test/mpi/distributed/row_scatterer.cpp) Ghost connections per rank
    // (global indices):
    //   Rank 0: {3, 5, 10, 11}
    //   Rank 1: {0, 1, 7, 12, 13}
    //   Rank 2: {3, 4, 17}
    //   Rank 3: {1, 2, 12, 14}
    //   Rank 4: {4, 5, 9, 10, 15, 16}
    //   Rank 5: {8, 12, 13, 14}
    // Row k is accessed by ranks that have k in recv_connections.
    // The scattered value = sum of b[k] over all accessing ranks.
    // Since b[k] = k for each rank that owns it, each accessing rank
    // scatters value k back. So accumulated = k * (number of accessing ranks).
    std::array<std::array<double, 3>, 6> expected = {{
        {0, 2, 2},  // rank 0: row 0 by r1=0, row 1 by r1+r3=2, row 2 by r3=2
        {6, 8,
         10},  // rank 1: row 3 by r0+r2=6, row 4 by r2+r4=8, row 5 by r0+r4=10
        {0, 7, 8},  // rank 2: row 6 by nobody=0, row 7 by r1=7, row 8 by r5=8
        {9, 20,
         11},  // rank 3: row 9 by r4=9, row 10 by r0+r4=20, row 11 by r0=11
        {36, 26, 28},  // rank 4: row 12 by r1+r3+r5=36, row 13 by r1+r5=26, row
                       // 14 by r3+r5=28
        {15, 16,
         17},  // rank 5: row 15 by r4=15, row 16 by r4=16, row 17 by r2=17
    }};

    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        EXPECT_DOUBLE_EQ(host_target_local->at(i, 0), expected[rank][i])
            << "rank=" << rank << " local_row=" << i;
    }
}


TYPED_TEST(RowScatterer, CanScatterConsecutively)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto num_local_rows = static_cast<gko::int64>(3);

    auto b =
        Vector::create(this->exec, this->comm, gko::dim<2>{18, 1},
                       gko::initialize<Dense>({1.0, 1.0, 1.0}, this->exec));

    auto recv_size = this->rg->get_collective_communicator()->get_recv_size();
    auto ghost_vals = Vector::create(
        this->mpi_exec, this->comm, gko::dim<2>{this->rg->get_size()[0], 1},
        gko::dim<2>{static_cast<gko::size_type>(recv_size), 1});
    this->rg->apply_async(b, ghost_vals).wait();

    auto target = Vector::create(
        this->exec, this->comm, gko::dim<2>{18, 1},
        gko::dim<2>{static_cast<gko::size_type>(num_local_rows), 1});

    // Scatter twice consecutively — should work without errors
    auto target_local = const_cast<Dense*>(target->get_local_vector());
    target_local->fill(0.0);
    auto req1 = this->rs->apply_async(ghost_vals);
    this->rs->wait_and_accumulate(req1, target);

    target_local->fill(0.0);
    auto req2 = this->rs->apply_async(ghost_vals);
    this->rs->wait_and_accumulate(req2, target);

    // Second scatter should produce same result as first
    auto host_target = gko::clone(this->ref, target);
    auto host_target_local = host_target->get_local_vector();

    // Uniform weight=1 scatter of all-1s vector: count contributions per row
    std::array<std::array<double, 3>, 6> expected = {{
        {1, 2, 1},
        {2, 2, 2},
        {0, 1, 1},
        {1, 2, 1},
        {3, 2, 2},
        {1, 1, 1},
    }};

    for (gko::int64 i = 0; i < num_local_rows; ++i) {
        EXPECT_DOUBLE_EQ(host_target_local->at(i, 0), expected[rank][i])
            << "rank=" << rank << " local_row=" << i;
    }
}
