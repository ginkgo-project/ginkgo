// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <memory>
#include <random>

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/distributed/row_gatherer.hpp>
#include <ginkgo/core/distributed/vector.hpp>

#include "core/test/utils.hpp"
#include "ginkgo/core/base/exception.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#if GINKGO_HAVE_OPENMPI_PRE_4_1_X
using CollCommType = gko::experimental::mpi::DenseCommunicator;
#else
using CollCommType = gko::experimental::mpi::NeighborhoodCommunicator;
#endif


template <typename IndexType>
class RowGatherer : public CommonMpiTestFixture {
protected:
    using index_type = IndexType;
    using part_type =
        gko::experimental::distributed::Partition<index_type, gko::int64>;
    using map_type =
        gko::experimental::distributed::index_map<index_type, gko::int64>;
    using row_gatherer_type =
        gko::experimental::distributed::RowGatherer<index_type>;

    RowGatherer()
    {
        int rank = comm.rank();
        auto part = gko::share(part_type::build_from_global_size_uniform(
            exec, comm.size(), comm.size() * 3));
        auto recv_connections = create_recv_connections<gko::int64>()[rank];
        auto imap = map_type{exec, part, comm.rank(), recv_connections};
        auto coll_comm = std::make_shared<CollCommType>(comm, imap);
        rg = row_gatherer_type::create(exec, coll_comm, imap);
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
};

TYPED_TEST_SUITE(RowGatherer, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(RowGatherer, CanApplyAsync)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b = Vector::create(
        this->exec, this->comm, gko::dim<2>{18, 1},
        gko::initialize<Dense>({offset, offset + 1, offset + 2}, this->exec));
    auto expected = this->template create_recv_connections<double>()[rank];
    auto x = Vector::create(this->mpi_exec, this->comm,
                            gko::dim<2>{this->rg->get_size()[0], 1},
                            gko::dim<2>{expected.get_size(), 1});

    auto req = this->rg->apply_async(b, x);
    req.wait();

    auto expected_vec = Vector::create(
        this->mpi_exec, this->comm, gko::dim<2>{this->rg->get_size()[0], 1},
        Dense::create(this->mpi_exec, gko::dim<2>{expected.get_size(), 1},
                      expected, 1));
    GKO_ASSERT_MTX_NEAR(x->get_local_vector(), expected_vec->get_local_vector(),
                        0.0);
}


TYPED_TEST(RowGatherer, CanApplyAsyncConsequetively)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b = Vector::create(
        this->exec, this->comm, gko::dim<2>{18, 1},
        gko::initialize<Dense>({offset, offset + 1, offset + 2}, this->exec));
    auto expected = this->template create_recv_connections<double>()[rank];
    auto x = Vector::create(this->mpi_exec, this->comm,
                            gko::dim<2>{this->rg->get_size()[0], 1},
                            gko::dim<2>{expected.get_size(), 1});

    this->rg->apply_async(b, x).wait();
    this->rg->apply_async(b, x).wait();

    auto expected_vec = Vector::create(
        this->mpi_exec, this->comm, gko::dim<2>{this->rg->get_size()[0], 1},
        Dense::create(this->mpi_exec, gko::dim<2>{expected.get_size(), 1},
                      expected, 1));
    GKO_ASSERT_MTX_NEAR(x->get_local_vector(), expected_vec->get_local_vector(),
                        0.0);
}


TYPED_TEST(RowGatherer, CanApplyAsyncWithWorkspace)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b = Vector::create(
        this->exec, this->comm, gko::dim<2>{18, 1},
        gko::initialize<Dense>({offset, offset + 1, offset + 2}, this->exec));
    auto expected = this->template create_recv_connections<double>()[rank];
    auto x = Vector::create(this->mpi_exec, this->comm,
                            gko::dim<2>{this->rg->get_size()[0], 1},
                            gko::dim<2>{expected.get_size(), 1});
    gko::array<char> workspace;

    auto req = this->rg->apply_async(b, x, workspace);
    req.wait();

    auto expected_vec = Vector::create(
        this->mpi_exec, this->comm, gko::dim<2>{this->rg->get_size()[0], 1},
        Dense::create(this->mpi_exec, gko::dim<2>{expected.get_size(), 1},
                      expected, 1));
    GKO_ASSERT_MTX_NEAR(x->get_local_vector(), expected_vec->get_local_vector(),
                        0.0);
    ASSERT_GT(workspace.get_size(), 0);
}


TYPED_TEST(RowGatherer, CanApplyAsyncMultipleTimesWithWorkspace)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b1 = Vector::create(
        this->exec, this->comm, gko::dim<2>{18, 1},
        gko::initialize<Dense>({offset, offset + 1, offset + 2}, this->exec));
    auto b2 = gko::clone(b1);
    b2->scale(gko::initialize<Dense>({-1}, this->exec));
    auto expected = this->template create_recv_connections<double>()[rank];
    auto x1 = Vector::create(this->mpi_exec, this->comm,
                             gko::dim<2>{this->rg->get_size()[0], 1},
                             gko::dim<2>{expected.get_size(), 1});
    auto x2 = gko::clone(x1);
    gko::array<char> workspace1;
    gko::array<char> workspace2;

    auto req1 = this->rg->apply_async(b1, x1, workspace1);
    auto req2 = this->rg->apply_async(b2, x2, workspace2);
    req1.wait();
    req2.wait();

    auto expected_vec1 = Vector::create(
        this->mpi_exec, this->comm, gko::dim<2>{this->rg->get_size()[0], 1},
        Dense::create(this->mpi_exec, gko::dim<2>{expected.get_size(), 1},
                      expected, 1));
    auto expected_vec2 = gko::clone(expected_vec1);
    expected_vec2->scale(gko::initialize<Dense>({-1}, this->exec));
    GKO_ASSERT_MTX_NEAR(x1->get_local_vector(),
                        expected_vec1->get_local_vector(), 0.0);
    GKO_ASSERT_MTX_NEAR(x2->get_local_vector(),
                        expected_vec2->get_local_vector(), 0.0);
}


TYPED_TEST(RowGatherer, CanApplyAsyncWithMultipleColumns)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b = Vector::create(
        this->exec, this->comm, gko::dim<2>{18, 2},
        gko::initialize<Dense>({{offset, offset * offset},
                                {offset + 1, offset * offset + 1},
                                {offset + 2, offset * offset + 2}},
                               this->exec));
    gko::array<double> expected[] = {
        gko::array<double>{this->mpi_exec, {3, 9, 5, 11, 10, 82, 11, 83}},
        gko::array<double>{this->mpi_exec,
                           {0, 0, 1, 1, 7, 37, 12, 144, 13, 145}},
        gko::array<double>{this->mpi_exec, {3, 9, 4, 10, 17, 227}},
        gko::array<double>{this->mpi_exec, {1, 1, 2, 2, 12, 144, 14, 146}},
        gko::array<double>{this->mpi_exec,
                           {4, 10, 5, 11, 9, 81, 10, 82, 15, 225, 16, 226}},
        gko::array<double>{this->mpi_exec, {8, 38, 12, 144, 13, 145, 14, 146}}};
    auto x = Vector::create(this->mpi_exec, this->comm,
                            gko::dim<2>{this->rg->get_size()[0], 2},
                            gko::dim<2>{expected[rank].get_size() / 2, 2});

    this->rg->apply_async(b, x).wait();

    auto expected_vec = Vector::create(
        this->mpi_exec, this->comm, gko::dim<2>{this->rg->get_size()[0], 2},
        Dense::create(this->mpi_exec,
                      gko::dim<2>{expected[rank].get_size() / 2, 2},
                      expected[rank], 2));
    GKO_ASSERT_MTX_NEAR(x->get_local_vector(), expected_vec->get_local_vector(),
                        0.0);
}


TYPED_TEST(RowGatherer, ThrowsOnNonMatchingExecutor)
{
    if (this->mpi_exec == this->exec) {
        GTEST_SKIP();
    }

    using RowGatherer = typename TestFixture::row_gatherer_type;
    using Vector = gko::experimental::distributed::Vector<double>;
    auto rg = RowGatherer::create(this->exec, this->comm);
    auto b = Vector::create(this->exec, this->comm);
    auto x = Vector::create(this->exec, this->comm);

    ASSERT_THROW(rg->apply_async(b, x).wait(), gko::InvalidStateError);
}
