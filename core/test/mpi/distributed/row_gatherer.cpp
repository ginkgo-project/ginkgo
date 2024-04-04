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

    RowGatherer()
    {
        int rank = this->comm.rank();
        auto part = gko::share(part_type::build_from_global_size_uniform(
            this->ref, this->comm.size(), this->comm.size() * 3));
        auto recv_connections =
            this->template create_recv_connections<long>()[rank];
        auto imap =
            map_type{this->ref, part, this->comm.rank(), recv_connections};
        auto coll_comm =
            std::make_shared<gko::experimental::mpi::NeighborhoodCommunicator>(
                this->comm, imap);
        rg = row_gatherer_type::create(ref, coll_comm, imap);
    }

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
    std::shared_ptr<row_gatherer_type> rg;
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


TYPED_TEST(RowGatherer, CanApply)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::initialize<Dense>({offset, offset + 1, offset + 2}, this->ref));
    auto x = Dense::create(this->ref, gko::dim<2>{this->rg->get_size()[0], 1});

    this->rg->apply(b, x);

    auto expected = this->template create_recv_connections<double>()[rank];
    auto expected_vec = Dense::create(
        this->ref, gko::dim<2>{expected.get_size(), 1}, expected, 1);
    GKO_ASSERT_MTX_NEAR(x, expected_vec, 0.0);
}


TYPED_TEST(RowGatherer, CanApplyAsync)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::initialize<Dense>({offset, offset + 1, offset + 2}, this->ref));
    auto x = Dense::create(this->ref, gko::dim<2>{this->rg->get_size()[0], 1});

    auto req = this->rg->apply_async(b, x);
    req.wait();

    auto expected = this->template create_recv_connections<double>()[rank];
    auto expected_vec = Dense::create(
        this->ref, gko::dim<2>{expected.get_size(), 1}, expected, 1);
    GKO_ASSERT_MTX_NEAR(x, expected_vec, 0.0);
}


TYPED_TEST(RowGatherer, CanApplyAsyncConsequetively)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::initialize<Dense>({offset, offset + 1, offset + 2}, this->ref));
    auto x = Dense::create(this->ref, gko::dim<2>{this->rg->get_size()[0], 1});

    this->rg->apply_async(b, x).wait();
    this->rg->apply_async(b, x).wait();

    auto expected = this->template create_recv_connections<double>()[rank];
    auto expected_vec = Dense::create(
        this->ref, gko::dim<2>{expected.get_size(), 1}, expected, 1);
    GKO_ASSERT_MTX_NEAR(x, expected_vec, 0.0);
}


TYPED_TEST(RowGatherer, CanApplyAsyncWithWorkspace)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::initialize<Dense>({offset, offset + 1, offset + 2}, this->ref));
    auto x = Dense::create(this->ref, gko::dim<2>{this->rg->get_size()[0], 1});
    gko::array<char> workspace(this->ref);

    auto req = this->rg->apply_async(b, x, workspace);
    req.wait();

    auto expected = this->template create_recv_connections<double>()[rank];
    auto expected_vec = Dense::create(
        this->ref, gko::dim<2>{expected.get_size(), 1}, expected, 1);
    GKO_ASSERT_MTX_NEAR(x, expected_vec, 0.0);
}


TYPED_TEST(RowGatherer, CanApplyAsyncMultipleTimesWithWorkspace)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b1 = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 1},
        gko::initialize<Dense>({offset, offset + 1, offset + 2}, this->ref));
    auto b2 = gko::clone(b1);
    b2->scale(gko::initialize<Dense>({-1}, this->ref));
    auto x1 = Dense::create(this->ref, gko::dim<2>{this->rg->get_size()[0], 1});
    auto x2 = gko::clone(x1);
    gko::array<char> workspace1(this->ref);
    gko::array<char> workspace2(this->ref);

    auto req1 = this->rg->apply_async(b1, x1, workspace1);
    auto req2 = this->rg->apply_async(b2, x2, workspace2);
    req1.wait();
    req2.wait();

    auto expected = this->template create_recv_connections<double>()[rank];
    auto expected_vec1 = Dense::create(
        this->ref, gko::dim<2>{expected.get_size(), 1}, expected, 1);
    auto expected_vec2 = gko::clone(expected_vec1);
    expected_vec2->scale(gko::initialize<Dense>({-1}, this->ref));
    GKO_ASSERT_MTX_NEAR(x1, expected_vec1, 0.0);
    GKO_ASSERT_MTX_NEAR(x2, expected_vec2, 0.0);
}


TYPED_TEST(RowGatherer, CanApplyAsyncWithMultipleColumns)
{
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    int rank = this->comm.rank();
    auto offset = static_cast<double>(rank * 3);
    auto b = Vector::create(
        this->ref, this->comm, gko::dim<2>{18, 2},
        gko::initialize<Dense>({{offset, offset * offset},
                                {offset + 1, offset * offset + 1},
                                {offset + 2, offset * offset + 2}},
                               this->ref));
    auto x = Dense::create(this->ref, gko::dim<2>{this->rg->get_size()[0], 2});

    this->rg->apply_async(b, x).wait();

    gko::array<double> expected[] = {
        gko::array<double>{this->ref, {3, 9, 5, 11, 10, 82, 11, 83}},
        gko::array<double>{this->ref, {0, 0, 1, 1, 7, 37, 12, 144, 13, 145}},
        gko::array<double>{this->ref, {3, 9, 4, 10, 17, 227}},
        gko::array<double>{this->ref, {1, 1, 2, 2, 12, 144, 14, 146}},
        gko::array<double>{this->ref,
                           {4, 10, 5, 11, 9, 81, 10, 82, 15, 225, 16, 226}},
        gko::array<double>{this->ref, {8, 38, 12, 144, 13, 145, 14, 146}}};
    auto expected_vec =
        Dense::create(this->ref, gko::dim<2>{expected[rank].get_size() / 2, 2},
                      expected[rank], 2);
    GKO_ASSERT_MTX_NEAR(x, expected_vec, 0.0);
}


TYPED_TEST(RowGatherer, ThrowsOnAdvancedApply)
{
    using RowGatherer = typename TestFixture::row_gatherer_type;
    using Dense = gko::matrix::Dense<double>;
    using Vector = gko::experimental::distributed::Vector<double>;
    auto rg = RowGatherer::create(this->ref, this->comm);
    auto b = Vector::create(this->ref, this->comm);
    auto x = Dense::create(this->ref);
    auto alpha = Dense::create(this->ref, gko::dim<2>{1, 1});
    auto beta = Dense::create(this->ref, gko::dim<2>{1, 1});

    ASSERT_THROW(rg->apply(alpha, b, beta, x), gko::NotImplemented);
}
