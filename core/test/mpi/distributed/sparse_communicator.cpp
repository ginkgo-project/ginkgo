// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/distributed/sparse_communicator.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/assertions.hpp"


class SparseCommunicator : public ::testing::Test {
protected:
    using part_type = gko::experimental::distributed::Partition<int, long>;
    using map_type = gko::experimental::distributed::index_map<int, long>;
    using Dense = gko::matrix::Dense<>;

    void SetUp()
    {
        rank = comm.rank();
        ASSERT_EQ(comm.size(), 6);

        auto offset = static_cast<double>(rank * 3);
        buffer = gko::initialize<Dense>({offset, offset + 1, offset + 2}, ref);
    }

    std::shared_ptr<gko::Executor> ref = gko::ReferenceExecutor::create();
    gko::experimental::mpi::communicator comm = MPI_COMM_WORLD;
    int rank = -1;

    // globally this is [0, ..., 17]
    std::unique_ptr<Dense> buffer;
    gko::detail::DenseCache<double> recv_buffer;
    gko::detail::DenseCache<double> send_buffer;
};

TEST_F(SparseCommunicator, CanDefaultConstruct)
{
    gko::experimental::distributed::sparse_communicator spcomm{};

    auto empty = Dense::create(ref);
    auto req = spcomm.communicate(empty.get(), send_buffer, recv_buffer);
    req.wait();

    gko::dim<2> zero{};
    GKO_ASSERT_EQUAL_DIMENSIONS(send_buffer.get(), zero);
    GKO_ASSERT_EQUAL_DIMENSIONS(recv_buffer.get(), zero);
}

TEST_F(SparseCommunicator, CanConstructFromIndexMap)
{
    auto part = gko::share(part_type::build_from_global_size_uniform(
        ref, comm.size(), comm.size() * 3));
    gko::array<long> recv_connections[] = {{ref, {3, 5, 10, 11}},
                                           {ref, {0, 1, 7, 12, 13}},
                                           {ref, {3, 4, 17}},
                                           {ref, {1, 2, 12, 14}},
                                           {ref, {4, 5, 9, 10, 15, 16}},
                                           {ref, {8, 12, 13, 14}}};
    auto imap = map_type{ref, part, comm.rank(), recv_connections[comm.rank()]};

    gko::experimental::distributed::sparse_communicator spcomm{comm, imap};

    auto req = spcomm.communicate(buffer.get(), send_buffer, recv_buffer);
    req.wait();
    ASSERT_NE(send_buffer.get(), nullptr);
    ASSERT_NE(recv_buffer.get(), nullptr);
    auto recv_size = recv_connections[rank].get_size();
    gko::size_type send_size[] = {4, 6, 2, 4, 7, 3};
    auto send_dim = gko::dim<2>{send_size[rank], 1};
    auto recv_dim = gko::dim<2>{recv_size, 1};
    GKO_ASSERT_EQUAL_DIMENSIONS(send_buffer.get(), send_dim);
    GKO_ASSERT_EQUAL_DIMENSIONS(recv_buffer.get(), recv_dim);
    // repeat recv_connections, since there is no conversion between long and
    // double
    gko::array<double> values[] = {{ref, {3, 5, 10, 11}},
                                   {ref, {0, 1, 7, 12, 13}},
                                   {ref, {3, 4, 17}},
                                   {ref, {1, 2, 12, 14}},
                                   {ref, {4, 5, 9, 10, 15, 16}},
                                   {ref, {8, 12, 13, 14}}};
    auto expected = Dense::create(ref, recv_dim, values[rank], 1);
    GKO_ASSERT_MTX_NEAR(recv_buffer.get(), expected, 0.0);
}


TEST_F(SparseCommunicator, CanConstructWithHooks)
{
    auto part = gko::share(part_type::build_from_global_size_uniform(
        ref, comm.size(), comm.size() * 3));
    gko::array<long> recv_connections[] = {{ref, {3, 5, 10, 11}},
                                           {ref, {0, 1, 7, 12, 13}},
                                           {ref, {3, 4, 17}},
                                           {ref, {1, 2, 12, 14}},
                                           {ref, {4, 5, 9, 10, 15, 16}},
                                           {ref, {8, 12, 13, 14}}};
    auto imap = map_type{ref, part, comm.rank(), recv_connections[comm.rank()]};

    gko::experimental::distributed::sparse_communicator spcomm{
        comm, imap,
        [this](gko::LinOp* v) {
            gko::as<Dense>(v)->scale(gko::initialize<Dense>({-1.0}, ref));
        },
        [this](gko::LinOp* v) {
            gko::as<Dense>(v)->scale(gko::initialize<Dense>({2}, ref));
        }};

    auto req = spcomm.communicate(buffer.get(), send_buffer, recv_buffer);
    req.wait();
    ASSERT_NE(send_buffer.get(), nullptr);
    ASSERT_NE(recv_buffer.get(), nullptr);
    auto recv_size = recv_connections[rank].get_size();
    gko::size_type send_size[] = {4, 6, 2, 4, 7, 3};
    auto send_dim = gko::dim<2>{send_size[rank], 1};
    auto recv_dim = gko::dim<2>{recv_size, 1};
    GKO_ASSERT_EQUAL_DIMENSIONS(send_buffer.get(), send_dim);
    GKO_ASSERT_EQUAL_DIMENSIONS(recv_buffer.get(), recv_dim);
    // repeat recv_connections, since there is no conversion between long and
    // double
    gko::array<double> values[] = {{ref, {-6, -10, -20, -22}},
                                   {ref, {0, -2, -14, -24, -26}},
                                   {ref, {-6, -8, -34}},
                                   {ref, {-2, -4, -24, -28}},
                                   {ref, {-8, -10, -18, -20, -30, -32}},
                                   {ref, {-16, -24, -26, -28}}};
    auto expected = Dense::create(ref, recv_dim, values[rank], 1);
    GKO_ASSERT_MTX_NEAR(recv_buffer.get(), expected, 0.0);
}
