// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/distributed/dense_communicator.hpp>
#include <ginkgo/core/distributed/row_scatterer.hpp>

#include "core/test/utils.hpp"


template <typename IndexType>
class RowScatterer : public ::testing::Test {
protected:
    using row_scatterer_type =
        gko::experimental::distributed::RowScatterer<IndexType>;

    void SetUp() override { ASSERT_EQ(comm.size(), 6); }

    std::shared_ptr<gko::Executor> ref = gko::ReferenceExecutor::create();
    gko::experimental::mpi::communicator comm = MPI_COMM_WORLD;
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
