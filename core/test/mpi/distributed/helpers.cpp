// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/distributed/helpers.hpp"
#include "core/test/utils.hpp"


int run_function(gko::experimental::distributed::Vector<>*) { return 1; }

int run_function(const gko::experimental::distributed::Vector<>*) { return 2; }

int run_function(gko::matrix::Dense<>*) { return 3; }

int run_function(const gko::matrix::Dense<>*) { return 4; }


class RunVector : public ::testing::Test {
public:
    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
};


TEST_F(RunVector, PicksDistributedVectorCorrectly)
{
    std::unique_ptr<gko::LinOp> dist_vector =
        gko::experimental::distributed::Vector<>::create(exec, MPI_COMM_WORLD);
    int result;

    gko::detail::vector_dispatch<double>(
        dist_vector.get(), [&](auto* dense) { result = run_function(dense); });

    ASSERT_EQ(result,
              run_function(gko::as<gko::experimental::distributed::Vector<>>(
                  dist_vector.get())));
}


TEST_F(RunVector, PicksConstDistributedVectorCorrectly)
{
    std::unique_ptr<const gko::LinOp> const_dist_vector =
        gko::experimental::distributed::Vector<>::create(exec, MPI_COMM_WORLD);
    int result;

    gko::detail::vector_dispatch<double>(
        const_dist_vector.get(),
        [&](auto* dense) { result = run_function(dense); });

    ASSERT_EQ(
        result,
        run_function(gko::as<const gko::experimental::distributed::Vector<>>(
            const_dist_vector.get())));
}


TEST_F(RunVector, PicksDenseVectorCorrectly)
{
    std::unique_ptr<gko::LinOp> dense_vector =
        gko::matrix::Dense<>::create(exec);
    int result;

    gko::detail::vector_dispatch<double>(
        dense_vector.get(), [&](auto* dense) { result = run_function(dense); });

    ASSERT_EQ(result,
              run_function(gko::as<gko::matrix::Dense<>>(dense_vector.get())));
}


TEST_F(RunVector, PicksConstDenseVectorCorrectly)
{
    std::unique_ptr<const gko::LinOp> const_dense_vector =
        gko::matrix::Dense<>::create(exec);
    int result;

    gko::detail::vector_dispatch<double>(
        const_dense_vector.get(),
        [&](auto* dense) { result = run_function(dense); });

    ASSERT_EQ(result, run_function(gko::as<const gko::matrix::Dense<>>(
                          const_dense_vector.get())));
}

TEST_F(RunVector, ThrowsIfWrongType)
{
    std::unique_ptr<gko::LinOp> csr = gko::matrix::Csr<>::create(exec);

    ASSERT_THROW(
        gko::detail::vector_dispatch<double>(csr.get(), [&](auto* dense) {}),
        gko::NotSupported);
}


TEST_F(RunVector, ThrowsIfNullptr)
{
    ASSERT_THROW(gko::detail::vector_dispatch<double>(
                     static_cast<gko::LinOp*>(nullptr), [&](auto* dense) {}),
                 gko::NotSupported);
}
