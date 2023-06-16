/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
