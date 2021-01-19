/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#include <mpi.h>

#include <gtest/gtest.h>

#include "gtest-mpi-listener.hpp"
#include "gtest-mpi-main.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


class Communicator : public ::testing::Test {
protected:
    Communicator() : mpi_exec(nullptr), mpi_exec2(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
        mpi_exec = gko::MpiExecutor::create(gko::ReferenceExecutor::create());
        sub_exec = mpi_exec->get_sub_executor();
        rank = mpi_exec->get_my_rank(mpi_exec->get_communicator());
        ASSERT_EQ(mpi_exec->get_num_ranks(mpi_exec->get_communicator()), 8);
    }

    void TearDown()
    {
        if (mpi_exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(mpi_exec->synchronize());
        }
    }

    std::shared_ptr<gko::MpiExecutor> mpi_exec;
    std::shared_ptr<gko::MpiExecutor> mpi_exec2;
    std::shared_ptr<const gko::Executor> sub_exec;
    int rank;
};


TEST_F(Communicator, KnowsItsDefaultCommunicator)
{
    auto comm_world = gko::mpi::communicator(MPI_COMM_WORLD);
    EXPECT_EQ(comm_world.compare(this->mpi_exec->get_communicator()), true);
}


TEST_F(Communicator, KnowsNumRanks)
{
    EXPECT_EQ(this->mpi_exec->get_num_ranks(mpi_exec->get_communicator()), 8);
}


TEST_F(Communicator, CanSetCustomCommunicator)
{
    auto world_rank = mpi_exec->get_my_rank(mpi_exec->get_communicator());
    auto world_size = mpi_exec->get_num_ranks(mpi_exec->get_communicator());
    auto color = world_rank / 4;

    auto main_comm = mpi_exec->get_communicator();
    auto row_comm = gko::mpi::communicator(main_comm, color, world_rank);
    mpi_exec2 = gko::MpiExecutor::create(gko::ReferenceExecutor::create(),
                                         row_comm.get());
    for (auto i = 0; i < world_size; ++i) {
        EXPECT_LT(mpi_exec2->get_my_rank(mpi_exec2->get_communicator()), 4);
    }
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
