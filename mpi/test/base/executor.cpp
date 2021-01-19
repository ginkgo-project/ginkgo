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


#include <memory>
#include <type_traits>


#include <mpi.h>


#include <gtest/gtest.h>


#include "gtest-mpi-listener.hpp"
#include "gtest-mpi-main.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>

class MpiExecutor : public ::testing::Test {
protected:
    MpiExecutor() : mpi(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
        mpi = gko::MpiExecutor::create(gko::ReferenceExecutor::create());
    }

    void TearDown()
    {
        if (mpi != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(mpi->synchronize());
        }
    }

    std::shared_ptr<gko::MpiExecutor> mpi;
};


TEST_F(MpiExecutor, KnowsItsSubExecutors)
{
    auto sub_exec = mpi->get_sub_executor();
    auto omp = gko::OmpExecutor::create();
    auto ref = gko::ReferenceExecutor::create();

    EXPECT_NE(typeid(*(omp.get())).name(), typeid(*(sub_exec.get())).name());
    EXPECT_EQ(typeid(*(ref.get())).name(), typeid(*(sub_exec.get())).name());
}


TEST_F(MpiExecutor, KnowsItsCommunicator)
{
    auto comm_world = gko::mpi::communicator(MPI_COMM_WORLD);
    EXPECT_EQ(comm_world.compare(mpi->get_communicator()), true);
}


TEST_F(MpiExecutor, KnowsItsSize)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    EXPECT_EQ(mpi->get_num_ranks(MPI_COMM_WORLD), size);
}


TEST_F(MpiExecutor, KnowsItsRanks)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    EXPECT_EQ(rank, mpi->get_my_rank(MPI_COMM_WORLD));
}


TEST_F(MpiExecutor, CanSetADefaultWindow)
{
    gko::mpi::window<int> win;
    ASSERT_EQ(win.get(), nullptr);
}


TEST_F(MpiExecutor, CanCreateWindow)
{
    using ValueType = int;
    ValueType *data;
    data = new ValueType[4]{1, 2, 3, 4};
    gko::mpi::window<ValueType> win(data, 4 * sizeof(ValueType));
    ASSERT_NE(win.get(), nullptr);
    delete data;
}


TEST_F(MpiExecutor, CanSendAndRecvValues)
{
    using ValueType = int;
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    auto send_array = gko::Array<ValueType>{sub_exec};
    auto recv_array = gko::Array<ValueType>{sub_exec};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
        send_array = gko::Array<ValueType>{
            sub_exec, gko::Array<ValueType>::view(sub_exec, 4, data)};
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                mpi->send<ValueType>(send_array.get_data(), 4, rank, 40 + rank);
            }
        }
    } else {
        recv_array = gko::Array<ValueType>{sub_exec, 4};
        mpi->recv<ValueType>(recv_array.get_data(), 4, 0, 40 + my_rank);
    }
    if (my_rank != 0) {
        ASSERT_EQ(recv_array.get_data()[0], 1);
        ASSERT_EQ(recv_array.get_data()[1], 2);
        ASSERT_EQ(recv_array.get_data()[2], 3);
        ASSERT_EQ(recv_array.get_data()[3], 4);
    }
    if (my_rank == 0) {
        delete data;
    }
}


TEST_F(MpiExecutor, CanNonBlockingSendAndNonBlockingRecvValues)
{
    using ValueType = int;
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    auto send_array = gko::Array<ValueType>{sub_exec};
    auto recv_array = gko::Array<ValueType>{sub_exec};
    int *data;
    auto req = mpi->create_requests_array(num_ranks);
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
        send_array = gko::Array<ValueType>{
            sub_exec, gko::Array<ValueType>::view(sub_exec, 4, data)};
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                mpi->send<ValueType>(send_array.get_data(), 4, rank, 40 + rank,
                                     req.get());
            }
        }
    } else {
        recv_array = gko::Array<ValueType>{sub_exec, 4};
        mpi->recv<ValueType>(recv_array.get_data(), 4, 0, 40 + my_rank,
                             req.get());
    }
    mpi->wait(req.get());
    if (my_rank != 0) {
        ASSERT_EQ(recv_array.get_data()[0], 1);
        ASSERT_EQ(recv_array.get_data()[1], 2);
        ASSERT_EQ(recv_array.get_data()[2], 3);
        ASSERT_EQ(recv_array.get_data()[3], 4);
    }
    if (my_rank == 0) {
        delete data;
    }
}


TEST_F(MpiExecutor, CanPutValuesWithLockAll)
{
    using ValueType = int;
    using Window = gko::mpi::window<ValueType>;
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    auto send_array = gko::Array<ValueType>{sub_exec};
    auto recv_array = gko::Array<ValueType>{sub_exec};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
        send_array = gko::Array<ValueType>{
            sub_exec, gko::Array<ValueType>::view(sub_exec, 4, data)};
    } else {
        recv_array = gko::Array<ValueType>{sub_exec, 4};
    }
    auto win = Window(recv_array.get_data(), 4 * sizeof(ValueType));
    win.lock_all();
    if (my_rank == 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                mpi->put<ValueType>(send_array.get_const_data(), 4, rank, 0, 4,
                                    win.get());
                win.flush(rank);
            }
        }
    }
    win.unlock_all();
    mpi->synchronize();
    if (my_rank != 0) {
        ASSERT_EQ(recv_array.get_data()[0], 1);
        ASSERT_EQ(recv_array.get_data()[1], 2);
        ASSERT_EQ(recv_array.get_data()[2], 3);
        ASSERT_EQ(recv_array.get_data()[3], 4);
    }
    if (my_rank == 0) {
        delete data;
    }
}


TEST_F(MpiExecutor, CanPutValuesWithExclusiveLock)
{
    using ValueType = int;
    using Window = gko::mpi::window<ValueType>;
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    auto send_array = gko::Array<ValueType>{sub_exec};
    auto recv_array = gko::Array<ValueType>{sub_exec};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
    } else {
        data = new ValueType[4]{0, 0, 0, 0};
    }
    auto win = Window(data, 4 * sizeof(ValueType), sizeof(ValueType),
                      MPI_INFO_NULL, comm, Window::win_type::create);
    if (my_rank == 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                win.lock(rank, 0, Window::lock_type::exclusive);
                mpi->put<ValueType>(data, 4, rank, 0, 4, win.get());
                win.flush(rank);
                win.unlock(rank);
            }
        }
    }
    mpi->synchronize();
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 2);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 4);
    delete data;
}


TEST_F(MpiExecutor, CanPutValuesWithFence)
{
    using ValueType = int;
    using Window = gko::mpi::window<ValueType>;
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    auto send_array = gko::Array<ValueType>{sub_exec};
    auto recv_array = gko::Array<ValueType>{sub_exec};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
        send_array = gko::Array<ValueType>{
            sub_exec, gko::Array<ValueType>::view(sub_exec, 4, data)};
    } else {
        recv_array = gko::Array<ValueType>{sub_exec, 4};
    }
    auto win = Window(recv_array.get_data(), 4 * sizeof(ValueType));
    win.fence();
    if (my_rank == 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                mpi->put<ValueType>(send_array.get_const_data(), 4, rank, 0, 4,
                                    win.get());
            }
        }
    }
    win.fence();
    mpi->synchronize();
    if (my_rank != 0) {
        ASSERT_EQ(recv_array.get_data()[0], 1);
        ASSERT_EQ(recv_array.get_data()[1], 2);
        ASSERT_EQ(recv_array.get_data()[2], 3);
        ASSERT_EQ(recv_array.get_data()[3], 4);
    }
    if (my_rank == 0) {
        delete data;
    }
}


TEST_F(MpiExecutor, CanGetValuesWithLockAll)
{
    using ValueType = int;
    using Window = gko::mpi::window<ValueType>;
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    auto send_array = gko::Array<ValueType>{sub_exec};
    auto recv_array = gko::Array<ValueType>{sub_exec};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
    } else {
        data = new ValueType[4]{0, 0, 0, 0};
    }
    auto win = Window(data, 4 * sizeof(ValueType), sizeof(ValueType),
                      MPI_INFO_NULL, comm, Window::win_type::create);
    if (my_rank != 0) {
        win.lock_all();
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                mpi->get<ValueType>(data, 4, 0, 0, 4, win.get());
                win.flush(0);
            }
        }
        win.unlock_all();
    }
    mpi->synchronize();
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 2);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 4);
    delete data;
}


TEST_F(MpiExecutor, CanGetValuesWithExclusiveLock)
{
    using ValueType = int;
    using Window = gko::mpi::window<ValueType>;
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    auto send_array = gko::Array<ValueType>{sub_exec};
    auto recv_array = gko::Array<ValueType>{sub_exec};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
    } else {
        data = new ValueType[4]{0, 0, 0, 0};
    }
    auto win = Window(data, 4 * sizeof(ValueType), sizeof(ValueType),
                      MPI_INFO_NULL, comm, Window::win_type::create);
    if (my_rank != 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                win.lock(0, 0, Window::lock_type::exclusive);
                mpi->get<ValueType>(data, 4, 0, 0, 4, win.get());
                win.flush(0);
                win.unlock(0);
            }
        }
    }
    mpi->synchronize();
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 2);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 4);
    delete data;
}


TEST_F(MpiExecutor, CanGetValuesWithFence)
{
    using ValueType = int;
    using Window = gko::mpi::window<ValueType>;
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    auto send_array = gko::Array<ValueType>{sub_exec};
    auto recv_array = gko::Array<ValueType>{sub_exec};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
    } else {
        data = new ValueType[4]{0, 0, 0, 0};
    }
    auto win = Window(data, 4 * sizeof(ValueType), sizeof(ValueType),
                      MPI_INFO_NULL, comm, Window::win_type::create);
    win.fence();
    if (my_rank != 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                mpi->get<ValueType>(data, 4, 0, 0, 4, win.get());
            }
        }
    }
    win.fence();
    mpi->synchronize();
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 2);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 4);
    delete data;
}


TEST_F(MpiExecutor, CanBroadcastValues)
{
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    double *data;
    auto array = gko::Array<double>{sub_exec, 8};
    if (my_rank == 0) {
        // clang-format off
        data = new double[8]{ 2.0, 3.0, 1.0,
                3.0,-1.0, 0.0 , 3.5, 1.5};
        // clang-format on
        array = gko::Array<double>{gko::Array<double>::view(sub_exec, 8, data)};
    }
    mpi->broadcast<double>(array.get_data(), 8, 0);
    auto comp_data = array.get_data();
    ASSERT_EQ(comp_data[0], 2.0);
    ASSERT_EQ(comp_data[1], 3.0);
    ASSERT_EQ(comp_data[2], 1.0);
    ASSERT_EQ(comp_data[3], 3.0);
    ASSERT_EQ(comp_data[4], -1.0);
    ASSERT_EQ(comp_data[5], 0.0);
    ASSERT_EQ(comp_data[6], 3.5);
    ASSERT_EQ(comp_data[7], 1.5);
    if (my_rank == 0) {
        delete data;
    }
}


TEST_F(MpiExecutor, CanReduceValues)
{
    using ValueType = double;
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    ValueType data, sum, max, min;
    if (my_rank == 0) {
        data = 3;
    } else if (my_rank == 1) {
        data = 5;
    } else if (my_rank == 2) {
        data = 2;
    } else if (my_rank == 3) {
        data = 6;
    }
    mpi->reduce<ValueType>(&data, &sum, 1, gko::mpi::op_type::sum, 0);
    mpi->reduce<ValueType>(&data, &max, 1, gko::mpi::op_type::max, 0);
    mpi->reduce<ValueType>(&data, &min, 1, gko::mpi::op_type::min, 0);
    if (my_rank == 0) {
        EXPECT_EQ(sum, 16.0);
        EXPECT_EQ(max, 6.0);
        EXPECT_EQ(min, 2.0);
    }
}


TEST_F(MpiExecutor, CanAllReduceValues)
{
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    int data, sum;
    if (my_rank == 0) {
        data = 3;
    } else if (my_rank == 1) {
        data = 5;
    } else if (my_rank == 2) {
        data = 2;
    } else if (my_rank == 3) {
        data = 6;
    }
    mpi->all_reduce<int>(&data, &sum, 1, gko::mpi::op_type::sum);
    ASSERT_EQ(sum, 16);
}


TEST_F(MpiExecutor, CanScatterValues)
{
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    double *data;
    auto scatter_from_array = gko::Array<double>{sub_exec->get_master()};
    if (my_rank == 0) {
        // clang-format off
        data = new double[8]{ 2.0, 3.0, 1.0,
                3.0,-1.0, 0.0 , 3.5, 1.5};
        // clang-format on
        scatter_from_array =
            gko::Array<double>{sub_exec->get_master(),
                               gko::Array<double>::view(sub_exec, 8, data)};
    }
    auto scatter_into_array = gko::Array<double>{sub_exec, 2};
    mpi->scatter<double, double>(scatter_from_array.get_data(), 2,
                                 scatter_into_array.get_data(), 2, 0);
    auto comp_data = scatter_into_array.get_data();
    if (my_rank == 0) {
        ASSERT_EQ(comp_data[0], 2.0);
        ASSERT_EQ(comp_data[1], 3.0);
        delete data;
    } else if (my_rank == 1) {
        ASSERT_EQ(comp_data[0], 1.0);
        ASSERT_EQ(comp_data[1], 3.0);
    } else if (my_rank == 2) {
        ASSERT_EQ(comp_data[0], -1.0);
        ASSERT_EQ(comp_data[1], 0.0);
    } else if (my_rank == 3) {
        ASSERT_EQ(comp_data[0], 3.5);
        ASSERT_EQ(comp_data[1], 1.5);
    }
}


TEST_F(MpiExecutor, CanGatherValues)
{
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    int data;
    if (my_rank == 0) {
        data = 3;
    } else if (my_rank == 1) {
        data = 5;
    } else if (my_rank == 2) {
        data = 2;
    } else if (my_rank == 3) {
        data = 6;
    }
    auto gather_array =
        gko::Array<int>{sub_exec, static_cast<gko::size_type>(num_ranks)};
    mpi->gather<int, int>(&data, 1, gather_array.get_data(), 1, 0);
    if (my_rank == 0) {
        ASSERT_EQ(gather_array.get_data()[0], 3);
        ASSERT_EQ(gather_array.get_data()[1], 5);
        ASSERT_EQ(gather_array.get_data()[2], 2);
        ASSERT_EQ(gather_array.get_data()[3], 6);
    }
}


TEST_F(MpiExecutor, CanScatterValuesWithDisplacements)
{
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    double *data;
    auto scatter_from_array = gko::Array<double>{sub_exec};
    auto scatter_into_array = gko::Array<double>{sub_exec};
    auto s_counts = gko::Array<int>{sub_exec->get_master(),
                                    static_cast<gko::size_type>(num_ranks)};
    auto displacements = gko::Array<int>{sub_exec->get_master()};
    int nelems;
    if (my_rank == 0) {
        // clang-format off
        data = new double[10]{ 2.0, 3.0, 1.0,
                3.0,-1.0, 0.0,
                2.5,-1.5, 0.5, 3.5};
        // clang-format on
        scatter_from_array = gko::Array<double>{
            sub_exec, gko::Array<double>::view(sub_exec, 10, data)};
        nelems = 2;
        displacements = gko::Array<int>{sub_exec, {0, 2, 6, 9}};
    } else if (my_rank == 1) {
        nelems = 4;
    } else if (my_rank == 2) {
        nelems = 3;
    } else if (my_rank == 3) {
        nelems = 1;
    }
    scatter_into_array =
        gko::Array<double>{sub_exec, static_cast<gko::size_type>(nelems)};
    mpi->gather<int, int>(&nelems, 1, s_counts.get_data(), 1, 0);
    mpi->scatter<double, double>(scatter_from_array.get_data(),
                                 s_counts.get_data(), displacements.get_data(),
                                 scatter_into_array.get_data(), nelems, 0);
    auto comp_data = scatter_into_array.get_data();
    if (my_rank == 0) {
        ASSERT_EQ(comp_data[0], 2.0);
        ASSERT_EQ(comp_data[1], 3.0);
        delete data;
    } else if (my_rank == 1) {
        ASSERT_EQ(comp_data[0], 1.0);
        ASSERT_EQ(comp_data[1], 3.0);
        ASSERT_EQ(comp_data[2], -1.0);
        ASSERT_EQ(comp_data[3], 0.0);
    } else if (my_rank == 2) {
        ASSERT_EQ(comp_data[0], 2.5);
        ASSERT_EQ(comp_data[1], -1.5);
        ASSERT_EQ(comp_data[2], 0.5);
    } else if (my_rank == 3) {
        ASSERT_EQ(comp_data[0], 3.5);
    }
}


TEST_F(MpiExecutor, CanGatherValuesWithDisplacements)
{
    auto sub_exec = mpi->get_sub_executor();
    auto comm = mpi->get_communicator();
    auto my_rank = mpi->get_my_rank(comm);
    auto num_ranks = mpi->get_num_ranks(comm);
    double *data;
    auto gather_from_array = gko::Array<double>{sub_exec};
    auto gather_into_array = gko::Array<double>{sub_exec};
    auto r_counts = gko::Array<int>{sub_exec->get_master(),
                                    static_cast<gko::size_type>(num_ranks)};
    auto displacements = gko::Array<int>{sub_exec->get_master()};
    int nelems;
    if (my_rank == 0) {
        data = new double[2]{2.0, 3.0};
        gather_from_array = gko::Array<double>{
            sub_exec->get_master(),
            gko::Array<double>::view(sub_exec->get_master(), 2, data)};
        nelems = 2;
        displacements = gko::Array<int>{sub_exec->get_master(), {0, 2, 6, 7}};
        gather_into_array = gko::Array<double>{sub_exec, 10};
    } else if (my_rank == 1) {
        data = new double[4]{1.5, 2.0, 1.0, 0.5};
        nelems = 4;
        gather_from_array = gko::Array<double>{
            sub_exec->get_master(),
            gko::Array<double>::view(sub_exec->get_master(), 4, data)};
    } else if (my_rank == 2) {
        data = new double[1]{1.0};
        nelems = 1;
        gather_from_array = gko::Array<double>{
            sub_exec->get_master(),
            gko::Array<double>::view(sub_exec->get_master(), 1, data)};
    } else if (my_rank == 3) {
        data = new double[3]{1.9, -4.0, 5.0};
        nelems = 3;
        gather_from_array = gko::Array<double>{
            sub_exec->get_master(),
            gko::Array<double>::view(sub_exec->get_master(), 3, data)};
    }

    mpi->gather<int, int>(&nelems, 1, r_counts.get_data(), 1, 0);
    mpi->gather<double, double>(
        gather_from_array.get_data(), nelems, gather_into_array.get_data(),
        r_counts.get_data(), displacements.get_data(), 0);
    auto comp_data = gather_into_array.get_data();
    if (my_rank == 0) {
        ASSERT_EQ(comp_data[0], 2.0);
        ASSERT_EQ(comp_data[1], 3.0);
        ASSERT_EQ(comp_data[2], 1.5);
        ASSERT_EQ(comp_data[3], 2.0);
        ASSERT_EQ(comp_data[4], 1.0);
        ASSERT_EQ(comp_data[5], 0.5);
        ASSERT_EQ(comp_data[6], 1.0);
        ASSERT_EQ(comp_data[7], 1.9);
        ASSERT_EQ(comp_data[8], -4.0);
        ASSERT_EQ(comp_data[9], 5.0);
    } else {
        ASSERT_EQ(comp_data, nullptr);
    }
    delete data;
}


// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
