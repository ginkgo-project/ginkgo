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
#include <ginkgo/core/base/mpi.hpp>

class MpiBindings : public ::testing::Test {
protected:
    MpiBindings() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<gko::Executor> ref;

    void assert_equal_arrays(gko::Array<double> &array_1,
                             gko::Array<double> &array_2)
    {
        ASSERT_EQ(array_1.get_num_elems(), array_2.get_num_elems());
        for (gko::size_type i = 0; i < array_1.get_num_elems(); ++i) {
            EXPECT_EQ(array_1.get_const_data()[i], array_2.get_const_data()[i]);
        }
    }
};


TEST_F(MpiBindings, CanSetADefaultWindow)
{
    gko::mpi::window<int> win;
    ASSERT_EQ(win.get(), nullptr);
}


TEST_F(MpiBindings, CanCreateWindow)
{
    using ValueType = int;
    ValueType *data;
    data = new ValueType[4]{1, 2, 3, 4};
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto win = gko::mpi::window<ValueType>(data, 4 * sizeof(ValueType), comm);
    ASSERT_NE(win.get(), nullptr);
    win.lock_all();
    win.unlock_all();
    delete data;
}


TEST_F(MpiBindings, CanSendAndRecvValues)
{
    using ValueType = int;
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm);
    auto num_ranks = gko::mpi::get_num_ranks(comm);
    auto send_array = gko::Array<ValueType>{ref};
    auto recv_array = gko::Array<ValueType>{ref};
    ValueType *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
        send_array =
            gko::Array<ValueType>{ref, gko::Array<ValueType>(ref, 4, data)};
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::send<ValueType>(send_array.get_const_data(), 4, rank,
                                          40 + rank);
            }
        }
    } else {
        recv_array = gko::Array<ValueType>{ref, 4};
        gko::mpi::recv<ValueType>(recv_array.get_data(), 4, 0, 40 + my_rank);
    }
    if (my_rank != 0) {
        ASSERT_EQ(recv_array.get_data()[0], 1);
        ASSERT_EQ(recv_array.get_data()[1], 2);
        ASSERT_EQ(recv_array.get_data()[2], 3);
        ASSERT_EQ(recv_array.get_data()[3], 4);
    }
}


TEST_F(MpiBindings, CanNonBlockingSendAndNonBlockingRecvValues)
{
    using ValueType = int;
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm);
    auto num_ranks = gko::mpi::get_num_ranks(comm);
    auto send_array = gko::Array<ValueType>{ref};
    auto recv_array = gko::Array<ValueType>{ref};
    ValueType *data;
    auto req = gko::mpi::request::create(num_ranks);
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
        send_array =
            gko::Array<ValueType>{ref, gko::Array<ValueType>(ref, 4, data)};
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::send<ValueType>(send_array.get_data(), 4, rank,
                                          40 + rank, req);
            }
        }
    } else {
        recv_array = gko::Array<ValueType>{ref, 4};
        gko::mpi::recv<ValueType>(recv_array.get_data(), 4, 0, 40 + my_rank,
                                  req);
    }
    gko::mpi::wait(req);
    if (my_rank != 0) {
        ASSERT_EQ(recv_array.get_data()[0], 1);
        ASSERT_EQ(recv_array.get_data()[1], 2);
        ASSERT_EQ(recv_array.get_data()[2], 3);
        ASSERT_EQ(recv_array.get_data()[3], 4);
    }
}


TEST_F(MpiBindings, CanPutValuesWithLockAll)
{
    using ValueType = int;
    using window = gko::mpi::window<ValueType>;
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = comm->rank();
    auto num_ranks = comm->size();
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
    } else {
        data = new ValueType[4]{0, 0, 0, 0};
    }
    auto win = window(data, 4 * sizeof(ValueType), comm);
    win.lock_all();
    if (my_rank == 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::put<ValueType>(data, 4, rank, 0, 4, win);
                win.flush(rank);
            }
        }
    }
    win.unlock_all();
    gko::mpi::synchronize();
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 2);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 4);
    delete data;
}


TEST_F(MpiBindings, CanPutValuesWithExclusiveLock)
{
    using ValueType = int;
    using window = gko::mpi::window<ValueType>;
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = comm->rank();
    auto num_ranks = comm->size();
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
    } else {
        data = new ValueType[4]{0, 0, 0, 0};
    }
    auto win = window(data, 4 * sizeof(ValueType), comm);
    if (my_rank == 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                win.lock(rank, 0, window::lock_type::exclusive);
                gko::mpi::put<ValueType>(data, 4, rank, 0, 4, win);
                win.flush(rank);
                win.unlock(rank);
            }
        }
    }
    gko::mpi::synchronize();
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 2);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 4);
    delete data;
}


TEST_F(MpiBindings, CanPutValuesWithFence)
{
    using ValueType = int;
    using window = gko::mpi::window<ValueType>;
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = comm->rank();
    auto num_ranks = comm->size();
    auto send_array = gko::Array<ValueType>{ref};
    auto recv_array = gko::Array<ValueType>{ref};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
    } else {
        data = new ValueType[4]{0, 0, 0, 0};
    }
    auto win = window(data, 4 * sizeof(ValueType), comm);
    win.fence();
    if (my_rank == 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::put<ValueType>(data, 4, rank, 0, 4, win);
            }
        }
    }
    win.fence();
    gko::mpi::synchronize();
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 2);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 4);
    delete data;
}


TEST_F(MpiBindings, CanGetValuesWithLockAll)
{
    using ValueType = int;
    using Window = gko::mpi::window<ValueType>;
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = comm->rank();
    auto num_ranks = comm->size();
    auto send_array = gko::Array<ValueType>{ref};
    auto recv_array = gko::Array<ValueType>{ref};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
    } else {
        data = new ValueType[4]{0, 0, 0, 0};
    }
    auto win = Window(data, 4 * sizeof(ValueType), comm);
    if (my_rank != 0) {
        win.lock_all();
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::get<ValueType>(data, 4, 0, 0, 4, win);
                win.flush(0);
            }
        }
        win.unlock_all();
    }
    gko::mpi::synchronize();
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 2);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 4);
    delete data;
}


TEST_F(MpiBindings, CanGetValuesWithExclusiveLock)
{
    using ValueType = int;
    using Window = gko::mpi::window<ValueType>;
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = comm->rank();
    auto num_ranks = comm->size();
    auto send_array = gko::Array<ValueType>{ref};
    auto recv_array = gko::Array<ValueType>{ref};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
    } else {
        data = new ValueType[4]{0, 0, 0, 0};
    }
    auto win = Window(data, 4 * sizeof(ValueType), comm);
    if (my_rank != 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                win.lock(0, 0, Window::lock_type::exclusive);
                gko::mpi::get<ValueType>(data, 4, 0, 0, 4, win);
                win.flush(0);
                win.unlock(0);
            }
        }
    }
    gko::mpi::synchronize();
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 2);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 4);
    delete data;
}


TEST_F(MpiBindings, CanGetValuesWithFence)
{
    using ValueType = int;
    using Window = gko::mpi::window<ValueType>;
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = comm->rank();
    auto num_ranks = comm->size();
    auto send_array = gko::Array<ValueType>{ref};
    auto recv_array = gko::Array<ValueType>{ref};
    int *data;
    if (my_rank == 0) {
        data = new ValueType[4]{1, 2, 3, 4};
    } else {
        data = new ValueType[4]{0, 0, 0, 0};
    }
    auto win = Window(data, 4 * sizeof(ValueType), comm);
    win.fence();
    if (my_rank != 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::get<ValueType>(data, 4, 0, 0, 4, win);
            }
        }
    }
    win.fence();
    gko::mpi::synchronize();
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 2);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 4);
    delete data;
}


TEST_F(MpiBindings, CanBroadcastValues)
{
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
    double *data;
    auto array = gko::Array<double>{ref, 8};
    if (my_rank == 0) {
        // clang-format off
        data = new double[8]{ 2.0, 3.0, 1.0,
                3.0,-1.0, 0.0 , 3.5, 1.5};
        // clang-format on
        array = gko::Array<double>{gko::Array<double>::view(ref, 8, data)};
    }
    gko::mpi::broadcast<double>(array.get_data(), 8, 0);
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


TEST_F(MpiBindings, CanReduceValues)
{
    using ValueType = double;
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
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
    gko::mpi::reduce<ValueType>(&data, &sum, 1, gko::mpi::op_type::sum, 0);
    gko::mpi::reduce<ValueType>(&data, &max, 1, gko::mpi::op_type::max, 0);
    gko::mpi::reduce<ValueType>(&data, &min, 1, gko::mpi::op_type::min, 0);
    if (my_rank == 0) {
        EXPECT_EQ(sum, 16.0);
        EXPECT_EQ(max, 6.0);
        EXPECT_EQ(min, 2.0);
    }
}


TEST_F(MpiBindings, CanAllReduceValues)
{
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
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
    gko::mpi::all_reduce<int>(&data, &sum, 1, gko::mpi::op_type::sum);
    ASSERT_EQ(sum, 16);
}


TEST_F(MpiBindings, CanAllReduceValuesInPlace)
{
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
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
    gko::mpi::all_reduce<int>(&data, 1, gko::mpi::op_type::sum);
    ASSERT_EQ(data, 16);
}


TEST_F(MpiBindings, CanScatterValues)
{
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
    double *data;
    auto scatter_from_array = gko::Array<double>{ref->get_master()};
    if (my_rank == 0) {
        // clang-format off
        data = new double[8]{ 2.0, 3.0, 1.0,
                3.0,-1.0, 0.0 , 3.5, 1.5};
        // clang-format on
        scatter_from_array = gko::Array<double>{
            ref->get_master(), gko::Array<double>::view(ref, 8, data)};
    }
    auto scatter_into_array = gko::Array<double>{ref, 2};
    gko::mpi::scatter<double, double>(scatter_from_array.get_data(), 2,
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


TEST_F(MpiBindings, CanGatherValues)
{
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
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
        gko::Array<int>{ref, static_cast<gko::size_type>(num_ranks)};
    gko::mpi::gather<int, int>(&data, 1, gather_array.get_data(), 1, 0);
    if (my_rank == 0) {
        ASSERT_EQ(gather_array.get_data()[0], 3);
        ASSERT_EQ(gather_array.get_data()[1], 5);
        ASSERT_EQ(gather_array.get_data()[2], 2);
        ASSERT_EQ(gather_array.get_data()[3], 6);
    }
}


TEST_F(MpiBindings, CanScatterValuesWithDisplacements)
{
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
    double *data;
    auto scatter_from_array = gko::Array<double>{ref};
    auto scatter_into_array = gko::Array<double>{ref};
    auto s_counts = gko::Array<int>{ref->get_master(),
                                    static_cast<gko::size_type>(num_ranks)};
    auto displacements = gko::Array<int>{ref->get_master()};
    int nelems;
    if (my_rank == 0) {
        // clang-format off
        data = new double[10]{ 2.0, 3.0, 1.0,
                3.0,-1.0, 0.0,
                2.5,-1.5, 0.5, 3.5};
        // clang-format on
        scatter_from_array =
            gko::Array<double>{ref, gko::Array<double>::view(ref, 10, data)};
        nelems = 2;
        displacements = gko::Array<int>{ref, {0, 2, 6, 9}};
    } else if (my_rank == 1) {
        nelems = 4;
    } else if (my_rank == 2) {
        nelems = 3;
    } else if (my_rank == 3) {
        nelems = 1;
    }
    scatter_into_array =
        gko::Array<double>{ref, static_cast<gko::size_type>(nelems)};
    gko::mpi::gather<int, int>(&nelems, 1, s_counts.get_data(), 1, 0);
    gko::mpi::scatter<double, double>(
        scatter_from_array.get_data(), s_counts.get_data(),
        displacements.get_data(), scatter_into_array.get_data(), nelems, 0);
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


TEST_F(MpiBindings, CanGatherValuesWithDisplacements)
{
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
    double *data;
    auto gather_from_array = gko::Array<double>{ref};
    auto gather_into_array = gko::Array<double>{ref};
    auto r_counts = gko::Array<int>{ref->get_master(),
                                    static_cast<gko::size_type>(num_ranks)};
    auto displacements = gko::Array<int>{ref->get_master()};
    int nelems;
    if (my_rank == 0) {
        data = new double[2]{2.0, 3.0};
        gather_from_array = gko::Array<double>{
            ref->get_master(),
            gko::Array<double>::view(ref->get_master(), 2, data)};
        nelems = 2;
        displacements = gko::Array<int>{ref->get_master(), {0, 2, 6, 7}};
        gather_into_array = gko::Array<double>{ref, 10};
    } else if (my_rank == 1) {
        data = new double[4]{1.5, 2.0, 1.0, 0.5};
        nelems = 4;
        gather_from_array = gko::Array<double>{
            ref->get_master(),
            gko::Array<double>::view(ref->get_master(), 4, data)};
    } else if (my_rank == 2) {
        data = new double[1]{1.0};
        nelems = 1;
        gather_from_array = gko::Array<double>{
            ref->get_master(),
            gko::Array<double>::view(ref->get_master(), 1, data)};
    } else if (my_rank == 3) {
        data = new double[3]{1.9, -4.0, 5.0};
        nelems = 3;
        gather_from_array = gko::Array<double>{
            ref->get_master(),
            gko::Array<double>::view(ref->get_master(), 3, data)};
    }

    gko::mpi::gather<int, int>(&nelems, 1, r_counts.get_data(), 1, 0);
    gko::mpi::gather<double, double>(
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


TEST_F(MpiBindings, AllToAllWorksCorrectly)
{
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
    auto send_array = gko::Array<double>{ref};
    auto recv_array = gko::Array<double>{ref};
    auto ref_array = gko::Array<double>{ref};
    recv_array = gko::Array<double>{ref, 4};
    if (my_rank == 0) {
        send_array = gko::Array<double>(ref, {2.5, 3.0, 1.5, 2.0});
        ref_array = gko::Array<double>(ref, {2.5, 2.5, 2.0, 5.5});
    } else if (my_rank == 1) {
        send_array = gko::Array<double>(ref, {2.5, 3.5, 1.0, 2.0});
        ref_array = gko::Array<double>(ref, {3.0, 3.5, 3.0, 3.5});
    } else if (my_rank == 2) {
        send_array = gko::Array<double>(ref, {2.0, 3.0, 1.5, 0.0});
        ref_array = gko::Array<double>(ref, {1.5, 1.0, 1.5, 3.5});
    } else if (my_rank == 3) {
        send_array = gko::Array<double>(ref, {5.5, 3.5, 3.5, -2.0});
        ref_array = gko::Array<double>(ref, {2.0, 2.0, 0.0, -2.0});
    }

    gko::mpi::all_to_all<double, double>(send_array.get_data(), 1,
                                         recv_array.get_data());
    this->assert_equal_arrays(recv_array, ref_array);
}


TEST_F(MpiBindings, AllToAllInPlaceWorksCorrectly)
{
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
    auto recv_array = gko::Array<double>{ref};
    auto ref_array = gko::Array<double>{ref};
    recv_array = gko::Array<double>{ref, 4};
    if (my_rank == 0) {
        recv_array = gko::Array<double>(ref, {2.5, 3.0, 1.5, 2.0});
        ref_array = gko::Array<double>(ref, {2.5, 2.5, 2.0, 5.5});
    } else if (my_rank == 1) {
        recv_array = gko::Array<double>(ref, {2.5, 3.5, 1.0, 2.0});
        ref_array = gko::Array<double>(ref, {3.0, 3.5, 3.0, 3.5});
    } else if (my_rank == 2) {
        recv_array = gko::Array<double>(ref, {2.0, 3.0, 1.5, 0.0});
        ref_array = gko::Array<double>(ref, {1.5, 1.0, 1.5, 3.5});
    } else if (my_rank == 3) {
        recv_array = gko::Array<double>(ref, {5.5, 3.5, 3.5, -2.0});
        ref_array = gko::Array<double>(ref, {2.0, 2.0, 0.0, -2.0});
    }

    gko::mpi::all_to_all<double>(recv_array.get_data(), 1);
    this->assert_equal_arrays(recv_array, ref_array);
}


TEST_F(MpiBindings, AllToAllVWorksCorrectly)
{
    auto comm = gko::mpi::communicator::create(MPI_COMM_WORLD);
    auto my_rank = gko::mpi::get_my_rank(comm->get());
    auto num_ranks = gko::mpi::get_num_ranks(comm->get());
    auto send_array = gko::Array<double>{ref};
    auto recv_array = gko::Array<double>{ref};
    auto ref_array = gko::Array<double>{ref};
    auto scounts_array = gko::Array<int>{ref};
    auto soffset_array = gko::Array<int>{ref};
    auto rcounts_array = gko::Array<int>{ref};
    auto roffset_array = gko::Array<int>{ref};
    if (my_rank == 0) {
        recv_array = gko::Array<double>{ref, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        send_array = gko::Array<double>{ref, {2.5, 3.0, 1.5, 2.0}};
        scounts_array = gko::Array<int>{ref, {1, 2, 1, 0}};
        rcounts_array = gko::Array<int>{ref, {1, 2, 2, 1}};
        soffset_array = gko::Array<int>{ref, {0, 1, 1, 0}};
        roffset_array = gko::Array<int>{ref, {0, 1, 3, 5}};
        ref_array = gko::Array<double>{ref, {2.5, 2.5, 3.5, 1.5, 2.4, 5.5}};
    } else if (my_rank == 1) {
        recv_array = gko::Array<double>{ref, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        send_array = gko::Array<double>{ref, {2.5, 3.5, 1.0, 2.0}};
        scounts_array = gko::Array<int>{ref, {2, 2, 1, 2}};
        rcounts_array = gko::Array<int>{ref, {2, 2, 2, 0}};
        soffset_array = gko::Array<int>{ref, {0, 1, 1, 0}};
        roffset_array = gko::Array<int>{ref, {0, 2, 4, 5}};
        ref_array = gko::Array<double>{ref, {3.0, 1.5, 3.5, 1.0, 3.0, 1.5}};
    } else if (my_rank == 2) {
        recv_array = gko::Array<double>{ref, {0.0, 0.0, 0.0, 0.0}};
        send_array = gko::Array<double>{ref, {2.0, 3.0, 1.5, 2.4}};
        scounts_array = gko::Array<int>{ref, {2, 2, 1, 1}};
        rcounts_array = gko::Array<int>{ref, {1, 1, 1, 1}};
        soffset_array = gko::Array<int>{ref, {2, 1, 1, 1}};
        roffset_array = gko::Array<int>{ref, {0, 1, 2, 3}};
        ref_array = gko::Array<double>{ref, {3.0, 3.5, 3.0, 3.5}};
    } else if (my_rank == 3) {
        recv_array = gko::Array<double>{ref, {0.0, 0.0, 0.0, 0.0}};
        send_array = gko::Array<double>{ref, {5.5, 3.5, 3.5, -2.0}};
        scounts_array = gko::Array<int>{ref, {1, 0, 1, 0}};
        rcounts_array = gko::Array<int>{ref, {0, 2, 1, 0}};
        soffset_array = gko::Array<int>{ref, {0, 1, 1, 0}};
        roffset_array = gko::Array<int>{ref, {0, 1, 3, 3}};
        ref_array = gko::Array<double>{ref, {0.0, 2.5, 3.5, 3.0}};
    }

    gko::mpi::all_to_all<double, double>(
        send_array.get_data(), scounts_array.get_data(),
        soffset_array.get_data(), recv_array.get_data(),
        rcounts_array.get_data(), roffset_array.get_data());
    this->assert_equal_arrays(recv_array, ref_array);
}


// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
