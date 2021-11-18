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


#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>


#include "core/test/utils.hpp"


template <typename T>
class MpiBindings : public ::testing::Test {
protected:
    using value_type = T;
    MpiBindings() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<gko::Executor> ref;
};

TYPED_TEST_SUITE(MpiBindings, gko::test::PODTypes, TypenameNameGenerator);


TYPED_TEST(MpiBindings, CanSetADefaultwindow)
{
    gko::mpi::window<TypeParam> win;
    ASSERT_EQ(win.get(), MPI_WIN_NULL);
}


TYPED_TEST(MpiBindings, CanCreatewindow)
{
    auto data = std::vector<TypeParam>{1, 2, 3, 4};
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto win =
        gko::mpi::window<TypeParam>(data.data(), 4 * sizeof(TypeParam), comm);
    ASSERT_NE(win.get(), MPI_WIN_NULL);
    win.lock_all();
    win.unlock_all();
}


TYPED_TEST(MpiBindings, CanSendAndRecvValues)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto recv_array = gko::Array<TypeParam>{this->ref};
    if (my_rank == 0) {
        auto send_array = std::vector<TypeParam>{1, 2, 3, 4};
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::send(send_array.data(), 4, rank, 40 + rank, comm);
            }
        }
    } else {
        recv_array = gko::Array<TypeParam>{this->ref, 4};
        gko::mpi::recv(recv_array.get_data(), 4, 0, 40 + my_rank, comm);
    }
    if (my_rank != 0) {
        auto ref_array = gko::Array<TypeParam>{this->ref, {1, 2, 3, 4}};
        GKO_ASSERT_ARRAY_EQ(ref_array, recv_array);
    }
}


TYPED_TEST(MpiBindings, CanNonBlockingSendAndNonBlockingRecvValues)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> send_array;
    auto recv_array = gko::Array<TypeParam>{this->ref};
    TypeParam* data;
    std::vector<MPI_Request> req1;
    MPI_Request req2;
    if (my_rank == 0) {
        send_array = std::vector<TypeParam>{1, 2, 3, 4};
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                req1.emplace_back(gko::mpi::i_send(send_array.data(), 4, rank,
                                                   40 + rank, comm));
            }
        }
    } else {
        recv_array = gko::Array<TypeParam>{this->ref, 4};
        req2 = std::move(
            gko::mpi::i_recv(recv_array.get_data(), 4, 0, 40 + my_rank, comm));
    }
    if (my_rank == 0) {
        auto stat1 = gko::mpi::wait_all(req1);
    } else {
        auto stat2 = gko::mpi::wait(req2);
        auto ref_array = gko::Array<TypeParam>{this->ref, {1, 2, 3, 4}};
        GKO_ASSERT_ARRAY_EQ(ref_array, recv_array);
    }
}


TYPED_TEST(MpiBindings, CanPutValuesWithLockAll)
{
    using window = gko::mpi::window<TypeParam>;
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(data.data(), 4 * sizeof(TypeParam), comm);
    win.lock_all();
    if (my_rank == 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::put(data.data(), 4, rank, 0, 4, win);
                win.flush(rank);
            }
        }
    }
    win.unlock_all();

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanPutValuesWithExclusiveLock)
{
    using window = gko::mpi::window<TypeParam>;
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(data.data(), 4 * sizeof(TypeParam), comm);
    if (my_rank == 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                win.lock(rank, 0, window::lock_type::exclusive);
                gko::mpi::put(data.data(), 4, rank, 0, 4, win);
                win.flush(rank);
                win.unlock(rank);
            }
        }
    }

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanPutValuesWithFence)
{
    using window = gko::mpi::window<TypeParam>;
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(data.data(), 4 * sizeof(TypeParam), comm);
    win.fence();
    if (my_rank == 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::put(data.data(), 4, rank, 0, 4, win);
            }
        }
    }
    win.fence();

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanGetValuesWithLockAll)
{
    using window = gko::mpi::window<TypeParam>;
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(data.data(), 4 * sizeof(TypeParam), comm);
    if (my_rank != 0) {
        win.lock_all();
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::get(data.data(), 4, 0, 0, 4, win);
                win.flush(0);
            }
        }
        win.unlock_all();
    }

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanGetValuesWithExclusiveLock)
{
    using window = gko::mpi::window<TypeParam>;
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(data.data(), 4 * sizeof(TypeParam), comm);
    if (my_rank != 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                win.lock(0, 0, window::lock_type::exclusive);
                gko::mpi::get(data.data(), 4, 0, 0, 4, win);
                win.flush(0);
                win.unlock(0);
            }
        }
    }

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanGetValuesWithFence)
{
    using window = gko::mpi::window<TypeParam>;
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(data.data(), 4 * sizeof(TypeParam), comm);
    win.fence();
    if (my_rank != 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                gko::mpi::get(data.data(), 4, 0, 0, 4, win);
            }
        }
    }
    win.fence();

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanBroadcastValues)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto array = gko::Array<TypeParam>{this->ref, 8};
    if (my_rank == 0) {
        array = gko::Array<TypeParam>(this->ref, {2, 3, 1, 3, -1, 0, 3, 1});
    }
    gko::mpi::broadcast(array.get_data(), 8, 0, comm);
    auto comp_data = array.get_data();
    ASSERT_EQ(comp_data[0], TypeParam{2});
    ASSERT_EQ(comp_data[1], TypeParam{3});
    ASSERT_EQ(comp_data[2], TypeParam{1});
    ASSERT_EQ(comp_data[3], TypeParam{3});
    ASSERT_EQ(comp_data[4], TypeParam{-1});
    ASSERT_EQ(comp_data[5], TypeParam{0});
    ASSERT_EQ(comp_data[6], TypeParam{3});
    ASSERT_EQ(comp_data[7], TypeParam{1});
}


TYPED_TEST(MpiBindings, CanReduceValues)
{
    using TypeParam = TypeParam;
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    TypeParam data, sum, max, min;
    if (my_rank == 0) {
        data = 3;
    } else if (my_rank == 1) {
        data = 5;
    } else if (my_rank == 2) {
        data = 2;
    } else if (my_rank == 3) {
        data = 6;
    }
    gko::mpi::reduce(&data, &sum, 1, MPI_SUM, 0, comm);
    gko::mpi::reduce(&data, &max, 1, MPI_MAX, 0, comm);
    gko::mpi::reduce(&data, &min, 1, MPI_MIN, 0, comm);
    if (my_rank == 0) {
        EXPECT_EQ(sum, TypeParam{16});
        EXPECT_EQ(max, TypeParam{6});
        EXPECT_EQ(min, TypeParam{2});
    }
}


TYPED_TEST(MpiBindings, CanAllReduceValues)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    TypeParam data, sum;
    if (my_rank == 0) {
        data = 3;
    } else if (my_rank == 1) {
        data = 5;
    } else if (my_rank == 2) {
        data = 2;
    } else if (my_rank == 3) {
        data = 6;
    }
    gko::mpi::all_reduce(&data, &sum, 1, MPI_SUM, comm);
    ASSERT_EQ(sum, TypeParam{16});
}


TYPED_TEST(MpiBindings, CanAllReduceValuesInPlace)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    TypeParam data;
    if (my_rank == 0) {
        data = 3;
    } else if (my_rank == 1) {
        data = 5;
    } else if (my_rank == 2) {
        data = 2;
    } else if (my_rank == 3) {
        data = 6;
    }
    gko::mpi::all_reduce(&data, 1, MPI_SUM, comm);
    ASSERT_EQ(data, TypeParam{16});
}


TYPED_TEST(MpiBindings, CanScatterValues)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto scatter_from_array = gko::Array<TypeParam>{this->ref};
    if (my_rank == 0) {
        scatter_from_array =
            gko::Array<TypeParam>{this->ref, {2, 3, 1, 3, -1, 0, 3, 1}};
    }
    auto scatter_into_array = gko::Array<TypeParam>{this->ref, 2};
    gko::mpi::scatter(scatter_from_array.get_data(), 2,
                      scatter_into_array.get_data(), 2, 0, comm);
    auto comp_data = scatter_into_array.get_data();
    if (my_rank == 0) {
        ASSERT_EQ(comp_data[0], TypeParam{2});
        ASSERT_EQ(comp_data[1], TypeParam{3});

    } else if (my_rank == 1) {
        ASSERT_EQ(comp_data[0], TypeParam{1});
        ASSERT_EQ(comp_data[1], TypeParam{3});
    } else if (my_rank == 2) {
        ASSERT_EQ(comp_data[0], TypeParam{-1});
        ASSERT_EQ(comp_data[1], TypeParam{0});
    } else if (my_rank == 3) {
        ASSERT_EQ(comp_data[0], TypeParam{3});
        ASSERT_EQ(comp_data[1], TypeParam{1});
    }
}


TYPED_TEST(MpiBindings, CanGatherValues)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    TypeParam data;
    if (my_rank == 0) {
        data = 3;
    } else if (my_rank == 1) {
        data = 5;
    } else if (my_rank == 2) {
        data = 2;
    } else if (my_rank == 3) {
        data = 6;
    }
    auto gather_array = gko::Array<TypeParam>{
        this->ref, static_cast<gko::size_type>(num_ranks)};
    gko::mpi::gather(&data, 1, gather_array.get_data(), 1, 0, comm);
    if (my_rank == 0) {
        ASSERT_EQ(gather_array.get_data()[0], TypeParam{3});
        ASSERT_EQ(gather_array.get_data()[1], TypeParam{5});
        ASSERT_EQ(gather_array.get_data()[2], TypeParam{2});
        ASSERT_EQ(gather_array.get_data()[3], TypeParam{6});
    }
}


TYPED_TEST(MpiBindings, CanScatterValuesWithDisplacements)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto scatter_from_array = gko::Array<TypeParam>{this->ref};
    auto scatter_into_array = gko::Array<TypeParam>{this->ref};
    auto s_counts = gko::Array<int>{this->ref->get_master(),
                                    static_cast<gko::size_type>(num_ranks)};
    auto displacements = gko::Array<int>{this->ref->get_master()};
    int nelems;
    if (my_rank == 0) {
        scatter_from_array =
            gko::Array<TypeParam>{this->ref, {2, 3, 1, 3, -1, 0, 2, -1, 0, 3}};
        nelems = 2;
        displacements = gko::Array<int>{this->ref, {0, 2, 6, 9}};
    } else if (my_rank == 1) {
        nelems = 4;
    } else if (my_rank == 2) {
        nelems = 3;
    } else if (my_rank == 3) {
        nelems = 1;
    }
    scatter_into_array =
        gko::Array<TypeParam>{this->ref, static_cast<gko::size_type>(nelems)};
    gko::mpi::gather(&nelems, 1, s_counts.get_data(), 1, 0, comm);
    gko::mpi::scatter_v(scatter_from_array.get_data(), s_counts.get_data(),
                        displacements.get_data(), scatter_into_array.get_data(),
                        nelems, 0, comm);
    auto comp_data = scatter_into_array.get_data();
    if (my_rank == 0) {
        ASSERT_EQ(comp_data[0], TypeParam{2});
        ASSERT_EQ(comp_data[1], TypeParam{3});

    } else if (my_rank == 1) {
        ASSERT_EQ(comp_data[0], TypeParam{1});
        ASSERT_EQ(comp_data[1], TypeParam{3});
        ASSERT_EQ(comp_data[2], TypeParam{-1});
        ASSERT_EQ(comp_data[3], TypeParam{0});
    } else if (my_rank == 2) {
        ASSERT_EQ(comp_data[0], TypeParam{2});
        ASSERT_EQ(comp_data[1], TypeParam{-1});
        ASSERT_EQ(comp_data[2], TypeParam{0});
    } else if (my_rank == 3) {
        ASSERT_EQ(comp_data[0], TypeParam{3});
    }
}


TYPED_TEST(MpiBindings, CanGatherValuesWithDisplacements)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto gather_from_array = gko::Array<TypeParam>{this->ref};
    auto gather_into_array = gko::Array<TypeParam>{this->ref};
    auto r_counts =
        gko::Array<int>{this->ref, static_cast<gko::size_type>(num_ranks)};
    auto displacements = gko::Array<int>{this->ref};
    int nelems;
    if (my_rank == 0) {
        gather_from_array = gko::Array<TypeParam>{this->ref, {2, 3}};
        nelems = 2;
        displacements = gko::Array<int>{this->ref, {0, 2, 6, 7}};
        gather_into_array = gko::Array<TypeParam>{this->ref, 10};
    } else if (my_rank == 1) {
        nelems = 4;
        gather_from_array = gko::Array<TypeParam>{this->ref, {1, 2, 1, 0}};
    } else if (my_rank == 2) {
        nelems = 1;
        gather_from_array = gko::Array<TypeParam>{this->ref, {1}};
    } else if (my_rank == 3) {
        nelems = 3;
        gather_from_array = gko::Array<TypeParam>{this->ref, {1, -4, 5}};
    }

    gko::mpi::gather(&nelems, 1, r_counts.get_data(), 1, 0, comm);
    gko::mpi::gather_v(gather_from_array.get_data(), nelems,
                       gather_into_array.get_data(), r_counts.get_data(),
                       displacements.get_data(), 0, comm);
    auto comp_data = gather_into_array.get_data();
    if (my_rank == 0) {
        auto ref_array =
            gko::Array<TypeParam>(this->ref, {2, 3, 1, 2, 1, 0, 1, 1, -4, 5});
        GKO_ASSERT_ARRAY_EQ(gather_into_array, ref_array);
    } else {
        ASSERT_EQ(comp_data, nullptr);
    }
}


TYPED_TEST(MpiBindings, AllToAllWorksCorrectly)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto send_array = gko::Array<TypeParam>{this->ref};
    auto recv_array = gko::Array<TypeParam>{this->ref};
    auto ref_array = gko::Array<TypeParam>{this->ref};
    recv_array = gko::Array<TypeParam>{this->ref, 4};
    if (my_rank == 0) {
        send_array = gko::Array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::Array<TypeParam>(this->ref, {2, 2, 2, 5});
    } else if (my_rank == 1) {
        send_array = gko::Array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::Array<TypeParam>(this->ref, {3, 3, 3, 3});
    } else if (my_rank == 2) {
        send_array = gko::Array<TypeParam>(this->ref, {2, 3, 1, 0});
        ref_array = gko::Array<TypeParam>(this->ref, {1, 1, 1, 3});
    } else if (my_rank == 3) {
        send_array = gko::Array<TypeParam>(this->ref, {5, 3, 3, -2});
        ref_array = gko::Array<TypeParam>(this->ref, {2, 2, 0, -2});
    }

    gko::mpi::all_to_all(send_array.get_data(), 1, recv_array.get_data(), 1,
                         comm);
    GKO_ASSERT_ARRAY_EQ(recv_array, ref_array);
}


TYPED_TEST(MpiBindings, AllToAllInPlaceWorksCorrectly)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto recv_array = gko::Array<TypeParam>{this->ref};
    auto ref_array = gko::Array<TypeParam>{this->ref};
    recv_array = gko::Array<TypeParam>{this->ref, 4};
    if (my_rank == 0) {
        recv_array = gko::Array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::Array<TypeParam>(this->ref, {2, 2, 2, 5});
    } else if (my_rank == 1) {
        recv_array = gko::Array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::Array<TypeParam>(this->ref, {3, 3, 3, 3});
    } else if (my_rank == 2) {
        recv_array = gko::Array<TypeParam>(this->ref, {2, 3, 1, 0});
        ref_array = gko::Array<TypeParam>(this->ref, {1, 1, 1, 3});
    } else if (my_rank == 3) {
        recv_array = gko::Array<TypeParam>(this->ref, {5, 3, 3, -2});
        ref_array = gko::Array<TypeParam>(this->ref, {2, 2, 0, -2});
    }

    gko::mpi::all_to_all(recv_array.get_data(), 1, comm);
    GKO_ASSERT_ARRAY_EQ(recv_array, ref_array);
}


TYPED_TEST(MpiBindings, AllToAllVWorksCorrectly)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto send_array = gko::Array<TypeParam>{this->ref};
    auto recv_array = gko::Array<TypeParam>{this->ref};
    auto ref_array = gko::Array<TypeParam>{this->ref};
    auto scounts_array = gko::Array<int>{this->ref};
    auto soffset_array = gko::Array<int>{this->ref};
    auto rcounts_array = gko::Array<int>{this->ref};
    auto roffset_array = gko::Array<int>{this->ref};
    if (my_rank == 0) {
        recv_array = gko::Array<TypeParam>{this->ref, {0, 0, 0, 0, 0, 0}};
        send_array = gko::Array<TypeParam>{this->ref, {2, 3, 1, 2}};
        scounts_array = gko::Array<int>{this->ref, {1, 2, 1, 0}};
        rcounts_array = gko::Array<int>{this->ref, {1, 2, 2, 1}};
        soffset_array = gko::Array<int>{this->ref, {0, 1, 1, 0}};
        roffset_array = gko::Array<int>{this->ref, {0, 1, 3, 5}};
        ref_array = gko::Array<TypeParam>{this->ref, {2, 2, 3, 1, 2, 5}};
    } else if (my_rank == 1) {
        recv_array = gko::Array<TypeParam>{this->ref, {0, 0, 0, 0, 0, 0}};
        send_array = gko::Array<TypeParam>{this->ref, {2, 3, 1, 2}};
        scounts_array = gko::Array<int>{this->ref, {2, 2, 1, 2}};
        rcounts_array = gko::Array<int>{this->ref, {2, 2, 2, 0}};
        soffset_array = gko::Array<int>{this->ref, {0, 1, 1, 0}};
        roffset_array = gko::Array<int>{this->ref, {0, 2, 4, 5}};
        ref_array = gko::Array<TypeParam>{this->ref, {3, 1, 3, 1, 3, 1}};
    } else if (my_rank == 2) {
        recv_array = gko::Array<TypeParam>{this->ref, {0, 0, 0, 0}};
        send_array = gko::Array<TypeParam>{this->ref, {2, 3, 1, 2}};
        scounts_array = gko::Array<int>{this->ref, {2, 2, 1, 1}};
        rcounts_array = gko::Array<int>{this->ref, {1, 1, 1, 1}};
        soffset_array = gko::Array<int>{this->ref, {2, 1, 1, 1}};
        roffset_array = gko::Array<int>{this->ref, {0, 1, 2, 3}};
        ref_array = gko::Array<TypeParam>{this->ref, {3, 3, 3, 3}};
    } else if (my_rank == 3) {
        recv_array = gko::Array<TypeParam>{this->ref, {0, 0, 0, 0}};
        send_array = gko::Array<TypeParam>{this->ref, {5, 3, 3, -2}};
        scounts_array = gko::Array<int>{this->ref, {1, 0, 1, 0}};
        rcounts_array = gko::Array<int>{this->ref, {0, 2, 1, 0}};
        soffset_array = gko::Array<int>{this->ref, {0, 1, 1, 0}};
        roffset_array = gko::Array<int>{this->ref, {0, 1, 3, 3}};
        ref_array = gko::Array<TypeParam>{this->ref, {0, 2, 3, 3}};
    }

    gko::mpi::all_to_all_v(send_array.get_data(), scounts_array.get_data(),
                           soffset_array.get_data(), recv_array.get_data(),
                           rcounts_array.get_data(), roffset_array.get_data(),
                           comm);
    GKO_ASSERT_ARRAY_EQ(recv_array, ref_array);
}


TYPED_TEST(MpiBindings, CanScanValues)
{
    auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    TypeParam data, sum, max, min;
    if (my_rank == 0) {
        data = 3;
    } else if (my_rank == 1) {
        data = 5;
    } else if (my_rank == 2) {
        data = 2;
    } else if (my_rank == 3) {
        data = 6;
    }
    gko::mpi::scan(&data, &sum, 1, MPI_SUM, comm);
    gko::mpi::scan(&data, &max, 1, MPI_MAX, comm);
    gko::mpi::scan(&data, &min, 1, MPI_MIN, comm);
    if (my_rank == 0) {
        EXPECT_EQ(sum, TypeParam{3});
        EXPECT_EQ(max, TypeParam{3});
        EXPECT_EQ(min, TypeParam{3});
    } else if (my_rank == 1) {
        EXPECT_EQ(sum, TypeParam{8});
        EXPECT_EQ(max, TypeParam{5});
        EXPECT_EQ(min, TypeParam{3});
    } else if (my_rank == 2) {
        EXPECT_EQ(sum, TypeParam{10});
        EXPECT_EQ(max, TypeParam{5});
        EXPECT_EQ(min, TypeParam{2});
    } else if (my_rank == 3) {
        EXPECT_EQ(sum, TypeParam{16});
        EXPECT_EQ(max, TypeParam{6});
        EXPECT_EQ(min, TypeParam{2});
    }
}
