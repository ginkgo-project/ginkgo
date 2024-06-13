// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    gko::experimental::mpi::window<TypeParam> win;
    ASSERT_EQ(win.get_window(), MPI_WIN_NULL);
}


TYPED_TEST(MpiBindings, CanCreatewindow)
{
    auto data = std::vector<TypeParam>{1, 2, 3, 4};
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);

    auto win = gko::experimental::mpi::window<TypeParam>(
        this->ref, data.data(), 4 * sizeof(TypeParam), comm);

    ASSERT_NE(win.get_window(), MPI_WIN_NULL);
    win.lock_all();
    win.unlock_all();
}


TYPED_TEST(MpiBindings, CanSendAndRecvValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto recv_array = gko::array<TypeParam>{this->ref};

    if (my_rank == 0) {
        auto send_array = std::vector<TypeParam>{1, 2, 3, 4};
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                comm.send(this->ref, send_array.data(), 4, rank, 40 + rank);
            }
        }
    } else {
        recv_array = gko::array<TypeParam>{this->ref, 4};
        comm.recv(this->ref, recv_array.get_data(), 4, 0, 40 + my_rank);
    }

    if (my_rank != 0) {
        auto ref_array = gko::array<TypeParam>{this->ref, {1, 2, 3, 4}};
        GKO_ASSERT_ARRAY_EQ(ref_array, recv_array);
    }
}


TYPED_TEST(MpiBindings, CanNonBlockingSendAndNonBlockingRecvValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> send_array;
    auto recv_array = gko::array<TypeParam>{this->ref};
    TypeParam* data;
    auto req1 = std::vector<gko::experimental::mpi::request>(num_ranks);
    auto req2 = gko::experimental::mpi::request();

    if (my_rank == 0) {
        send_array = std::vector<TypeParam>{1, 2, 3, 4};
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                req1[rank] = comm.i_send(this->ref, send_array.data(), 4, rank,
                                         40 + rank);
            }
        }
    } else {
        recv_array = gko::array<TypeParam>{this->ref, 4};
        req2 =
            comm.i_recv(this->ref, recv_array.get_data(), 4, 0, 40 + my_rank);
    }

    if (my_rank == 0) {
        auto stat1 = wait_all(req1);
    } else {
        auto stat2 = req2.wait();
        auto count = stat2.get_count(recv_array.get_data());
        ASSERT_EQ(count, 4);
        auto ref_array = gko::array<TypeParam>{this->ref, {1, 2, 3, 4}};
        GKO_ASSERT_ARRAY_EQ(ref_array, recv_array);
    }
}


TYPED_TEST(MpiBindings, CanPutValuesWithLockAll)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }

    {
        auto win = window(this->ref, data.data(), 4, comm);
        if (my_rank == 0) {
            win.lock_all();
            for (auto rank = 0; rank < num_ranks; ++rank) {
                if (rank != my_rank) {
                    win.put(this->ref, data.data(), 4, rank, 0, 4);
                }
            }
            win.flush_all();
            win.unlock_all();
        }
    }

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanNonBlockingPutValuesWithLockAll)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }

    {
        gko::experimental::mpi::request req;
        auto win = window(this->ref, data.data(), 4, comm);
        if (my_rank == 0) {
            win.lock_all();
            for (auto rank = 0; rank < num_ranks; ++rank) {
                if (rank != my_rank) {
                    req = win.r_put(this->ref, data.data(), 4, rank, 0, 4);
                }
            }
            req.wait();
            win.flush_all();
            win.unlock_all();
        }
    }

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanPutValuesWithExclusiveLock)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;

    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }

    {
        auto win = window(this->ref, data.data(), 4, comm);
        if (my_rank == 0) {
            for (auto rank = 0; rank < num_ranks; ++rank) {
                if (rank != my_rank) {
                    win.lock(rank, window::lock_type::exclusive);
                    win.put(this->ref, data.data(), 4, rank, 0, 4);
                    win.flush(rank);
                    win.unlock(rank);
                }
            }
        }
    }

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanPutValuesWithSharedLock)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;

    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }

    {
        auto win = window(this->ref, data.data(), 4, comm);
        if (my_rank == 0) {
            for (auto rank = 0; rank < num_ranks; ++rank) {
                if (rank != my_rank) {
                    win.lock(rank);
                    win.put(this->ref, data.data(), 4, rank, 0, 4);
                    win.flush(rank);
                    win.unlock(rank);
                }
            }
        }
    }

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanPutValuesWithFence)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(this->ref, data.data(), 4, comm);

    win.fence();
    if (my_rank == 0) {
        for (auto rank = 0; rank < num_ranks; ++rank) {
            if (rank != my_rank) {
                win.put(this->ref, data.data(), 4, rank, 0, 4);
            }
        }
    }
    win.fence();

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanAccumulateValues)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else if (my_rank == 1) {
        data = std::vector<TypeParam>{5, 6, 7, 8};
    } else if (my_rank == 2) {
        data = std::vector<TypeParam>{9, 10, 11, 12};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }

    {
        auto win = window(this->ref, data.data(), 4, comm);
        if (my_rank == 0) {
            win.lock_all();
            for (auto rank = 0; rank < num_ranks; ++rank) {
                if (rank != my_rank) {
                    win.accumulate(this->ref, data.data(), 4, rank, 0, 4,
                                   MPI_SUM);
                }
            }
            win.unlock_all();
        }
    }

    std::vector<TypeParam> ref;
    if (my_rank == 0) {
        ref = std::vector<TypeParam>{1, 2, 3, 4};
        ASSERT_EQ(data, ref);
    } else if (my_rank == 1) {
        ref = std::vector<TypeParam>{6, 8, 10, 12};
        ASSERT_EQ(data, ref);
    } else if (my_rank == 2) {
        ref = std::vector<TypeParam>{10, 12, 14, 16};
        ASSERT_EQ(data, ref);
    } else {
        ref = std::vector<TypeParam>{1, 2, 3, 4};
        ASSERT_EQ(data, ref);
    }
}


TYPED_TEST(MpiBindings, CanNonBlockingAccumulateValues)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else if (my_rank == 1) {
        data = std::vector<TypeParam>{5, 6, 7, 8};
    } else if (my_rank == 2) {
        data = std::vector<TypeParam>{9, 10, 11, 12};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }

    gko::experimental::mpi::request req;
    {
        auto win = window(this->ref, data.data(), 4, comm);
        if (my_rank == 0) {
            win.lock_all();
            for (auto rank = 0; rank < num_ranks; ++rank) {
                if (rank != my_rank) {
                    req = win.r_accumulate(this->ref, data.data(), 4, rank, 0,
                                           4, MPI_SUM);
                }
            }
            win.unlock_all();
        }
    }

    req.wait();
    std::vector<TypeParam> ref;
    if (my_rank == 0) {
        ref = std::vector<TypeParam>{1, 2, 3, 4};
        ASSERT_EQ(data, ref);
    } else if (my_rank == 1) {
        ref = std::vector<TypeParam>{6, 8, 10, 12};
        ASSERT_EQ(data, ref);
    } else if (my_rank == 2) {
        ref = std::vector<TypeParam>{10, 12, 14, 16};
        ASSERT_EQ(data, ref);
    } else {
        ref = std::vector<TypeParam>{1, 2, 3, 4};
        ASSERT_EQ(data, ref);
    }
}


TYPED_TEST(MpiBindings, CanGetValuesWithLockAll)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(this->ref, data.data(), 4, comm);

    if (my_rank != 0) {
        win.lock_all();
        win.get(this->ref, data.data(), 4, 0, 0, 4);
        win.unlock_all();
    }

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanNonBlockingGetValuesWithLockAll)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    gko::experimental::mpi::request req;
    auto win = window(this->ref, data.data(), 4, comm);

    if (my_rank != 0) {
        win.lock_all();
        req = win.r_get(this->ref, data.data(), 4, 0, 0, 4);
        win.unlock_all();
    }

    req.wait();
    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanGetValuesWithExclusiveLock)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(this->ref, data.data(), 4, comm);

    if (my_rank != 0) {
        win.lock(0, window::lock_type::exclusive);
        win.get(this->ref, data.data(), 4, 0, 0, 4);
        win.unlock(0);
    }

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanGetValuesWithSharedLock)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(this->ref, data.data(), 4, comm);

    if (my_rank != 0) {
        win.lock(0);
        win.get(this->ref, data.data(), 4, 0, 0, 4);
        win.unlock(0);
    }

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanGetValuesWithFence)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
    }
    auto win = window(this->ref, data.data(), 4, comm);

    win.fence();
    if (my_rank != 0) {
        win.get(this->ref, data.data(), 4, 0, 0, 4);
    }
    win.fence();

    auto ref = std::vector<TypeParam>{1, 2, 3, 4};
    ASSERT_EQ(data, ref);
}


TYPED_TEST(MpiBindings, CanGetAccumulateValuesWithLockAll)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    std::vector<TypeParam> target;
    std::vector<TypeParam> result(4, 0);
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
        target = std::vector<TypeParam>{1, 2, 3, 4};
    } else if (my_rank == 1) {
        data = std::vector<TypeParam>{5, 6, 7, 8};
        target = std::vector<TypeParam>{5, 6, 7, 8};
    } else if (my_rank == 2) {
        data = std::vector<TypeParam>{9, 10, 11, 12};
        target = std::vector<TypeParam>{9, 10, 11, 12};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
        target = std::vector<TypeParam>{0, 0, 0, 0};
    }

    {
        auto win = window(this->ref, target.data(), 4, comm);

        if (my_rank == 2) {
            win.lock_all();
            win.get_accumulate(this->ref, data.data(), 4, result.data(), 4, 0,
                               0, 4, MPI_SUM);
            win.unlock_all();
        }
    }

    std::vector<TypeParam> ref;
    std::vector<TypeParam> ref2;
    if (my_rank == 0) {
        ref = std::vector<TypeParam>{10, 12, 14, 16};
        EXPECT_EQ(target, ref);
    } else if (my_rank == 2) {
        ref = std::vector<TypeParam>{1, 2, 3, 4};
        EXPECT_EQ(result, ref);
    }
}


TYPED_TEST(MpiBindings, CanNonBlockingGetAccumulateValuesWithLockAll)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    std::vector<TypeParam> target;
    std::vector<TypeParam> result(4, 0);
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
        target = std::vector<TypeParam>{1, 2, 3, 4};
    } else if (my_rank == 1) {
        data = std::vector<TypeParam>{5, 6, 7, 8};
        target = std::vector<TypeParam>{5, 6, 7, 8};
    } else if (my_rank == 2) {
        data = std::vector<TypeParam>{9, 10, 11, 12};
        target = std::vector<TypeParam>{9, 10, 11, 12};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
        target = std::vector<TypeParam>{0, 0, 0, 0};
    }
    gko::experimental::mpi::request req;

    {
        auto win = window(this->ref, target.data(), 4, comm);

        if (my_rank == 2) {
            win.lock_all();
            req = win.r_get_accumulate(this->ref, data.data(), 4, result.data(),
                                       4, 0, 0, 4, MPI_SUM);
            win.unlock_all();
        }
    }

    req.wait();
    std::vector<TypeParam> ref;
    std::vector<TypeParam> ref2;
    if (my_rank == 0) {
        ref = std::vector<TypeParam>{10, 12, 14, 16};
        ref2 = std::vector<TypeParam>{1, 2, 3, 4};
        EXPECT_EQ(target, ref);
        EXPECT_EQ(data, ref2);
    } else if (my_rank == 2) {
        ref = std::vector<TypeParam>{1, 2, 3, 4};
        ref2 = std::vector<TypeParam>{9, 10, 11, 12};
        EXPECT_EQ(result, ref);
        EXPECT_EQ(target, ref2);
        EXPECT_EQ(data, ref2);
    }
}


TYPED_TEST(MpiBindings, CanFetchAndOperate)
{
    using window = gko::experimental::mpi::window<TypeParam>;
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    std::vector<TypeParam> data;
    std::vector<TypeParam> target;
    std::vector<TypeParam> result(4, 0);
    if (my_rank == 0) {
        data = std::vector<TypeParam>{1, 2, 3, 4};
        target = std::vector<TypeParam>{1, 2, 3, 4};
    } else if (my_rank == 1) {
        data = std::vector<TypeParam>{5, 6, 7, 8};
        target = std::vector<TypeParam>{5, 6, 7, 8};
    } else if (my_rank == 2) {
        data = std::vector<TypeParam>{9, 10, 11, 12};
        target = std::vector<TypeParam>{9, 10, 11, 12};
    } else {
        data = std::vector<TypeParam>{0, 0, 0, 0};
        target = std::vector<TypeParam>{0, 0, 0, 0};
    }

    {
        auto win = window(this->ref, target.data(), 4, comm);

        if (my_rank == 2) {
            win.lock_all();
            win.fetch_and_op(this->ref, data.data(), result.data(), 0, 1,
                             MPI_SUM);
            win.unlock_all();
        }
    }

    std::vector<TypeParam> ref;
    std::vector<TypeParam> ref2;
    if (my_rank == 0) {
        ref = std::vector<TypeParam>{1, 11, 3, 4};
        EXPECT_EQ(target, ref);
    } else if (my_rank == 2) {
        ref = std::vector<TypeParam>{2, 0, 0, 0};
        EXPECT_EQ(result, ref);
    }
}


TYPED_TEST(MpiBindings, CanBroadcastValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto array = gko::array<TypeParam>{this->ref, 8};
    if (my_rank == 0) {
        array = gko::array<TypeParam>(this->ref, {2, 3, 1, 3, -1, 0, 3, 1});
    }

    comm.broadcast(this->ref, array.get_data(), 8, 0);

    auto ref = gko::array<TypeParam>(this->ref, {2, 3, 1, 3, -1, 0, 3, 1});
    GKO_ASSERT_ARRAY_EQ(ref, array);
}


TYPED_TEST(MpiBindings, CanNonBlockingBroadcastValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto array = gko::array<TypeParam>{this->ref, 8};
    if (my_rank == 0) {
        array = gko::array<TypeParam>(this->ref, {2, 3, 1, 3, -1, 0, 3, 1});
    }

    auto req = comm.i_broadcast(this->ref, array.get_data(), 8, 0);

    req.wait();
    auto ref = gko::array<TypeParam>(this->ref, {2, 3, 1, 3, -1, 0, 3, 1});
    GKO_ASSERT_ARRAY_EQ(ref, array);
}


TYPED_TEST(MpiBindings, CanReduceValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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

    comm.reduce(this->ref, &data, &sum, 1, MPI_SUM, 0);
    comm.reduce(this->ref, &data, &max, 1, MPI_MAX, 0);
    comm.reduce(this->ref, &data, &min, 1, MPI_MIN, 0);

    if (my_rank == 0) {
        EXPECT_EQ(sum, TypeParam{16});
        EXPECT_EQ(max, TypeParam{6});
        EXPECT_EQ(min, TypeParam{2});
    }
}


TYPED_TEST(MpiBindings, CanNonBlockingReduceValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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

    auto req1 = comm.i_reduce(this->ref, &data, &sum, 1, MPI_SUM, 0);
    auto req2 = comm.i_reduce(this->ref, &data, &max, 1, MPI_MAX, 0);
    auto req3 = comm.i_reduce(this->ref, &data, &min, 1, MPI_MIN, 0);

    req1.wait();
    req2.wait();
    req3.wait();
    if (my_rank == 0) {
        EXPECT_EQ(sum, TypeParam{16});
        EXPECT_EQ(max, TypeParam{6});
        EXPECT_EQ(min, TypeParam{2});
    }
}


TYPED_TEST(MpiBindings, CanAllReduceValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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

    comm.all_reduce(this->ref, &data, &sum, 1, MPI_SUM);

    ASSERT_EQ(sum, TypeParam{16});
}


TYPED_TEST(MpiBindings, CanAllReduceValuesInPlace)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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

    comm.all_reduce(this->ref, &data, 1, MPI_SUM);

    ASSERT_EQ(data, TypeParam{16});
}


TYPED_TEST(MpiBindings, CanNonBlockingAllReduceValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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

    auto req = comm.i_all_reduce(this->ref, &data, &sum, 1, MPI_SUM);

    req.wait();
    ASSERT_EQ(sum, TypeParam{16});
}


TYPED_TEST(MpiBindings, CanNonBlockingAllReduceValuesInPlace)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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

    auto req = comm.i_all_reduce(this->ref, &data, 1, MPI_SUM);

    req.wait();
    ASSERT_EQ(data, TypeParam{16});
}


TYPED_TEST(MpiBindings, CanGatherValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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
    auto gather_array = gko::array<TypeParam>{
        this->ref, static_cast<gko::size_type>(num_ranks)};

    comm.gather(this->ref, &data, 1, gather_array.get_data(), 1, 0);

    if (my_rank == 0) {
        auto ref = gko::array<TypeParam>(this->ref, {3, 5, 2, 6});
        GKO_ASSERT_ARRAY_EQ(ref, gather_array);
    }
}


TYPED_TEST(MpiBindings, CanNonBlockingGatherValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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
    auto gather_array = gko::array<TypeParam>{
        this->ref, static_cast<gko::size_type>(num_ranks)};

    auto req =
        comm.i_gather(this->ref, &data, 1, gather_array.get_data(), 1, 0);

    req.wait();
    if (my_rank == 0) {
        auto ref = gko::array<TypeParam>(this->ref, {3, 5, 2, 6});
        GKO_ASSERT_ARRAY_EQ(ref, gather_array);
    }
}


TYPED_TEST(MpiBindings, CanAllGatherValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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
    auto gather_array = gko::array<TypeParam>{
        this->ref, static_cast<gko::size_type>(num_ranks)};

    comm.all_gather(this->ref, &data, 1, gather_array.get_data(), 1);

    auto ref = gko::array<TypeParam>(this->ref, {3, 5, 2, 6});
    GKO_ASSERT_ARRAY_EQ(ref, gather_array);
}


TYPED_TEST(MpiBindings, CanNonBlockingAllGatherValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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
    auto gather_array = gko::array<TypeParam>{
        this->ref, static_cast<gko::size_type>(num_ranks)};

    auto req =
        comm.i_all_gather(this->ref, &data, 1, gather_array.get_data(), 1);

    req.wait();
    auto ref = gko::array<TypeParam>(this->ref, {3, 5, 2, 6});
    GKO_ASSERT_ARRAY_EQ(ref, gather_array);
}


TYPED_TEST(MpiBindings, CanGatherValuesWithDisplacements)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto gather_from_array = gko::array<TypeParam>{this->ref};
    auto gather_into_array = gko::array<TypeParam>{this->ref};
    auto r_counts =
        gko::array<int>{this->ref, static_cast<gko::size_type>(num_ranks)};
    auto displacements = gko::array<int>{this->ref};
    int nelems;
    if (my_rank == 0) {
        gather_from_array = gko::array<TypeParam>{this->ref, {2, 3}};
        nelems = 2;
        displacements = gko::array<int>{this->ref, {0, 2, 6, 7}};
        gather_into_array = gko::array<TypeParam>{this->ref, 10};
    } else if (my_rank == 1) {
        nelems = 4;
        gather_from_array = gko::array<TypeParam>{this->ref, {1, 2, 1, 0}};
    } else if (my_rank == 2) {
        nelems = 1;
        gather_from_array = gko::array<TypeParam>{this->ref, {1}};
    } else if (my_rank == 3) {
        nelems = 3;
        gather_from_array = gko::array<TypeParam>{this->ref, {1, -4, 5}};
    }

    comm.gather(this->ref, &nelems, 1, r_counts.get_data(), 1, 0);
    comm.gather_v(this->ref, gather_from_array.get_data(), nelems,
                  gather_into_array.get_data(), r_counts.get_data(),
                  displacements.get_data(), 0);

    auto comp_data = gather_into_array.get_data();
    if (my_rank == 0) {
        auto ref_array =
            gko::array<TypeParam>(this->ref, {2, 3, 1, 2, 1, 0, 1, 1, -4, 5});
        GKO_ASSERT_ARRAY_EQ(gather_into_array, ref_array);
    } else {
        ASSERT_EQ(comp_data, nullptr);
    }
}


TYPED_TEST(MpiBindings, CanNonBlockingGatherValuesWithDisplacements)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto gather_from_array = gko::array<TypeParam>{this->ref};
    auto gather_into_array = gko::array<TypeParam>{this->ref};
    auto r_counts =
        gko::array<int>{this->ref, static_cast<gko::size_type>(num_ranks)};
    auto displacements = gko::array<int>{this->ref};
    int nelems;
    if (my_rank == 0) {
        gather_from_array = gko::array<TypeParam>{this->ref, {2, 3}};
        nelems = 2;
        displacements = gko::array<int>{this->ref, {0, 2, 6, 7}};
        gather_into_array = gko::array<TypeParam>{this->ref, 10};
    } else if (my_rank == 1) {
        nelems = 4;
        gather_from_array = gko::array<TypeParam>{this->ref, {1, 2, 1, 0}};
    } else if (my_rank == 2) {
        nelems = 1;
        gather_from_array = gko::array<TypeParam>{this->ref, {1}};
    } else if (my_rank == 3) {
        nelems = 3;
        gather_from_array = gko::array<TypeParam>{this->ref, {1, -4, 5}};
    }

    comm.gather(this->ref, &nelems, 1, r_counts.get_data(), 1, 0);
    auto req =
        comm.i_gather_v(this->ref, gather_from_array.get_data(), nelems,
                        gather_into_array.get_data(), r_counts.get_data(),
                        displacements.get_data(), 0);

    req.wait();
    auto comp_data = gather_into_array.get_data();
    if (my_rank == 0) {
        auto ref_array =
            gko::array<TypeParam>(this->ref, {2, 3, 1, 2, 1, 0, 1, 1, -4, 5});
        GKO_ASSERT_ARRAY_EQ(gather_into_array, ref_array);
    } else {
        ASSERT_EQ(comp_data, nullptr);
    }
}


TYPED_TEST(MpiBindings, CanScatterValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto scatter_from_array = gko::array<TypeParam>{this->ref};
    if (my_rank == 0) {
        scatter_from_array =
            gko::array<TypeParam>{this->ref, {2, 3, 1, 3, -1, 0, 3, 1}};
    }
    auto scatter_into_array = gko::array<TypeParam>{this->ref, 2};

    comm.scatter(this->ref, scatter_from_array.get_data(), 2,
                 scatter_into_array.get_data(), 2, 0);

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


TYPED_TEST(MpiBindings, CanNonBlockingScatterValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto scatter_from_array = gko::array<TypeParam>{this->ref};
    if (my_rank == 0) {
        scatter_from_array =
            gko::array<TypeParam>{this->ref, {2, 3, 1, 3, -1, 0, 3, 1}};
    }
    auto scatter_into_array = gko::array<TypeParam>{this->ref, 2};

    auto req = comm.i_scatter(this->ref, scatter_from_array.get_data(), 2,
                              scatter_into_array.get_data(), 2, 0);

    req.wait();
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


TYPED_TEST(MpiBindings, CanScatterValuesWithDisplacements)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto scatter_from_array = gko::array<TypeParam>{this->ref};
    auto scatter_into_array = gko::array<TypeParam>{this->ref};
    auto s_counts = gko::array<int>{this->ref->get_master(),
                                    static_cast<gko::size_type>(num_ranks)};
    auto displacements = gko::array<int>{this->ref->get_master()};
    int nelems;
    if (my_rank == 0) {
        scatter_from_array =
            gko::array<TypeParam>{this->ref, {2, 3, 1, 3, -1, 0, 2, -1, 0, 3}};
        nelems = 2;
        displacements = gko::array<int>{this->ref, {0, 2, 6, 9}};
    } else if (my_rank == 1) {
        nelems = 4;
    } else if (my_rank == 2) {
        nelems = 3;
    } else if (my_rank == 3) {
        nelems = 1;
    }
    scatter_into_array =
        gko::array<TypeParam>{this->ref, static_cast<gko::size_type>(nelems)};

    comm.gather(this->ref, &nelems, 1, s_counts.get_data(), 1, 0);
    comm.scatter_v(this->ref, scatter_from_array.get_data(),
                   s_counts.get_data(), displacements.get_data(),
                   scatter_into_array.get_data(), nelems, 0);

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


TYPED_TEST(MpiBindings, CanNonBlockingScatterValuesWithDisplacements)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto scatter_from_array = gko::array<TypeParam>{this->ref};
    auto scatter_into_array = gko::array<TypeParam>{this->ref};
    auto s_counts = gko::array<int>{this->ref->get_master(),
                                    static_cast<gko::size_type>(num_ranks)};
    auto displacements = gko::array<int>{this->ref->get_master()};
    int nelems;
    if (my_rank == 0) {
        scatter_from_array =
            gko::array<TypeParam>{this->ref, {2, 3, 1, 3, -1, 0, 2, -1, 0, 3}};
        nelems = 2;
        displacements = gko::array<int>{this->ref, {0, 2, 6, 9}};
    } else if (my_rank == 1) {
        nelems = 4;
    } else if (my_rank == 2) {
        nelems = 3;
    } else if (my_rank == 3) {
        nelems = 1;
    }
    scatter_into_array =
        gko::array<TypeParam>{this->ref, static_cast<gko::size_type>(nelems)};

    comm.gather(this->ref, &nelems, 1, s_counts.get_data(), 1, 0);
    auto req = comm.i_scatter_v(this->ref, scatter_from_array.get_data(),
                                s_counts.get_data(), displacements.get_data(),
                                scatter_into_array.get_data(), nelems, 0);

    req.wait();
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


TYPED_TEST(MpiBindings, AllToAllWorksCorrectly)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto send_array = gko::array<TypeParam>{this->ref};
    auto recv_array = gko::array<TypeParam>{this->ref};
    auto ref_array = gko::array<TypeParam>{this->ref};
    recv_array = gko::array<TypeParam>{this->ref, 4};
    if (my_rank == 0) {
        send_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::array<TypeParam>(this->ref, {2, 2, 2, 5});
    } else if (my_rank == 1) {
        send_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::array<TypeParam>(this->ref, {3, 3, 3, 3});
    } else if (my_rank == 2) {
        send_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 0});
        ref_array = gko::array<TypeParam>(this->ref, {1, 1, 1, 3});
    } else if (my_rank == 3) {
        send_array = gko::array<TypeParam>(this->ref, {5, 3, 3, -2});
        ref_array = gko::array<TypeParam>(this->ref, {2, 2, 0, -2});
    }

    comm.all_to_all(this->ref, send_array.get_data(), 1, recv_array.get_data(),
                    1);

    GKO_ASSERT_ARRAY_EQ(recv_array, ref_array);
}


TYPED_TEST(MpiBindings, NonBlockingAllToAllWorksCorrectly)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto send_array = gko::array<TypeParam>{this->ref};
    auto recv_array = gko::array<TypeParam>{this->ref};
    auto ref_array = gko::array<TypeParam>{this->ref};
    recv_array = gko::array<TypeParam>{this->ref, 4};
    if (my_rank == 0) {
        send_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::array<TypeParam>(this->ref, {2, 2, 2, 5});
    } else if (my_rank == 1) {
        send_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::array<TypeParam>(this->ref, {3, 3, 3, 3});
    } else if (my_rank == 2) {
        send_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 0});
        ref_array = gko::array<TypeParam>(this->ref, {1, 1, 1, 3});
    } else if (my_rank == 3) {
        send_array = gko::array<TypeParam>(this->ref, {5, 3, 3, -2});
        ref_array = gko::array<TypeParam>(this->ref, {2, 2, 0, -2});
    }

    auto req = comm.i_all_to_all(this->ref, send_array.get_data(), 1,
                                 recv_array.get_data(), 1);

    req.wait();
    GKO_ASSERT_ARRAY_EQ(recv_array, ref_array);
}


TYPED_TEST(MpiBindings, AllToAllInPlaceWorksCorrectly)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto recv_array = gko::array<TypeParam>{this->ref};
    auto ref_array = gko::array<TypeParam>{this->ref};
    recv_array = gko::array<TypeParam>{this->ref, 4};
    if (my_rank == 0) {
        recv_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::array<TypeParam>(this->ref, {2, 2, 2, 5});
    } else if (my_rank == 1) {
        recv_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::array<TypeParam>(this->ref, {3, 3, 3, 3});
    } else if (my_rank == 2) {
        recv_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 0});
        ref_array = gko::array<TypeParam>(this->ref, {1, 1, 1, 3});
    } else if (my_rank == 3) {
        recv_array = gko::array<TypeParam>(this->ref, {5, 3, 3, -2});
        ref_array = gko::array<TypeParam>(this->ref, {2, 2, 0, -2});
    }

    comm.all_to_all(this->ref, recv_array.get_data(), 1);
    GKO_ASSERT_ARRAY_EQ(recv_array, ref_array);
}


TYPED_TEST(MpiBindings, NonBlockingAllToAllInPlaceWorksCorrectly)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto recv_array = gko::array<TypeParam>{this->ref};
    auto ref_array = gko::array<TypeParam>{this->ref};
    recv_array = gko::array<TypeParam>{this->ref, 4};
    if (my_rank == 0) {
        recv_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::array<TypeParam>(this->ref, {2, 2, 2, 5});
    } else if (my_rank == 1) {
        recv_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 2});
        ref_array = gko::array<TypeParam>(this->ref, {3, 3, 3, 3});
    } else if (my_rank == 2) {
        recv_array = gko::array<TypeParam>(this->ref, {2, 3, 1, 0});
        ref_array = gko::array<TypeParam>(this->ref, {1, 1, 1, 3});
    } else if (my_rank == 3) {
        recv_array = gko::array<TypeParam>(this->ref, {5, 3, 3, -2});
        ref_array = gko::array<TypeParam>(this->ref, {2, 2, 0, -2});
    }

    auto req = comm.i_all_to_all(this->ref, recv_array.get_data(), 1);

    req.wait();
    GKO_ASSERT_ARRAY_EQ(recv_array, ref_array);
}


TYPED_TEST(MpiBindings, AllToAllVWorksCorrectly)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto send_array = gko::array<TypeParam>{this->ref};
    auto recv_array = gko::array<TypeParam>{this->ref};
    auto ref_array = gko::array<TypeParam>{this->ref};
    auto scounts_array = gko::array<int>{this->ref};
    auto soffset_array = gko::array<int>{this->ref};
    auto rcounts_array = gko::array<int>{this->ref};
    auto roffset_array = gko::array<int>{this->ref};
    if (my_rank == 0) {
        recv_array = gko::array<TypeParam>{this->ref, {0, 0, 0, 0, 0, 0}};
        send_array = gko::array<TypeParam>{this->ref, {2, 3, 1, 2}};
        scounts_array = gko::array<int>{this->ref, {1, 2, 1, 0}};
        rcounts_array = gko::array<int>{this->ref, {1, 2, 2, 1}};
        soffset_array = gko::array<int>{this->ref, {0, 1, 1, 0}};
        roffset_array = gko::array<int>{this->ref, {0, 1, 3, 5}};
        ref_array = gko::array<TypeParam>{this->ref, {2, 2, 3, 1, 2, 5}};
    } else if (my_rank == 1) {
        recv_array = gko::array<TypeParam>{this->ref, {0, 0, 0, 0, 0, 0}};
        send_array = gko::array<TypeParam>{this->ref, {2, 3, 1, 2}};
        scounts_array = gko::array<int>{this->ref, {2, 2, 1, 2}};
        rcounts_array = gko::array<int>{this->ref, {2, 2, 2, 0}};
        soffset_array = gko::array<int>{this->ref, {0, 1, 1, 0}};
        roffset_array = gko::array<int>{this->ref, {0, 2, 4, 5}};
        ref_array = gko::array<TypeParam>{this->ref, {3, 1, 3, 1, 3, 1}};
    } else if (my_rank == 2) {
        recv_array = gko::array<TypeParam>{this->ref, {0, 0, 0, 0}};
        send_array = gko::array<TypeParam>{this->ref, {2, 3, 1, 2}};
        scounts_array = gko::array<int>{this->ref, {2, 2, 1, 1}};
        rcounts_array = gko::array<int>{this->ref, {1, 1, 1, 1}};
        soffset_array = gko::array<int>{this->ref, {2, 1, 1, 1}};
        roffset_array = gko::array<int>{this->ref, {0, 1, 2, 3}};
        ref_array = gko::array<TypeParam>{this->ref, {3, 3, 3, 3}};
    } else if (my_rank == 3) {
        recv_array = gko::array<TypeParam>{this->ref, {0, 0, 0, 0}};
        send_array = gko::array<TypeParam>{this->ref, {5, 3, 3, -2}};
        scounts_array = gko::array<int>{this->ref, {1, 0, 1, 0}};
        rcounts_array = gko::array<int>{this->ref, {0, 2, 1, 0}};
        soffset_array = gko::array<int>{this->ref, {0, 1, 1, 0}};
        roffset_array = gko::array<int>{this->ref, {0, 1, 3, 3}};
        ref_array = gko::array<TypeParam>{this->ref, {0, 2, 3, 3}};
    }

    comm.all_to_all_v(this->ref, send_array.get_data(),
                      scounts_array.get_data(), soffset_array.get_data(),
                      recv_array.get_data(), rcounts_array.get_data(),
                      roffset_array.get_data());
    GKO_ASSERT_ARRAY_EQ(recv_array, ref_array);
}


TYPED_TEST(MpiBindings, NonBlockingAllToAllVWorksCorrectly)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    auto my_rank = comm.rank();
    auto num_ranks = comm.size();
    auto send_array = gko::array<TypeParam>{this->ref};
    auto recv_array = gko::array<TypeParam>{this->ref};
    auto ref_array = gko::array<TypeParam>{this->ref};
    auto scounts_array = gko::array<int>{this->ref};
    auto soffset_array = gko::array<int>{this->ref};
    auto rcounts_array = gko::array<int>{this->ref};
    auto roffset_array = gko::array<int>{this->ref};
    if (my_rank == 0) {
        recv_array = gko::array<TypeParam>{this->ref, {0, 0, 0, 0, 0, 0}};
        send_array = gko::array<TypeParam>{this->ref, {2, 3, 1, 2}};
        scounts_array = gko::array<int>{this->ref, {1, 2, 1, 0}};
        rcounts_array = gko::array<int>{this->ref, {1, 2, 2, 1}};
        soffset_array = gko::array<int>{this->ref, {0, 1, 1, 0}};
        roffset_array = gko::array<int>{this->ref, {0, 1, 3, 5}};
        ref_array = gko::array<TypeParam>{this->ref, {2, 2, 3, 1, 2, 5}};
    } else if (my_rank == 1) {
        recv_array = gko::array<TypeParam>{this->ref, {0, 0, 0, 0, 0, 0}};
        send_array = gko::array<TypeParam>{this->ref, {2, 3, 1, 2}};
        scounts_array = gko::array<int>{this->ref, {2, 2, 1, 2}};
        rcounts_array = gko::array<int>{this->ref, {2, 2, 2, 0}};
        soffset_array = gko::array<int>{this->ref, {0, 1, 1, 0}};
        roffset_array = gko::array<int>{this->ref, {0, 2, 4, 5}};
        ref_array = gko::array<TypeParam>{this->ref, {3, 1, 3, 1, 3, 1}};
    } else if (my_rank == 2) {
        recv_array = gko::array<TypeParam>{this->ref, {0, 0, 0, 0}};
        send_array = gko::array<TypeParam>{this->ref, {2, 3, 1, 2}};
        scounts_array = gko::array<int>{this->ref, {2, 2, 1, 1}};
        rcounts_array = gko::array<int>{this->ref, {1, 1, 1, 1}};
        soffset_array = gko::array<int>{this->ref, {2, 1, 1, 1}};
        roffset_array = gko::array<int>{this->ref, {0, 1, 2, 3}};
        ref_array = gko::array<TypeParam>{this->ref, {3, 3, 3, 3}};
    } else if (my_rank == 3) {
        recv_array = gko::array<TypeParam>{this->ref, {0, 0, 0, 0}};
        send_array = gko::array<TypeParam>{this->ref, {5, 3, 3, -2}};
        scounts_array = gko::array<int>{this->ref, {1, 0, 1, 0}};
        rcounts_array = gko::array<int>{this->ref, {0, 2, 1, 0}};
        soffset_array = gko::array<int>{this->ref, {0, 1, 1, 0}};
        roffset_array = gko::array<int>{this->ref, {0, 1, 3, 3}};
        ref_array = gko::array<TypeParam>{this->ref, {0, 2, 3, 3}};
    }

    auto req = comm.i_all_to_all_v(
        this->ref, send_array.get_data(), scounts_array.get_data(),
        soffset_array.get_data(), recv_array.get_data(),
        rcounts_array.get_data(), roffset_array.get_data());

    req.wait();
    GKO_ASSERT_ARRAY_EQ(recv_array, ref_array);
}


TYPED_TEST(MpiBindings, CanScanValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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

    comm.scan(this->ref, &data, &sum, 1, MPI_SUM);
    comm.scan(this->ref, &data, &max, 1, MPI_MAX);
    comm.scan(this->ref, &data, &min, 1, MPI_MIN);

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


TYPED_TEST(MpiBindings, CanNonBlockingScanValues)
{
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
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

    auto req1 = comm.i_scan(this->ref, &data, &sum, 1, MPI_SUM);
    auto req2 = comm.i_scan(this->ref, &data, &max, 1, MPI_MAX);
    auto req3 = comm.i_scan(this->ref, &data, &min, 1, MPI_MIN);

    req1.wait();
    req2.wait();
    req3.wait();
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
