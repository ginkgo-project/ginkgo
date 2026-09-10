// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace {


class TestLogger : public gko::log::Logger {
public:
    TestLogger() : gko::log::Logger(gko::log::Logger::mpi_events_mask) {}

    void on_mpi_all_reduce_started(const gko::Executor* exec,
                                   const gko::size_type& count,
                                   const gko::size_type& bytes) const override
    {
        all_reduce_started++;
    }
    void on_mpi_all_reduce_completed(const gko::Executor* exec,
                                     const gko::size_type& count,
                                     const gko::size_type& bytes) const override
    {
        all_reduce_completed++;
    }

    void on_mpi_all_to_all_started(
        const gko::Executor* exec, const gko::size_type& send_count,
        const gko::size_type& send_bytes, const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        all_to_all_started++;
    }
    void on_mpi_all_to_all_completed(
        const gko::Executor* exec, const gko::size_type& send_count,
        const gko::size_type& send_bytes, const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        all_to_all_completed++;
    }

    mutable int all_reduce_started = 0;
    mutable int all_reduce_completed = 0;
    mutable int all_to_all_started = 0;
    mutable int all_to_all_completed = 0;
};


class Communicator : public ::testing::Test {
protected:
    Communicator() : comm(MPI_COMM_WORLD) {}

    void SetUp()
    {
        rank = comm.rank();
        ASSERT_EQ(comm.size(), 8);
    }

    gko::experimental::mpi::communicator comm;
    int rank;
};


TEST_F(Communicator, CommKnowsItsSize)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    EXPECT_EQ(comm.size(), size);
}


TEST_F(Communicator, CommKnowsItsRank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    EXPECT_EQ(comm.rank(), rank);
}


TEST_F(Communicator, CommKnowsItsLocalRank)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Expect local rank to be same as rank when on one node
    EXPECT_EQ(comm.node_local_rank(), rank);
}


TEST_F(Communicator, CommunicatorCanBeCopyConstructed)
{
    gko::experimental::mpi::communicator copy(comm);

    EXPECT_TRUE(copy == comm);
}


TEST_F(Communicator, CommunicatorCanBeCopyAssigned)
{
    gko::experimental::mpi::communicator copy = comm;

    EXPECT_TRUE(copy == comm);
}


TEST_F(Communicator, CommunicatorCanBeMoveConstructed)
{
    gko::experimental::mpi::communicator comm2(MPI_COMM_WORLD);
    gko::experimental::mpi::communicator copy(std::move(comm2));

    EXPECT_TRUE(copy == comm);
}


TEST_F(Communicator, CommunicatorCanBeMoveAssigned)
{
    gko::experimental::mpi::communicator comm2(MPI_COMM_WORLD);
    gko::experimental::mpi::communicator copy(MPI_COMM_NULL);
    copy = std::move(comm2);

    EXPECT_TRUE(copy == comm);
}


TEST_F(Communicator, CommunicatorCanBeSynchronized)
{
    ASSERT_NO_THROW(comm.synchronize());
}


TEST_F(Communicator, CanSetCustomCommunicator)
{
    auto world_rank = comm.rank();
    auto world_size = comm.size();
    auto color = world_rank / 4;

    auto row_comm =
        gko::experimental::mpi::communicator(comm.get(), color, world_rank);
    for (auto i = 0; i < world_size; ++i) {
        EXPECT_LT(row_comm.rank(), 4);
    }
}


TEST_F(Communicator, LogsBlockingAllReduce)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;
    int res = 0;

    comm.all_reduce(exec, &val, &res, 1, MPI_SUM);

    EXPECT_EQ(logger->all_reduce_started, 1);
    EXPECT_EQ(logger->all_reduce_completed, 1);
}


TEST_F(Communicator, LogsNonBlockingAllReduce)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;
    int res = 0;

    auto req = comm.i_all_reduce(exec, &val, &res, 1, MPI_SUM);

    EXPECT_EQ(logger->all_reduce_started, 1);
    EXPECT_EQ(logger->all_reduce_completed, 0);  // Not completed until wait

    req.wait();

    EXPECT_EQ(logger->all_reduce_completed, 1);
}


TEST_F(Communicator, LogsBlockingAllToAllV)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    auto size = comm.size();
    std::vector<int> send_vals(size, 42);
    std::vector<int> counts(size, 1);
    std::vector<int> offsets(size);
    for (int i = 0; i < size; ++i) {
        offsets[i] = i;
    }
    std::vector<int> res(size);

    comm.all_to_all_v(exec, send_vals.data(), counts.data(), offsets.data(),
                      res.data(), counts.data(), offsets.data());

    EXPECT_EQ(logger->all_to_all_started, 1);
    EXPECT_EQ(logger->all_to_all_completed, 1);
}

TEST_F(Communicator, LogsNonBlockingAllToAllV)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    auto size = comm.size();
    std::vector<int> send_vals(size, 42);
    std::vector<int> counts(size, 1);
    std::vector<int> offsets(size);
    for (int i = 0; i < size; ++i) {
        offsets[i] = i;
    }
    std::vector<int> res(size);

    auto req = comm.i_all_to_all_v(exec, send_vals.data(), counts.data(),
                                   offsets.data(), res.data(), counts.data(),
                                   offsets.data());

    EXPECT_EQ(logger->all_to_all_started, 1);
    EXPECT_EQ(logger->all_to_all_completed, 0);

    req.wait();

    EXPECT_EQ(logger->all_to_all_completed, 1);
}


TEST_F(Communicator, DestructorWaitsAndFiresCompletedEvent)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);

    int val = 42;

    {
        auto req = comm.i_all_reduce(exec, &val, 1, MPI_SUM);
        EXPECT_EQ(logger->all_reduce_started, 1);
        EXPECT_EQ(logger->all_reduce_completed, 0);
        // intentionally do not call req.wait(); destructor must wait
        // and fire the completed event.
    }
    EXPECT_EQ(logger->all_reduce_completed, 1);
}


TEST_F(Communicator, MoveAssigningOntoLiveRequestFiresItsCompletedEvent)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);

    int val_a = 1;
    int val_b = 2;

    auto req1 = comm.i_all_reduce(exec, &val_a, 1, MPI_SUM);
    auto req2 = comm.i_all_reduce(exec, &val_b, 1, MPI_SUM);
    EXPECT_EQ(logger->all_reduce_started, 2);
    EXPECT_EQ(logger->all_reduce_completed, 0);
    // Move-assigning over a live request must wait on req1 and fire its
    // completed event, instead of silently leaking the in-flight op.
    req1 = std::move(req2);
    EXPECT_EQ(logger->all_reduce_completed, 1);
    req1.wait();
    EXPECT_EQ(logger->all_reduce_completed, 2);
}


TEST_F(Communicator, MultipleLoggersAllReceiveCompletedEvent)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger_a = std::make_shared<TestLogger>();
    auto logger_b = std::make_shared<TestLogger>();
    comm.add_logger(logger_a);
    comm.add_logger(logger_b);

    int val = 42;

    auto req = comm.i_all_reduce(exec, &val, 1, MPI_SUM);
    req.wait();

    EXPECT_EQ(logger_a->all_reduce_completed, 1);
    EXPECT_EQ(logger_b->all_reduce_completed, 1);
}

}  // namespace
