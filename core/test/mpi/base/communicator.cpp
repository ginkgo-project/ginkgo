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

    void on_mpi_send_started(const gko::Executor* exec, const int& dest,
                             const gko::size_type& count,
                             const gko::size_type& bytes) const override
    {
        send_started++;
    }
    void on_mpi_send_completed(const gko::Executor* exec, const int& dest,
                               const gko::size_type& count,
                               const gko::size_type& bytes) const override
    {
        send_completed++;
    }
    void on_mpi_recv_started(const gko::Executor* exec, const int& src,
                             const gko::size_type& count,
                             const gko::size_type& bytes) const override
    {
        recv_started++;
    }
    void on_mpi_recv_completed(const gko::Executor* exec, const int& src,
                               const gko::size_type& count,
                               const gko::size_type& bytes) const override
    {
        recv_completed++;
    }
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

    void on_mpi_broadcast_started(const gko::Executor* exec,
                                  const int& root_rank,
                                  const gko::size_type& count,
                                  const gko::size_type& bytes) const override
    {
        broadcast_started++;
    }
    void on_mpi_broadcast_completed(const gko::Executor* exec,
                                    const int& root_rank,
                                    const gko::size_type& count,
                                    const gko::size_type& bytes) const override
    {
        broadcast_completed++;
    }

    void on_mpi_reduce_started(const gko::Executor* exec, const int& root_rank,
                               const gko::size_type& count,
                               const gko::size_type& bytes) const override
    {
        reduce_started++;
    }
    void on_mpi_reduce_completed(const gko::Executor* exec,
                                 const int& root_rank,
                                 const gko::size_type& count,
                                 const gko::size_type& bytes) const override
    {
        reduce_completed++;
    }

    void on_mpi_gather_started(const gko::Executor* exec, const int& root_rank,
                               const gko::size_type& send_count,
                               const gko::size_type& send_bytes,
                               const gko::size_type& recv_count,
                               const gko::size_type& recv_bytes) const override
    {
        gather_started++;
    }
    void on_mpi_gather_completed(
        const gko::Executor* exec, const int& root_rank,
        const gko::size_type& send_count, const gko::size_type& send_bytes,
        const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        gather_completed++;
    }

    void on_mpi_scatter_started(const gko::Executor* exec, const int& root_rank,
                                const gko::size_type& send_count,
                                const gko::size_type& send_bytes,
                                const gko::size_type& recv_count,
                                const gko::size_type& recv_bytes) const override
    {
        scatter_started++;
    }
    void on_mpi_scatter_completed(
        const gko::Executor* exec, const int& root_rank,
        const gko::size_type& send_count, const gko::size_type& send_bytes,
        const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        scatter_completed++;
    }

    void on_mpi_all_gather_started(
        const gko::Executor* exec, const gko::size_type& send_count,
        const gko::size_type& send_bytes, const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        all_gather_started++;
    }
    void on_mpi_all_gather_completed(
        const gko::Executor* exec, const gko::size_type& send_count,
        const gko::size_type& send_bytes, const gko::size_type& recv_count,
        const gko::size_type& recv_bytes) const override
    {
        all_gather_completed++;
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

    void on_mpi_scan_started(const gko::Executor* exec,
                             const gko::size_type& count,
                             const gko::size_type& bytes) const override
    {
        scan_started++;
    }
    void on_mpi_scan_completed(const gko::Executor* exec,
                               const gko::size_type& count,
                               const gko::size_type& bytes) const override
    {
        scan_completed++;
    }

    void on_mpi_barrier_started(const gko::Executor* exec) const override
    {
        barrier_started++;
    }
    void on_mpi_barrier_completed(const gko::Executor* exec) const override
    {
        barrier_completed++;
    }

    mutable int send_started = 0;
    mutable int send_completed = 0;
    mutable int recv_started = 0;
    mutable int recv_completed = 0;
    mutable int all_reduce_started = 0;
    mutable int all_reduce_completed = 0;
    mutable int broadcast_started = 0;
    mutable int broadcast_completed = 0;
    mutable int reduce_started = 0;
    mutable int reduce_completed = 0;
    mutable int gather_started = 0;
    mutable int gather_completed = 0;
    mutable int scatter_started = 0;
    mutable int scatter_completed = 0;
    mutable int all_gather_started = 0;
    mutable int all_gather_completed = 0;
    mutable int all_to_all_started = 0;
    mutable int all_to_all_completed = 0;
    mutable int scan_started = 0;
    mutable int scan_completed = 0;
    mutable int barrier_started = 0;
    mutable int barrier_completed = 0;
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


TEST_F(Communicator, LogsBlockingSendRecv)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);

    int val = 42;
    int tag = 0;

    if (comm.rank() == 0) {
        comm.send(exec, &val, 1, 1, tag);
        EXPECT_EQ(logger->send_started, 1);
        EXPECT_EQ(logger->send_completed, 1);
    } else if (comm.rank() == 1) {
        comm.recv(exec, &val, 1, 0, tag);
        EXPECT_EQ(logger->recv_started, 1);
        EXPECT_EQ(logger->recv_completed, 1);
    }
}


TEST_F(Communicator, LogsNonBlockingSendRecv)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);

    int val = 42;
    int tag = 1;

    if (comm.rank() == 0) {
        auto req = comm.i_send(exec, &val, 1, 1, tag);
        EXPECT_EQ(logger->send_started, 1);
        EXPECT_EQ(logger->send_completed, 0);  // Not completed until wait
        req.wait();
        EXPECT_EQ(logger->send_completed, 1);
    } else if (comm.rank() == 1) {
        auto req = comm.i_recv(exec, &val, 1, 0, tag);
        EXPECT_EQ(logger->recv_started, 1);
        EXPECT_EQ(logger->recv_completed, 0);  // Not completed until wait
        req.wait();
        EXPECT_EQ(logger->recv_completed, 1);
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


TEST_F(Communicator, LogsBlockingBroadcast)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;

    comm.broadcast(exec, &val, 1, 0);

    EXPECT_EQ(logger->broadcast_started, 1);
    EXPECT_EQ(logger->broadcast_completed, 1);
}

TEST_F(Communicator, LogsNonBlockingBroadcast)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;

    auto req = comm.i_broadcast(exec, &val, 1, 0);

    EXPECT_EQ(logger->broadcast_started, 1);
    EXPECT_EQ(logger->broadcast_completed, 0);

    req.wait();

    EXPECT_EQ(logger->broadcast_completed, 1);
}

TEST_F(Communicator, LogsBlockingReduce)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;
    int res = 0;

    comm.reduce(exec, &val, &res, 1, MPI_SUM, 0);

    EXPECT_EQ(logger->reduce_started, 1);
    EXPECT_EQ(logger->reduce_completed, 1);
}

TEST_F(Communicator, LogsNonBlockingReduce)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;
    int res = 0;

    auto req = comm.i_reduce(exec, &val, &res, 1, MPI_SUM, 0);

    EXPECT_EQ(logger->reduce_started, 1);
    EXPECT_EQ(logger->reduce_completed, 0);

    req.wait();

    EXPECT_EQ(logger->reduce_completed, 1);
}

TEST_F(Communicator, LogsBlockingGather)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;
    std::vector<int> res(comm.size());

    comm.gather(exec, &val, 1, res.data(), 1, 0);

    EXPECT_EQ(logger->gather_started, 1);
    EXPECT_EQ(logger->gather_completed, 1);
}

TEST_F(Communicator, LogsNonBlockingGather)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;
    std::vector<int> res(comm.size());

    auto req = comm.i_gather(exec, &val, 1, res.data(), 1, 0);

    EXPECT_EQ(logger->gather_started, 1);
    EXPECT_EQ(logger->gather_completed, 0);

    req.wait();

    EXPECT_EQ(logger->gather_completed, 1);
}

TEST_F(Communicator, LogsBlockingScatter)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    std::vector<int> send_vals(comm.size(), 42);
    int res = 0;

    comm.scatter(exec, send_vals.data(), 1, &res, 1, 0);

    EXPECT_EQ(logger->scatter_started, 1);
    EXPECT_EQ(logger->scatter_completed, 1);
}

TEST_F(Communicator, LogsNonBlockingScatter)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    std::vector<int> send_vals(comm.size(), 42);
    int res = 0;

    auto req = comm.i_scatter(exec, send_vals.data(), 1, &res, 1, 0);

    EXPECT_EQ(logger->scatter_started, 1);
    EXPECT_EQ(logger->scatter_completed, 0);

    req.wait();

    EXPECT_EQ(logger->scatter_completed, 1);
}

TEST_F(Communicator, LogsBlockingAllGather)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;
    std::vector<int> res(comm.size());

    comm.all_gather(exec, &val, 1, res.data(), 1);

    EXPECT_EQ(logger->all_gather_started, 1);
    EXPECT_EQ(logger->all_gather_completed, 1);
}

TEST_F(Communicator, LogsNonBlockingAllGather)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;
    std::vector<int> res(comm.size());

    auto req = comm.i_all_gather(exec, &val, 1, res.data(), 1);

    EXPECT_EQ(logger->all_gather_started, 1);
    EXPECT_EQ(logger->all_gather_completed, 0);

    req.wait();

    EXPECT_EQ(logger->all_gather_completed, 1);
}

TEST_F(Communicator, LogsBlockingAllToAll)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    std::vector<int> send_vals(comm.size(), 42);
    std::vector<int> res(comm.size());

    comm.all_to_all(exec, send_vals.data(), 1, res.data(), 1);

    EXPECT_EQ(logger->all_to_all_started, 1);
    EXPECT_EQ(logger->all_to_all_completed, 1);
}

TEST_F(Communicator, LogsNonBlockingAllToAll)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    std::vector<int> send_vals(comm.size(), 42);
    std::vector<int> res(comm.size());

    auto req = comm.i_all_to_all(exec, send_vals.data(), 1, res.data(), 1);

    EXPECT_EQ(logger->all_to_all_started, 1);
    EXPECT_EQ(logger->all_to_all_completed, 0);

    req.wait();

    EXPECT_EQ(logger->all_to_all_completed, 1);
}

TEST_F(Communicator, LogsBlockingScan)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;
    int res = 0;

    comm.scan(exec, &val, &res, 1, MPI_SUM);

    EXPECT_EQ(logger->scan_started, 1);
    EXPECT_EQ(logger->scan_completed, 1);
}

TEST_F(Communicator, LogsNonBlockingScan)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);
    int val = 42;
    int res = 0;

    auto req = comm.i_scan(exec, &val, &res, 1, MPI_SUM);

    EXPECT_EQ(logger->scan_started, 1);
    EXPECT_EQ(logger->scan_completed, 0);

    req.wait();

    EXPECT_EQ(logger->scan_completed, 1);
}

TEST_F(Communicator, LogsBarrier)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);

    comm.synchronize();

    EXPECT_EQ(logger->barrier_started, 1);
    EXPECT_EQ(logger->barrier_completed, 1);
}


TEST_F(Communicator, AbandonedNonBlockingRequestFiresCompletedOnDestruction)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);

    int val = 42;
    int tag = 7;

    if (comm.rank() == 0) {
        {
            auto req = comm.i_send(exec, &val, 1, 1, tag);
            EXPECT_EQ(logger->send_started, 1);
            EXPECT_EQ(logger->send_completed, 0);
            // intentionally do not call req.wait(); destructor must wait
            // and fire the completed event.
        }
        EXPECT_EQ(logger->send_completed, 1);
    } else if (comm.rank() == 1) {
        int recv_val = 0;
        comm.recv(exec, &recv_val, 1, 0, tag);
    }
}


TEST_F(Communicator, MoveAssigningOntoLiveRequestFiresItsCompletedEvent)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = std::make_shared<TestLogger>();
    comm.add_logger(logger);

    int val_a = 1;
    int val_b = 2;
    int tag_a = 11;
    int tag_b = 12;

    if (comm.rank() == 0) {
        auto req1 = comm.i_send(exec, &val_a, 1, 1, tag_a);
        auto req2 = comm.i_send(exec, &val_b, 1, 1, tag_b);
        EXPECT_EQ(logger->send_started, 2);
        EXPECT_EQ(logger->send_completed, 0);
        // Move-assigning over a live request must wait on req1 and fire its
        // completed event, instead of silently leaking the in-flight op.
        req1 = std::move(req2);
        EXPECT_EQ(logger->send_completed, 1);
        req1.wait();
        EXPECT_EQ(logger->send_completed, 2);
    } else if (comm.rank() == 1) {
        int recv_a = 0;
        int recv_b = 0;
        comm.recv(exec, &recv_a, 1, 0, tag_a);
        comm.recv(exec, &recv_b, 1, 0, tag_b);
    }
}


TEST_F(Communicator, MultipleLoggersAllReceiveCompletedEvent)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger_a = std::make_shared<TestLogger>();
    auto logger_b = std::make_shared<TestLogger>();
    comm.add_logger(logger_a);
    comm.add_logger(logger_b);

    int val = 42;
    int tag = 9;

    if (comm.rank() == 0) {
        auto req = comm.i_send(exec, &val, 1, 1, tag);
        req.wait();
        EXPECT_EQ(logger_a->send_completed, 1);
        EXPECT_EQ(logger_b->send_completed, 1);
    } else if (comm.rank() == 1) {
        int recv_val = 0;
        comm.recv(exec, &recv_val, 1, 0, tag);
    }
}

}  // namespace
