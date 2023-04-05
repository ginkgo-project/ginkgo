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

#include <ginkgo/core/log/logger.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/log/record.hpp>
#include <ginkgo/core/log/stream.hpp>


namespace {


constexpr int num_iters = 10;


struct DummyLoggedClass : gko::log::EnableLogging<DummyLoggedClass> {
    DummyLoggedClass(std::shared_ptr<const gko::Executor> exec)
        : exec{std::move(exec)}
    {}

    int get_num_loggers() { return loggers_.size(); }

    void apply()
    {
        this->log<gko::log::Logger::iteration_complete>(
            nullptr, num_iters, nullptr, nullptr, nullptr);
    }

    std::shared_ptr<const gko::Executor> get_executor() const { return exec; }

    std::shared_ptr<const gko::Executor> exec;
};


TEST(DummyLogged, CanAddLogger)
{
    auto exec = gko::ReferenceExecutor::create();
    DummyLoggedClass c{exec};

    c.add_logger(
        gko::log::Convergence<>::create(gko::log::Logger::all_events_mask));

    ASSERT_EQ(c.get_num_loggers(), 1);
}


TEST(DummyLogged, CanAddMultipleLoggers)
{
    auto exec = gko::ReferenceExecutor::create();
    DummyLoggedClass c{exec};

    c.add_logger(
        gko::log::Convergence<>::create(gko::log::Logger::all_events_mask));
    c.add_logger(gko::log::Stream<>::create(gko::log::Logger::all_events_mask,
                                            std::cout));

    ASSERT_EQ(c.get_num_loggers(), 2);
}


TEST(DummyLogged, CanAccessLoggers)
{
    auto exec = gko::ReferenceExecutor::create();
    DummyLoggedClass c{exec};

    auto logger1 =
        gko::share(gko::log::Record::create(gko::log::Logger::all_events_mask));
    auto logger2 = gko::share(gko::log::Stream<>::create(
        gko::log::Logger::all_events_mask, std::cout));

    c.add_logger(logger1);
    c.add_logger(logger2);

    ASSERT_EQ(c.get_loggers()[0], logger1);
    ASSERT_EQ(c.get_loggers()[1], logger2);
    ASSERT_EQ(c.get_num_loggers(), 2);
}


TEST(DummyLogged, CanClearLoggers)
{
    auto exec = gko::ReferenceExecutor::create();
    DummyLoggedClass c{exec};
    c.add_logger(gko::log::Record::create(gko::log::Logger::all_events_mask));
    c.add_logger(gko::log::Stream<>::create(gko::log::Logger::all_events_mask,
                                            std::cout));

    c.clear_loggers();

    ASSERT_EQ(c.get_num_loggers(), 0);
}


TEST(DummyLogged, CanRemoveLogger)
{
    auto exec = gko::ReferenceExecutor::create();
    DummyLoggedClass c{exec};
    auto r = gko::share(
        gko::log::Convergence<>::create(gko::log::Logger::all_events_mask));
    c.add_logger(r);
    c.add_logger(gko::log::Stream<>::create(gko::log::Logger::all_events_mask,
                                            std::cout));

    c.remove_logger(r);

    ASSERT_EQ(c.get_num_loggers(), 1);
}

struct DummyLogger : gko::log::Logger {
    using Logger = gko::log::Logger;

    explicit DummyLogger(
        bool propagate = false,
        const mask_type& enabled_events = Logger::all_events_mask)
        : Logger(enabled_events), propagate_(propagate)
    {}

    void on_iteration_complete(
        const gko::LinOp* solver, const gko::size_type& num_iterations,
        const gko::LinOp* residual, const gko::LinOp* solution = nullptr,
        const gko::LinOp* residual_norm = nullptr) const override
    {
        this->num_iterations_ = num_iterations;
    }

    bool needs_propagation() const override { return propagate_; }

    bool propagate_{};
    mutable gko::size_type num_iterations_{};
};


TEST(DummyLogged, CanLogEvents)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::shared_ptr<DummyLogger>(
        new DummyLogger(false, gko::log::Logger::iteration_complete_mask));
    DummyLoggedClass c{exec};
    c.add_logger(l);

    c.apply();

    ASSERT_EQ(num_iters, l->num_iterations_);
}


TEST(DummyLogged, DoesNotPropagateEventsWhenNotPropagating)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::shared_ptr<DummyLogger>(
        new DummyLogger(false, gko::log::Logger::iteration_complete_mask));
    DummyLoggedClass c{exec};
    exec->add_logger(l);

    c.apply();

    ASSERT_EQ(0, l->num_iterations_);
}


TEST(DummyLogged, PropagatesEventsWhenPropagating)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::shared_ptr<DummyLogger>(
        new DummyLogger(true, gko::log::Logger::iteration_complete_mask));
    DummyLoggedClass c{exec};
    exec->add_logger(l);

    c.apply();

    ASSERT_EQ(num_iters, l->num_iterations_);
}


TEST(DummyLogged, DoesNotPropagateEventsWhenDisabled)
{
    auto exec = gko::ReferenceExecutor::create();
    exec->set_log_propagation_mode(gko::log_propagate_mode::never);
    auto l = std::shared_ptr<DummyLogger>(
        new DummyLogger(true, gko::log::Logger::iteration_complete_mask));
    DummyLoggedClass c{exec};
    exec->add_logger(l);

    c.apply();

    ASSERT_EQ(0, l->num_iterations_);
}


struct DummyMpiLogger : gko::log::Logger {
    using Logger = gko::log::Logger;

    mutable int blocking_count = 0;
    mutable int non_blocking_count = 0;

    explicit DummyMpiLogger(
        Logger::mpi_mode_mask_type mask = Logger::all_mpi_modes_mask)
        : Logger(Logger::mpi_events_mask, mask)
    {}

    void on_mpi_point_to_point_communication_started(
        const gko::Executor* exec, gko::log::mpi::mode mode, const char* name,
        const void* comm, gko::log::mpi::pt2pt data) const override
    {
        increase_count(mode);
    }

    void on_mpi_point_to_point_communication_completed(
        const gko::Executor* exec, gko::log::mpi::mode mode, const char* name,
        const void* comm, gko::log::mpi::pt2pt data) const override
    {
        increase_count(mode);
    }

protected:
    void increase_count(gko::log::mpi::mode mode) const
    {
        switch (mode.index()) {
        case 0:
            blocking_count++;
            break;
        case 1:
            non_blocking_count++;
            break;
        }
    }
};


TEST(DummyMpiLogger, CanLogBlockingMpiEvents)
{
    auto exec = gko::ReferenceExecutor::create();
    using Logger = gko::log::Logger;
    auto l = std::make_shared<DummyMpiLogger>();

    l->template on<Logger::mpi_point_to_point_communication_started>(
        exec.get(), gko::log::mpi::blocking{}, "", nullptr,
        gko::log::mpi::pt2pt{});

    ASSERT_EQ(l->blocking_count, 1);
}


TEST(DummyMpiLogger, CanLogNonBlockingMpiEvents)
{
    auto exec = gko::ReferenceExecutor::create();
    using Logger = gko::log::Logger;
    auto l = std::make_shared<DummyMpiLogger>();

    l->template on<Logger::mpi_point_to_point_communication_started>(
        exec.get(), gko::log::mpi::non_blocking{}, "", nullptr,
        gko::log::mpi::pt2pt{});

    ASSERT_EQ(l->non_blocking_count, 1);
}


TEST(DummyMpiLogger, CanExclusivlyLogBlockingMpiEvents)
{
    auto exec = gko::ReferenceExecutor::create();
    using Logger = gko::log::Logger;
    auto l = std::make_shared<DummyMpiLogger>(gko::log::mpi_mode::blocking);

    l->template on<Logger::mpi_point_to_point_communication_started>(
        exec.get(), gko::log::mpi::blocking{}, "", nullptr,
        gko::log::mpi::pt2pt{});
    l->template on<Logger::mpi_point_to_point_communication_started>(
        exec.get(), gko::log::mpi::non_blocking{}, "", nullptr,
        gko::log::mpi::pt2pt{});

    ASSERT_EQ(l->blocking_count, 1);
    ASSERT_EQ(l->non_blocking_count, 0);
}


TEST(DummyMpiLogger, CanExclusivlyLogNonBlockingMpiEvents)
{
    auto exec = gko::ReferenceExecutor::create();
    using Logger = gko::log::Logger;
    auto l = std::make_shared<DummyMpiLogger>(gko::log::mpi_mode::non_blocking);

    l->template on<Logger::mpi_point_to_point_communication_started>(
        exec.get(), gko::log::mpi::blocking{}, "", nullptr,
        gko::log::mpi::pt2pt{});
    l->template on<Logger::mpi_point_to_point_communication_started>(
        exec.get(), gko::log::mpi::non_blocking{}, "", nullptr,
        gko::log::mpi::pt2pt{});

    ASSERT_EQ(l->blocking_count, 0);
    ASSERT_EQ(l->non_blocking_count, 1);
}


TEST(DummyMpiLogger, CanLogBlockingAndNonBlockingMpiEventsSimultaneously)
{
    auto exec = gko::ReferenceExecutor::create();
    using Logger = gko::log::Logger;
    auto l = std::make_shared<DummyMpiLogger>();

    l->template on<Logger::mpi_point_to_point_communication_started>(
        exec.get(), gko::log::mpi::blocking{}, "", nullptr,
        gko::log::mpi::pt2pt{});
    l->template on<Logger::mpi_point_to_point_communication_started>(
        exec.get(), gko::log::mpi::non_blocking{}, "", nullptr,
        gko::log::mpi::pt2pt{});

    ASSERT_EQ(l->blocking_count, 1);
    ASSERT_EQ(l->non_blocking_count, 1);
}


}  // namespace
