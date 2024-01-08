// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// force-top: on
#include <ginkgo/core/base/types.hpp>
GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS
// force-top: off


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
            nullptr, nullptr, nullptr, num_iters, nullptr, nullptr, nullptr,
            nullptr, false);
    }

    void deprecated_apply_1()
    {
        this->log<gko::log::Logger::iteration_complete>(
            nullptr, num_iters, nullptr, nullptr, nullptr);
    }

    void deprecated_apply_2()
    {
        this->log<gko::log::Logger::iteration_complete>(
            nullptr, num_iters, nullptr, nullptr, nullptr, nullptr);
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
    auto l = std::make_shared<DummyLogger>(
        false, gko::log::Logger::iteration_complete_mask);
    DummyLoggedClass c{exec};
    c.add_logger(l);

    c.apply();

    ASSERT_EQ(num_iters, l->num_iterations_);
}


TEST(DummyLogged, CanLogEventsDeprecated1)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::make_shared<DummyLogger>(
        false, gko::log::Logger::iteration_complete_mask);
    DummyLoggedClass c{exec};
    c.add_logger(l);

    c.deprecated_apply_1();

    ASSERT_EQ(num_iters, l->num_iterations_);
}


TEST(DummyLogged, CanLogEventsDeprecated2)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::make_shared<DummyLogger>(
        false, gko::log::Logger::iteration_complete_mask);
    DummyLoggedClass c{exec};
    c.add_logger(l);

    c.deprecated_apply_2();

    ASSERT_EQ(num_iters, l->num_iterations_);
}


TEST(DummyLogged, DoesNotPropagateEventsWhenNotPropagating)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::make_shared<DummyLogger>(
        false, gko::log::Logger::iteration_complete_mask);
    DummyLoggedClass c{exec};
    exec->add_logger(l);

    c.apply();

    ASSERT_EQ(0, l->num_iterations_);
}


TEST(DummyLogged, PropagatesEventsWhenPropagating)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::make_shared<DummyLogger>(
        true, gko::log::Logger::iteration_complete_mask);
    DummyLoggedClass c{exec};
    exec->add_logger(l);

    c.apply();

    ASSERT_EQ(num_iters, l->num_iterations_);
}


TEST(DummyLogged, DoesNotPropagateEventsWhenDisabled)
{
    auto exec = gko::ReferenceExecutor::create();
    exec->set_log_propagation_mode(gko::log_propagation_mode::never);
    auto l = std::make_shared<DummyLogger>(
        true, gko::log::Logger::iteration_complete_mask);
    DummyLoggedClass c{exec};
    exec->add_logger(l);

    c.apply();

    ASSERT_EQ(0, l->num_iterations_);
}


struct DummyLoggerExtended : gko::log::Logger {
    using Logger = gko::log::Logger;

    explicit DummyLoggerExtended() : Logger(Logger::iteration_complete_mask) {}

    void on_iteration_complete(
        const gko::LinOp* solver, const gko::size_type& num_iterations,
        const gko::LinOp* residual, const gko::LinOp* solution = nullptr,
        const gko::LinOp* residual_norm = nullptr) const override
    {
        this->logged_deprecated_1 = true;
    }

    void on_iteration_complete(const gko::LinOp* solver,
                               const gko::size_type& it, const gko::LinOp* r,
                               const gko::LinOp* x, const gko::LinOp* tau,
                               const gko::LinOp* implicit_tau_sq) const override
    {
        this->logged_deprecated_2 = true;
    }

    void on_iteration_complete(const gko::LinOp* solver, const gko::LinOp* b,
                               const gko::LinOp* x, const gko::size_type& it,
                               const gko::LinOp* r, const gko::LinOp* tau,
                               const gko::LinOp* implicit_tau_sq,
                               const gko::array<gko::stopping_status>* status,
                               bool stopped) const override
    {
        this->logged_current = true;
    }

    mutable bool logged_deprecated_1 = false;
    mutable bool logged_deprecated_2 = false;
    mutable bool logged_current = false;
};


TEST(IterationCompleteOverload, CanLogFirstDeprecated)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::make_shared<DummyLoggerExtended>();
    DummyLoggedClass c{exec};
    c.add_logger(l);

    c.deprecated_apply_1();

    ASSERT_TRUE(l->logged_deprecated_1);
    ASSERT_FALSE(l->logged_deprecated_2);
    ASSERT_FALSE(l->logged_current);
}


TEST(IterationCompleteOverload, CanLogSecondDeprecated)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::make_shared<DummyLoggerExtended>();
    DummyLoggedClass c{exec};
    c.add_logger(l);

    c.deprecated_apply_2();

    ASSERT_FALSE(l->logged_deprecated_1);
    ASSERT_TRUE(l->logged_deprecated_2);
    ASSERT_FALSE(l->logged_current);
}


TEST(IterationCompleteOverload, CanLogCurrent)
{
    auto exec = gko::ReferenceExecutor::create();
    auto l = std::make_shared<DummyLoggerExtended>();
    DummyLoggedClass c{exec};
    c.add_logger(l);

    c.apply();

    ASSERT_FALSE(l->logged_deprecated_1);
    ASSERT_FALSE(l->logged_deprecated_2);
    ASSERT_TRUE(l->logged_current);
}


}  // namespace


GKO_END_DISABLE_DEPRECATION_WARNINGS
