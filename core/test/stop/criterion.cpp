// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/criterion.hpp>


#include <gtest/gtest.h>


namespace {


struct DummyLogger : public gko::log::Logger {
    DummyLogger() : gko::log::Logger(gko::log::Logger::criterion_events_mask) {}

    void on_criterion_check_started(const gko::stop::Criterion* criterion,
                                    const gko::size_type& num_iterations,
                                    const gko::LinOp* residual,
                                    const gko::LinOp* residual_norm,
                                    const gko::LinOp* solution,
                                    const gko::uint8& stopping_id,
                                    const bool& set_finalized) const override
    {
        criterion_check_started++;
    }

    void on_criterion_check_completed(
        const gko::stop::Criterion* criterion,
        const gko::size_type& num_iterations, const gko::LinOp* residual,
        const gko::LinOp* residual_norm, const gko::LinOp* solution,
        const gko::uint8& stopping_id, const bool& set_finalized,
        const gko::array<gko::stopping_status>* status, const bool& one_changed,
        const bool& all_converged) const override
    {
        criterion_check_completed++;
    }


    mutable int criterion_check_started = 0;
    mutable int criterion_check_completed = 0;
};


class DummyCriterion
    : public gko::EnablePolymorphicObject<DummyCriterion,
                                          gko::stop::Criterion> {
    friend class gko::EnablePolymorphicObject<DummyCriterion,
                                              gko::stop::Criterion>;

public:
    explicit DummyCriterion(std::shared_ptr<const gko::Executor> exec)
        : gko::EnablePolymorphicObject<DummyCriterion, gko::stop::Criterion>(
              std::move(exec))
    {}

protected:
    bool check_impl(gko::uint8 stopping_id, bool set_finalized,
                    gko::array<gko::stopping_status>* stop_status,
                    bool* one_changed, const Updater& updater) override
    {
        return true;
    }
};


class Criterion : public ::testing::Test {
protected:
    Criterion()
        : exec{gko::ReferenceExecutor::create()},
          criterion{std::make_unique<DummyCriterion>(exec)},
          stopping_status{exec},
          logger{std::make_shared<DummyLogger>()}
    {
        criterion->add_logger(logger);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<DummyCriterion> criterion;
    gko::array<gko::stopping_status> stopping_status;
    std::shared_ptr<DummyLogger> logger;
};


TEST_F(Criterion, DefaultUpdateStatus)
{
    EXPECT_EQ(criterion->update().num_iterations_, 0);
    EXPECT_EQ(criterion->update().ignore_residual_check_, false);
    EXPECT_EQ(criterion->update().residual_, nullptr);
    EXPECT_EQ(criterion->update().residual_norm_, nullptr);
    EXPECT_EQ(criterion->update().implicit_sq_residual_norm_, nullptr);
    EXPECT_EQ(criterion->update().solution_, nullptr);
}


TEST_F(Criterion, CanLogCheck)
{
    auto before_logger = *logger;
    bool one_changed = false;

    criterion->check(gko::uint8(0), false, &stopping_status, &one_changed,
                     criterion->update());

    ASSERT_EQ(logger->criterion_check_started,
              before_logger.criterion_check_started + 1);
    ASSERT_EQ(logger->criterion_check_completed,
              before_logger.criterion_check_completed + 1);
}


}  // namespace
