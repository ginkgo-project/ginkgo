// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/criterion.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/stop/iteration.hpp>


namespace {


constexpr gko::size_type test_iterations = 10;


class Criterion : public ::testing::Test {
protected:
    Criterion()
    {
        exec_ = gko::ReferenceExecutor::create();
        // Actually use an iteration stopping criterion because Criterion is an
        // abstract class
        factory_ = gko::stop::Iteration::build()
                       .with_max_iters(test_iterations)
                       .on(exec_);
    }

    std::unique_ptr<gko::stop::Iteration::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};


TEST_F(Criterion, SetsOneStopStatus)
{
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    gko::array<gko::stopping_status> stop_status(exec_, 1);
    stop_status.get_data()[0].reset();

    criterion->update()
        .num_iterations(test_iterations)
        .check(RelativeStoppingId, true, &stop_status, &one_changed);

    ASSERT_EQ(stop_status.get_data()[0].has_stopped(), true);
}


TEST_F(Criterion, SetsMultipleStopStatuses)
{
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    gko::array<gko::stopping_status> stop_status(exec_, 3);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.get_data()[2].reset();

    criterion->update()
        .num_iterations(test_iterations)
        .check(RelativeStoppingId, true, &stop_status, &one_changed);

    ASSERT_EQ(stop_status.get_data()[0].has_stopped(), true);
    ASSERT_EQ(stop_status.get_data()[1].has_stopped(), true);
    ASSERT_EQ(stop_status.get_data()[2].has_stopped(), true);
}


}  // namespace
