// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "test/utils/executor.hpp"


constexpr gko::size_type test_iterations = 10;


class Criterion : public CommonTestFixture {
protected:
    Criterion()
    {
        // Actually use an iteration stopping criterion because Criterion is an
        // abstract class
        factory = gko::stop::Iteration::build()
                      .with_max_iters(test_iterations)
                      .on(exec);
    }

    std::unique_ptr<gko::stop::Iteration::Factory> factory;
};


TEST_F(Criterion, SetsOneStopStatus)
{
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory->generate(nullptr, nullptr, nullptr);
    gko::array<gko::stopping_status> stop_status(ref, 1);
    stop_status.get_data()[0].reset();

    stop_status.set_executor(exec);
    criterion->update()
        .num_iterations(test_iterations)
        .check(RelativeStoppingId, true, &stop_status, &one_changed);
    stop_status.set_executor(ref);

    ASSERT_EQ(stop_status.get_data()[0].has_stopped(), true);
}


TEST_F(Criterion, SetsMultipleStopStatuses)
{
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory->generate(nullptr, nullptr, nullptr);
    gko::array<gko::stopping_status> stop_status(ref, 3);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.get_data()[2].reset();

    stop_status.set_executor(exec);
    criterion->update()
        .num_iterations(test_iterations)
        .check(RelativeStoppingId, true, &stop_status, &one_changed);
    stop_status.set_executor(ref);

    ASSERT_EQ(stop_status.get_data()[0].has_stopped(), true);
    ASSERT_EQ(stop_status.get_data()[1].has_stopped(), true);
    ASSERT_EQ(stop_status.get_data()[2].has_stopped(), true);
}
