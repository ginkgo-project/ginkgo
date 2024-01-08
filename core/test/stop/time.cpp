// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/time.hpp>


#include <chrono>
#include <thread>


#include <gtest/gtest.h>


namespace {


constexpr long test_ms = 500;
constexpr double eps = 1.0e-4;
using double_seconds = std::chrono::duration<double, std::milli>;


class Time : public ::testing::Test {
protected:
    Time() : exec_{gko::ReferenceExecutor::create()}
    {
        factory_ = gko::stop::Time::build()
                       .with_time_limit(std::chrono::milliseconds(test_ms))
                       .on(exec_);
    }

    std::unique_ptr<gko::stop::Time::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};


TEST_F(Time, CanCreateFactory)
{
    ASSERT_NE(factory_, nullptr);
    ASSERT_EQ(factory_->get_parameters().time_limit,
              std::chrono::milliseconds(test_ms));
}


TEST_F(Time, CanCreateCriterion)
{
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    ASSERT_NE(criterion, nullptr);
}


}  // namespace
