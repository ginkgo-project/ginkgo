// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/time.hpp>


#include <chrono>
#include <thread>
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#endif  // defined(_WIN32) || defined(__CYGWIN__)


#include <gtest/gtest.h>


namespace {


constexpr long test_ms = 500;
constexpr double eps = 1.0e-4;
using double_seconds = std::chrono::duration<double, std::milli>;


inline void sleep_millisecond(unsigned int ms)
{
#if defined(_WIN32) || defined(__CYGWIN__)
    Sleep(ms);
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#endif
}


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


TEST_F(Time, WaitsTillTime)
{
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    bool one_changed{};
    gko::array<gko::stopping_status> stop_status(exec_, 1);
    stop_status.get_data()[0].reset();
    constexpr gko::uint8 RelativeStoppingId{1};

    sleep_millisecond(test_ms);

    ASSERT_TRUE(criterion->update().check(RelativeStoppingId, true,
                                          &stop_status, &one_changed));
}


}  // namespace
