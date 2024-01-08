// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/combined.hpp>


#include <chrono>
#include <thread>
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#endif  // defined(_WIN32) || defined(__CYGWIN__)


#include <gtest/gtest.h>


#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/time.hpp>


namespace {


constexpr gko::size_type test_iterations = 10;
constexpr int test_seconds = 999;  // we will never converge through seconds
constexpr double eps = 1.0e-4;
using double_seconds = std::chrono::duration<double>;


inline void sleep_millisecond(unsigned int ms)
{
#if defined(_WIN32) || defined(__CYGWIN__)
    Sleep(ms);
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#endif
}


class Combined : public ::testing::Test {
protected:
    Combined()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ =
            gko::stop::Combined::build()
                .with_criteria(
                    gko::stop::Iteration::build()
                        .with_max_iters(test_iterations)
                        .on(exec_),
                    gko::stop::Time::build()
                        .with_time_limit(std::chrono::seconds(test_seconds))
                        .on(exec_))
                .on(exec_);
    }

    std::unique_ptr<gko::stop::Combined::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};


/** The purpose of this test is to check that the iteration process stops due to
 * the correct stopping criterion: the iteration criterion and not time due to
 * the huge time picked. */
TEST_F(Combined, WaitsTillIteration)
{
    bool one_changed{};
    gko::array<gko::stopping_status> stop_status(exec_, 1);
    stop_status.get_data()[0].reset();
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    gko::array<bool> converged(exec_, 1);

    ASSERT_FALSE(
        criterion->update()
            .num_iterations(test_iterations - 1)
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_TRUE(
        criterion->update()
            .num_iterations(test_iterations)
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_TRUE(
        criterion->update()
            .num_iterations(test_iterations + 1)
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(static_cast<int>(stop_status.get_data()[0].get_id()), 1);
}


/** The purpose of this test is to check that the iteration process stops due to
 * the correct stopping criterion: the time criterion and not iteration due to
 * the very small time picked and huge iteration count. */
TEST_F(Combined, WaitsTillTime)
{
    constexpr int testiters = 10;
    constexpr int timelimit_ms = 10;
    factory_ =
        gko::stop::Combined::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(9999u).on(exec_),
                gko::stop::Time::build()
                    .with_time_limit(std::chrono::milliseconds(timelimit_ms))
                    .on(exec_))
            .on(exec_);
    unsigned int iters = 0;
    bool one_changed{};
    gko::array<gko::stopping_status> stop_status(exec_, 1);
    stop_status.get_data()[0].reset();
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < testiters; i++) {
        sleep_millisecond(timelimit_ms / testiters);
        if (criterion->update().num_iterations(i).check(
                RelativeStoppingId, true, &stop_status, &one_changed))
            break;
    }
    auto time = std::chrono::steady_clock::now() - start;
    double time_d = std::chrono::duration_cast<double_seconds>(time).count();

    ASSERT_GE(time_d, timelimit_ms * 1e-3);
    ASSERT_EQ(static_cast<int>(stop_status.get_data()[0].get_id()), 2);
}


}  // namespace
