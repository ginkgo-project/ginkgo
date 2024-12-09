// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <map>
#include <thread>

#include <gtest/gtest.h>

#include <ginkgo/core/base/timer.hpp>

#include "core/test/utils/assertions.hpp"
#include "test/utils/common_fixture.hpp"


class Timer : public CommonTestFixture {
#ifdef GKO_COMPILING_DPCPP
public:
    Timer()
    {
        // require profiling capability
        const auto property = dpcpp_queue_property::in_order |
                              dpcpp_queue_property::enable_profiling;
        if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
            exec = gko::DpcppExecutor::create(0, ref, "gpu", property);
        } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
            exec = gko::DpcppExecutor::create(0, ref, "cpu", property);
        } else {
            throw std::runtime_error{"No suitable DPC++ devices"};
        }
    }
#endif
};


TEST_F(Timer, WorksAsync)
{
    auto timer = gko::Timer::create_for_executor(this->exec);
    auto start = timer->create_time_point();
    auto stop = timer->create_time_point();
    gko::array<int> dummy{this->exec, {0}};
    auto dummy2 = dummy;
    this->exec->synchronize();
    // we do some minimal work to work around Intel GPU timers running backwards

    timer->record(start);
    dummy = dummy2;
    std::this_thread::sleep_for(std::chrono::seconds{5});
    dummy = dummy2;
    timer->record(stop);
    timer->wait(stop);

    ASSERT_GT(timer->difference_async(start, stop), std::chrono::seconds{1});
}


TEST_F(Timer, Works)
{
    auto timer = gko::Timer::create_for_executor(this->exec);
    auto start = timer->create_time_point();
    auto stop = timer->create_time_point();
    gko::array<int> dummy{this->exec, {0}};
    auto dummy2 = dummy;
    this->exec->synchronize();
    // we do some minimal work to work around Intel GPU timers running backwards

    timer->record(start);
    dummy = dummy2;
    std::this_thread::sleep_for(std::chrono::seconds{5});
    dummy = dummy2;
    timer->record(stop);

    ASSERT_GT(timer->difference(start, stop), std::chrono::seconds{1});
}


TEST_F(Timer, DoesntOwnExecutor)
{
    const auto old_use_count = this->exec.use_count();

    auto timer = gko::Timer::create_for_executor(this->exec);

    ASSERT_EQ(this->exec.use_count(), old_use_count);
}
