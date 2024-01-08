// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>


#include "benchmark/utils/timer_impl.hpp"


/**
 * HipTimer uses hip executor and hipEvent to measure the timing.
 */
class HipTimer : public Timer {
public:
    /**
     * Create a HipTimer.
     *
     * @param exec  Executor which should be a HipExecutor
     */
    HipTimer(std::shared_ptr<const gko::Executor> exec)
        : HipTimer(std::dynamic_pointer_cast<const gko::HipExecutor>(exec))
    {}

    /**
     * Create a HipTimer.
     *
     * @param exec  HipExecutor associated to the timer
     */
    HipTimer(std::shared_ptr<const gko::HipExecutor> exec) : Timer()
    {
        assert(exec != nullptr);
        exec_ = exec;
        auto guard = exec_->get_scoped_device_id_guard();
        GKO_ASSERT_NO_HIP_ERRORS(hipEventCreate(&start_));
        GKO_ASSERT_NO_HIP_ERRORS(hipEventCreate(&stop_));
    }

protected:
    void tic_impl() override
    {
        exec_->synchronize();
        auto guard = exec_->get_scoped_device_id_guard();
        // Currently, gko::HipExecutor always use default stream.
        GKO_ASSERT_NO_HIP_ERRORS(hipEventRecord(start_));
    }

    double toc_impl() override
    {
        auto guard = exec_->get_scoped_device_id_guard();
        // Currently, gko::HipExecutor always use default stream.
        GKO_ASSERT_NO_HIP_ERRORS(hipEventRecord(stop_));
        GKO_ASSERT_NO_HIP_ERRORS(hipEventSynchronize(stop_));
        float duration_time = 0;
        // hipEventElapsedTime gives the duration_time in milliseconds with a
        // resolution of around 0.5 microseconds
        GKO_ASSERT_NO_HIP_ERRORS(
            hipEventElapsedTime(&duration_time, start_, stop_));
        constexpr int sec_in_ms = 1e3;
        return static_cast<double>(duration_time) / sec_in_ms;
    }

private:
    std::shared_ptr<const gko::HipExecutor> exec_;
    hipEvent_t start_;
    hipEvent_t stop_;
};


std::shared_ptr<Timer> get_hip_timer(
    std::shared_ptr<const gko::HipExecutor> exec)
{
    return std::make_shared<HipTimer>(exec);
}
