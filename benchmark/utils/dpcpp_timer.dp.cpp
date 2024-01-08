// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>


#include <CL/sycl.hpp>


#include "benchmark/utils/timer_impl.hpp"


/**
 * DpcppTimer uses dpcpp executor and event to measure the timing.
 */
class DpcppTimer : public Timer {
public:
    /**
     * Create a DpcppTimer.
     *
     * @param exec  Executor which should be a DpcppExecutor
     */
    DpcppTimer(std::shared_ptr<const gko::Executor> exec)
        : DpcppTimer(std::dynamic_pointer_cast<const gko::DpcppExecutor>(exec))
    {}

    /**
     * Create a DpcppTimer.
     *
     * @param exec  DpcppExecutor associated to the timer
     */
    DpcppTimer(std::shared_ptr<const gko::DpcppExecutor> exec) : Timer()
    {
        assert(exec != nullptr);
        if (!exec->get_queue()
                 ->template has_property<
                     sycl::property::queue::enable_profiling>()) {
            GKO_NOT_SUPPORTED(exec);
        }
        exec_ = exec;
    }

protected:
    void tic_impl() override
    {
        exec_->synchronize();
        // Currently, gko::DpcppExecutor always use default stream.
        start_ = exec_->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(1, [=](sycl::id<1> id) {});
        });
    }

    double toc_impl() override
    {
        auto stop = exec_->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(1, [=](sycl::id<1> id) {});
        });
        stop.wait_and_throw();
        // get the start time of stop
        auto stop_time = stop.get_profiling_info<
            sycl::info::event_profiling::command_start>();
        // get the end time of start
        auto start_time =
            start_
                .get_profiling_info<sycl::info::event_profiling::command_end>();
        return (stop_time - start_time) / double{1.0e9};
    }

private:
    std::shared_ptr<const gko::DpcppExecutor> exec_;
    sycl::event start_;
    int id_;
};


std::shared_ptr<Timer> get_dpcpp_timer(
    std::shared_ptr<const gko::DpcppExecutor> exec)
{
    return std::make_shared<DpcppTimer>(exec);
}
