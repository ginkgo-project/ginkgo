// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/timer.hpp>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


DpcppTimer::DpcppTimer(std::shared_ptr<const DpcppExecutor> exec)
    : queue_{exec->get_queue()}
{
    if (!queue_->template has_property<
            sycl::property::queue::enable_profiling>()) {
        GKO_NOT_SUPPORTED(exec);
    }
}


void DpcppTimer::init_time_point(time_point& time)
{
    time.type_ = time_point::type::dpcpp;
    time.data_.dpcpp_event = new sycl::event{};
}


void DpcppTimer::record(time_point& time)
{
    GKO_ASSERT(time.type_ == time_point::type::dpcpp);
    *time.data_.dpcpp_event = queue_->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(1, [=](sycl::id<1> id) {});
    });
}


void DpcppTimer::wait(time_point& time)
{
    GKO_ASSERT(time.type_ == time_point::type::dpcpp);
    time.data_.dpcpp_event->wait_and_throw();
}


std::chrono::nanoseconds DpcppTimer::difference_async(const time_point& start,
                                                      const time_point& stop)
{
    GKO_ASSERT(start.type_ == time_point::type::dpcpp);
    GKO_ASSERT(stop.type_ == time_point::type::dpcpp);
    stop.data_.dpcpp_event->wait_and_throw();
    auto stop_time =
        stop.data_.dpcpp_event
            ->get_profiling_info<sycl::info::event_profiling::command_start>();
    auto start_time =
        start.data_.dpcpp_event
            ->get_profiling_info<sycl::info::event_profiling::command_end>();
    return std::chrono::nanoseconds{static_cast<int64>(stop_time - start_time)};
}


}  // namespace gko
