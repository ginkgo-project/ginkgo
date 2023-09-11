// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/timer.hpp>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "hip/base/scoped_device_id.hip.hpp"


namespace gko {


HipTimer::HipTimer(std::shared_ptr<const HipExecutor> exec)
    : exec_{std::move(exec)}
{}


void HipTimer::init_time_point(time_point& time)
{
    detail::hip_scoped_device_id_guard guard{exec_->get_device_id()};
    time.type_ = time_point::type::hip;
    GKO_ASSERT_NO_HIP_ERRORS(hipEventCreate(&time.data_.hip_event));
}


void HipTimer::record(time_point& time)
{
    detail::hip_scoped_device_id_guard guard{exec_->get_device_id()};
    // HIP assertions are broken
    // GKO_ASSERT(time.type_ == time_point::type::hip);
    GKO_ASSERT_NO_HIP_ERRORS(
        hipEventRecord(time.data_.hip_event, exec_->get_stream()));
}


void HipTimer::wait(time_point& time)
{
    detail::hip_scoped_device_id_guard guard{exec_->get_device_id()};
    // HIP assertions are broken
    // GKO_ASSERT(time.type_ == time_point::type::hip);
    GKO_ASSERT_NO_HIP_ERRORS(hipEventSynchronize(time.data_.hip_event));
}


std::chrono::nanoseconds HipTimer::difference_async(const time_point& start,
                                                    const time_point& stop)
{
    detail::hip_scoped_device_id_guard guard{exec_->get_device_id()};
    // HIP assertions are broken
    // GKO_ASSERT(start.type_ == time_point::type::hip);
    // GKO_ASSERT(stop.type_ == time_point::type::hip);
    GKO_ASSERT_NO_HIP_ERRORS(hipEventSynchronize(stop.data_.hip_event));
    float ms{};
    GKO_ASSERT_NO_HIP_ERRORS(
        hipEventElapsedTime(&ms, start.data_.hip_event, stop.data_.hip_event));
    return std::chrono::nanoseconds{static_cast<int64>(ms * double{1e6})};
}


}  // namespace gko
