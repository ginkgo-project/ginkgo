/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/timer.hpp>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


DpcppTimer::DpcppTimer(std::shared_ptr<const DpcppExecutor> exec)
    : exec_{std::move(exec)}
{
    if (!exec_->get_queue()
             ->template has_property<
                 sycl::property::queue::enable_profiling>()) {
        GKO_NOT_SUPPORTED(exec_);
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
    *time.data_.dpcpp_event =
        exec_->get_queue()->submit([&](sycl::handler& cgh) {
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
