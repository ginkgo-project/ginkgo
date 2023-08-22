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

#include <iostream>


#include <CL/sycl.hpp>


#include "benchmark/utils/timer_impl.hpp"


/**
 * SyclTimer uses sycl executor and event to measure the timing.
 */
class SyclTimer : public Timer {
public:
    /**
     * Create a SyclTimer.
     *
     * @param exec  Executor which should be a SyclExecutor
     */
    SyclTimer(std::shared_ptr<const gko::Executor> exec)
        : SyclTimer(std::dynamic_pointer_cast<const gko::SyclExecutor>(exec))
    {}

    /**
     * Create a SyclTimer.
     *
     * @param exec  SyclExecutor associated to the timer
     */
    SyclTimer(std::shared_ptr<const gko::SyclExecutor> exec) : Timer()
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
        // Currently, gko::SyclExecutor always use default stream.
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
    std::shared_ptr<const gko::SyclExecutor> exec_;
    sycl::event start_;
    int id_;
};


std::shared_ptr<Timer> get_sycl_timer(
    std::shared_ptr<const gko::SyclExecutor> exec)
{
    return std::make_shared<SyclTimer>(exec);
}
