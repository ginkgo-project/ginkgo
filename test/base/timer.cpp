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


#include <map>
#include <thread>


#include <gtest/gtest.h>


#include "core/test/utils/assertions.hpp"
#include "test/utils/executor.hpp"


class Timer : public CommonTestFixture {
#ifdef GKO_COMPILING_DPCPP
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

    timer->record(start);
    std::this_thread::sleep_for(std::chrono::seconds{5});
    timer->record(stop);
    timer->wait(stop);

    ASSERT_GT(timer->difference_async(start, stop), std::chrono::seconds{1});
}


TEST_F(Timer, Works)
{
    auto timer = gko::Timer::create_for_executor(this->exec);
    auto start = timer->create_time_point();
    auto stop = timer->create_time_point();

    timer->record(start);
    std::this_thread::sleep_for(std::chrono::seconds{5});
    timer->record(stop);

    ASSERT_GT(timer->difference(start, stop), std::chrono::seconds{1});
}
