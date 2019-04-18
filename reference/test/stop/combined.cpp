/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/stop/combined.hpp>


#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/time.hpp>


#include <gtest/gtest.h>
#include <thread>


namespace {


constexpr gko::size_type test_iterations = 10;
constexpr int test_seconds = 999;  // we will never converge through seconds
constexpr double eps = 1.0e-4;
using double_seconds = std::chrono::duration<double>;


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
    gko::Array<gko::stopping_status> stop_status(exec_, 1);
    stop_status.get_data()[0].reset();
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    gko::Array<bool> converged(exec_, 1);

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
    constexpr double timelimit = 1.0e-9;
    constexpr int testiters = 10;
    factory_ =
        gko::stop::Combined::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(9999u).on(exec_),
                gko::stop::Time::build()
                    .with_time_limit(std::chrono::nanoseconds(1))
                    .on(exec_))
            .on(exec_);
    unsigned int iters = 0;
    bool one_changed{};
    gko::Array<gko::stopping_status> stop_status(exec_, 1);
    stop_status.get_data()[0].reset();
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < testiters; i++) {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(timelimit / testiters));
        if (criterion->update().num_iterations(i).check(
                RelativeStoppingId, true, &stop_status, &one_changed))
            break;
    }
    auto time = std::chrono::steady_clock::now() - start;
    double time_d = std::chrono::duration_cast<double_seconds>(time).count();


    ASSERT_GE(time_d + eps, 1.0e-9);
    ASSERT_EQ(static_cast<int>(stop_status.get_data()[0].get_id()), 2);
}


}  // namespace
