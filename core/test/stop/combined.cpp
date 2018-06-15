/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/stop/combined.hpp>


#include <core/stop/iteration.hpp>
#include <core/stop/time.hpp>


#include <gtest/gtest.h>


namespace {


constexpr unsigned int test_iterations = 10;
constexpr double test_seconds = 999;  // we will never converge through seconds
constexpr double eps = 1.0e-4;
using double_seconds = std::chrono::duration<double>;


class Combined : public ::testing::Test {
protected:
    Combined()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ = gko::stop::Combined::Factory::create(
            exec_,
            gko::stop::Iteration::Factory::create(exec_, test_iterations),
            gko::stop::Time::Factory::create(exec_, test_seconds));
    }

    std::unique_ptr<gko::stop::Combined::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};


TEST_F(Combined, CanCreateFactory)
{
    ASSERT_NE(factory_, nullptr);
    ASSERT_EQ(factory_->v_.size(), 2);
}


TEST_F(Combined, CanCreateCriterion)
{
    auto criterion = factory_->create_criterion(nullptr, nullptr, nullptr);
    ASSERT_NE(criterion, nullptr);
}


/* We know iteration will converge and not time due to huge time picked */
TEST_F(Combined, WaitsTillIteration)
{
    bool one_changed{};
    gko::Array<gko::stopping_status> stop_status(exec_, 1);
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory_->create_criterion(nullptr, nullptr, nullptr);
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
}


/* Here we know time will converge */
TEST_F(Combined, WaitsTillTime)
{
    factory_ = gko::stop::Combined::Factory::create(
        exec_, gko::stop::Iteration::Factory::create(exec_, 9999),
        gko::stop::Time::Factory::create(exec_, 1.0e-9));
    unsigned int iters = 0;
    bool one_changed{};
    gko::Array<gko::stopping_status> stop_status(exec_, 1);
    constexpr gko::uint8 RelativeStoppingId{1};
    auto criterion = factory_->create_criterion(nullptr, nullptr, nullptr);
    auto start = std::chrono::system_clock::now();

    while (1) {
        if (criterion->update().num_iterations(iters).check(
                RelativeStoppingId, true, &stop_status, &one_changed))
            break;
        iters++;
    }
    auto time = std::chrono::system_clock::now() - start;
    double time_d = std::chrono::duration_cast<double_seconds>(time).count();


    ASSERT_GE(time_d + eps, 1.0e-9);
}


}  // namespace
