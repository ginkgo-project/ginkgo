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

#include <core/stop/time.hpp>


#include <gtest/gtest.h>
#include <chrono>


namespace {


constexpr double test_seconds = 0.5;
constexpr double eps = 1.0e-4;
using double_seconds = std::chrono::duration<double>;


class Time : public ::testing::Test {
protected:
    Time()
        : factory_{gko::stop::Time::Factory::create(test_seconds)},
          exec_{gko::ReferenceExecutor::create()}
    {}

    std::unique_ptr<gko::stop::Time::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};


TEST_F(Time, CanCreateFactory)
{
    ASSERT_NE(factory_, nullptr);
    ASSERT_EQ(std::chrono::duration_cast<double_seconds>(factory_->v_),
              double_seconds(test_seconds));
}


TEST_F(Time, CanCreateCriterion)
{
    auto criterion = factory_->create_criterion(nullptr, nullptr, nullptr);
    ASSERT_NE(criterion, nullptr);
}


TEST_F(Time, WaitsTillTime)
{
    auto criterion = factory_->create_criterion(nullptr, nullptr, nullptr);
    unsigned int iters = 0;
    gko::Array<bool> converged(exec_, 1);
    auto start = std::chrono::system_clock::now();

    while (1) {
        if (criterion->update().num_iterations(iters).check(converged)) break;
        iters++;
    }
    auto time = std::chrono::system_clock::now() - start;
    double time_d = std::chrono::duration_cast<double_seconds>(time).count();

    /** Somehow this can be very imprecise, maybe due to the duration cast
     * therefore I add an epsilon (here of 0.1ms)
     */
    ASSERT_GE(time_d + eps, test_seconds);
}


}  // namespace
