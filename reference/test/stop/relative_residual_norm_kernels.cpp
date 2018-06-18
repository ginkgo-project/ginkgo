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

#include <core/stop/relative_residual_norm.hpp>


#include <gtest/gtest.h>


namespace {


constexpr double residual_goal = 1.0e-14;


class RelativeResidualNorm : public ::testing::Test {
protected:
    RelativeResidualNorm()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ = gko::stop::RelativeResidualNorm<>::Factory::create()
                       .with_rel_residual_goal(residual_goal)
                       .on_executor(exec_);
    }

    std::unique_ptr<gko::stop::RelativeResidualNorm<>::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};


TEST_F(RelativeResidualNorm, WaitsTillResidualGoal)
{
    auto criterion = factory_->generate(nullptr);
    bool one_changed{};
    gko::Array<gko::stopping_status> stop_status(exec_, 1);
    stop_status.get_data()[0].clear();
    constexpr gko::uint8 RelativeStoppingId{1};
    auto scalar = gko::initialize<gko::matrix::Dense<>>({1.0}, exec_);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(scalar.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    scalar->at(0) = residual_goal * 1.0e+2;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(scalar.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    scalar->at(0) = residual_goal * 1.0e-2;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(scalar.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TEST_F(RelativeResidualNorm, WaitsTillResidualGoalMultipleRHS)
{
    auto criterion = factory_->generate(nullptr);
    bool one_changed{};
    gko::Array<gko::stopping_status> stop_status(exec_, 2);
    // Array only does malloc, it *does not* construct the object
    // therefore you get undefined values in your objects whatever you do.
    // Proper fix is not easy, we can't just call memset. We can probably not
    // call the placement constructor either
    stop_status.get_data()[0].clear();
    stop_status.get_data()[1].clear();
    constexpr gko::uint8 RelativeStoppingId{1};
    auto mtx = gko::initialize<gko::matrix::Dense<>>({{1.0, 1.0}}, exec_);

    ASSERT_FALSE(criterion->update().residual_norm(mtx.get()).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    mtx->at(0, 0) = residual_goal * 1.0e-2;
    ASSERT_FALSE(criterion->update().residual_norm(mtx.get()).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
    one_changed = false;

    mtx->at(0, 1) = residual_goal * 1.0e-2;
    ASSERT_TRUE(criterion->update().residual_norm(mtx.get()).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


}  // namespace
