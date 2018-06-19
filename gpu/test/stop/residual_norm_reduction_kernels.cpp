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

#include <core/stop/residual_norm_reduction.hpp>


#include <gtest/gtest.h>


namespace {


constexpr double reduction_factor = 1.0e-14;


class ResidualNormReduction : public ::testing::Test {
protected:
    ResidualNormReduction()
    {
        ref_ = gko::ReferenceExecutor::create();
        gpu_ = gko::GpuExecutor::create(0, ref_);
        factory_ = gko::stop::ResidualNormReduction<>::Factory::create()
                       .with_reduction_factor(reduction_factor)
                       .on_executor(gpu_);
    }

    std::unique_ptr<gko::stop::ResidualNormReduction<>::Factory> factory_;
    std::shared_ptr<const gko::GpuExecutor> gpu_;
    std::shared_ptr<gko::ReferenceExecutor> ref_;
};


TEST_F(ResidualNormReduction, WaitsTillResidualGoal)
{
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    bool one_changed{};
    gko::Array<gko::stopping_status> stop_status(ref_, 1);
    stop_status.get_data()[0].clear();
    constexpr gko::uint8 RelativeStoppingId{1};
    auto scalar = gko::initialize<gko::matrix::Dense<>>({1.0}, ref_);
    auto d_scalar = gko::matrix::Dense<>::create(gpu_);
    d_scalar->copy_from(scalar.get());
    stop_status.set_executor(gpu_);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_scalar.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    scalar->at(0) = reduction_factor * 1.0e+2;
    d_scalar->copy_from(scalar.get());
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_scalar.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    stop_status.set_executor(gpu_);
    ASSERT_EQ(one_changed, false);

    scalar->at(0) = reduction_factor * 1.0e-2;
    d_scalar->copy_from(scalar.get());
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(d_scalar.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TEST_F(ResidualNormReduction, WaitsTillResidualGoalMultipleRHS)
{
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    bool one_changed{};
    gko::Array<gko::stopping_status> stop_status(ref_, 2);
    stop_status.get_data()[0].clear();
    stop_status.get_data()[1].clear();
    constexpr gko::uint8 RelativeStoppingId{1};
    auto mtx = gko::initialize<gko::matrix::Dense<>>({{1.0, 1.0}}, ref_);
    auto d_mtx = gko::matrix::Dense<>::create(gpu_);
    d_mtx->copy_from(mtx.get());
    stop_status.set_executor(gpu_);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_mtx.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    mtx->at(0, 0) = reduction_factor * 1.0e-2;
    d_mtx->copy_from(mtx.get());
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_mtx.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    stop_status.set_executor(gpu_);
    ASSERT_EQ(one_changed, true);

    mtx->at(0, 1) = reduction_factor * 1.0e-2;
    d_mtx->copy_from(mtx.get());
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(d_mtx.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


}  // namespace
