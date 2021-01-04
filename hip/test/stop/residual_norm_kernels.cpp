/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/stop/residual_norm.hpp>


#include <gtest/gtest.h>


#include "hip/test/utils.hip.hpp"


namespace {


constexpr double tol = 1.0e-14;


class ResidualNormReduction : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;

    ResidualNormReduction()
    {
        ref_ = gko::ReferenceExecutor::create();
        hip_ = gko::HipExecutor::create(0, ref_);
        factory_ = gko::stop::ResidualNormReduction<>::build()
                       .with_reduction_factor(tol)
                       .on(hip_);
    }

    std::unique_ptr<gko::stop::ResidualNormReduction<>::Factory> factory_;
    std::shared_ptr<const gko::HipExecutor> hip_;
    std::shared_ptr<gko::ReferenceExecutor> ref_;
};


TEST_F(ResidualNormReduction, WaitsTillResidualGoal)
{
    auto res = gko::initialize<Mtx>({100.0}, ref_);
    auto d_res = Mtx::create(hip_);
    d_res->copy_from(res.get());
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, ref_);
    std::shared_ptr<gko::LinOp> d_rhs = Mtx::create(hip_);
    d_rhs->copy_from(rhs.get());
    auto criterion = factory_->generate(nullptr, d_rhs, nullptr, d_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(ref_, 1);
    stop_status.get_data()[0].reset();
    stop_status.set_executor(hip_);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res->at(0) = tol * 1.1e+2;
    d_res->copy_from(res.get());
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    stop_status.set_executor(hip_);
    ASSERT_FALSE(one_changed);

    res->at(0) = tol * 0.9e+2;
    d_res->copy_from(res.get());
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
}


TEST_F(ResidualNormReduction, WaitsTillResidualGoalMultipleRHS)
{
    auto res = gko::initialize<Mtx>({{100.0, 100.0}}, ref_);
    auto d_res = Mtx::create(hip_);
    d_res->copy_from(res.get());
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({{10.0, 10.0}}, ref_);
    std::shared_ptr<gko::LinOp> d_rhs = Mtx::create(hip_);
    d_rhs->copy_from(rhs.get());
    auto criterion = factory_->generate(nullptr, d_rhs, nullptr, d_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(ref_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.set_executor(hip_);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res->at(0, 0) = tol * 0.9e+2;
    d_res->copy_from(res.get());
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    stop_status.set_executor(hip_);
    ASSERT_TRUE(one_changed);

    res->at(0, 1) = tol * 0.9e+2;
    d_res->copy_from(res.get());
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_TRUE(stop_status.get_data()[1].has_converged());
    ASSERT_TRUE(one_changed);
}


class RelativeResidualNorm : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;

    RelativeResidualNorm()
    {
        ref_ = gko::ReferenceExecutor::create();
        hip_ = gko::HipExecutor::create(0, ref_);
        factory_ =
            gko::stop::RelativeResidualNorm<>::build().with_tolerance(tol).on(
                hip_);
    }

    std::unique_ptr<gko::stop::RelativeResidualNorm<>::Factory> factory_;
    std::shared_ptr<const gko::HipExecutor> hip_;
    std::shared_ptr<gko::ReferenceExecutor> ref_;
};


TEST_F(RelativeResidualNorm, WaitsTillResidualGoal)
{
    auto res = gko::initialize<Mtx>({100.0}, ref_);
    auto d_res = Mtx::create(hip_);
    d_res->copy_from(res.get());
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, ref_);
    std::shared_ptr<gko::LinOp> d_rhs = Mtx::create(hip_);
    d_rhs->copy_from(rhs.get());
    auto criterion = factory_->generate(nullptr, d_rhs, nullptr, d_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(ref_, 1);
    stop_status.get_data()[0].reset();
    stop_status.set_executor(hip_);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res->at(0) = tol * 1.1e+1;
    d_res->copy_from(res.get());
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    stop_status.set_executor(hip_);
    ASSERT_FALSE(one_changed);

    res->at(0) = tol * 0.9e+1;
    d_res->copy_from(res.get());
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
}


TEST_F(RelativeResidualNorm, WaitsTillResidualGoalMultipleRHS)
{
    auto res = gko::initialize<Mtx>({{100.0, 100.0}}, ref_);
    auto d_res = Mtx::create(hip_);
    d_res->copy_from(res.get());
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({{10.0, 10.0}}, ref_);
    std::shared_ptr<gko::LinOp> d_rhs = Mtx::create(hip_);
    d_rhs->copy_from(rhs.get());
    auto criterion = factory_->generate(nullptr, d_rhs, nullptr, d_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(ref_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.set_executor(hip_);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res->at(0, 0) = tol * 0.9e+1;
    d_res->copy_from(res.get());
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    stop_status.set_executor(hip_);
    ASSERT_TRUE(one_changed);

    res->at(0, 1) = tol * 0.9e+1;
    d_res->copy_from(res.get());
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_TRUE(stop_status.get_data()[1].has_converged());
    ASSERT_TRUE(one_changed);
}


class AbsoluteResidualNorm : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;

    AbsoluteResidualNorm()
    {
        ref_ = gko::ReferenceExecutor::create();
        hip_ = gko::HipExecutor::create(0, ref_);
        factory_ =
            gko::stop::AbsoluteResidualNorm<>::build().with_tolerance(tol).on(
                hip_);
    }

    std::unique_ptr<gko::stop::AbsoluteResidualNorm<>::Factory> factory_;
    std::shared_ptr<const gko::HipExecutor> hip_;
    std::shared_ptr<gko::ReferenceExecutor> ref_;
};


TEST_F(AbsoluteResidualNorm, WaitsTillResidualGoal)
{
    auto res = gko::initialize<Mtx>({100.0}, ref_);
    auto d_res = Mtx::create(hip_);
    d_res->copy_from(res.get());
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, ref_);
    std::shared_ptr<gko::LinOp> d_rhs = Mtx::create(hip_);
    d_rhs->copy_from(rhs.get());
    auto criterion = factory_->generate(nullptr, d_rhs, nullptr, d_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(ref_, 1);
    stop_status.get_data()[0].reset();
    stop_status.set_executor(hip_);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res->at(0) = tol * 1.1;
    d_res->copy_from(res.get());
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    stop_status.set_executor(hip_);
    ASSERT_FALSE(one_changed);

    res->at(0) = tol * 0.9;
    d_res->copy_from(res.get());
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
}


TEST_F(AbsoluteResidualNorm, WaitsTillResidualGoalMultipleRHS)
{
    auto res = gko::initialize<Mtx>({{100.0, 100.0}}, ref_);
    auto d_res = Mtx::create(hip_);
    d_res->copy_from(res.get());
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({{10.0, 10.0}}, ref_);
    std::shared_ptr<gko::LinOp> d_rhs = Mtx::create(hip_);
    d_rhs->copy_from(rhs.get());
    auto criterion = factory_->generate(nullptr, d_rhs, nullptr, d_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(ref_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.set_executor(hip_);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res->at(0, 0) = tol * 0.9;
    d_res->copy_from(res.get());
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    stop_status.set_executor(hip_);
    ASSERT_TRUE(one_changed);

    res->at(0, 1) = tol * 0.9;
    d_res->copy_from(res.get());
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(d_res.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(ref_);
    ASSERT_TRUE(stop_status.get_data()[1].has_converged());
    ASSERT_TRUE(one_changed);
}


}  // namespace
