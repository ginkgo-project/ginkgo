/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <type_traits>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class ResidualNorm : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ResidualNorm()
    {
        exec_ = gko::ReferenceExecutor::create();
        rhs_factory_ = gko::stop::ResidualNorm<T>::build()
                           .with_reduction_factor(r<T>::value)
                           .on(exec_);
        rel_factory_ = gko::stop::ResidualNorm<T>::build()
                           .with_reduction_factor(r<T>::value)
                           .with_baseline(gko::stop::mode::initial_resnorm)
                           .on(exec_);
        abs_factory_ = gko::stop::ResidualNorm<T>::build()
                           .with_reduction_factor(r<T>::value)
                           .with_baseline(gko::stop::mode::absolute)
                           .on(exec_);
    }

    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> rhs_factory_;
    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> rel_factory_;
    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> abs_factory_;
    std::shared_ptr<const gko::Executor> exec_;
};

TYPED_TEST_SUITE(ResidualNorm, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(ResidualNorm, CanCreateFactory)
{
    ASSERT_NE(this->rhs_factory_, nullptr);
    ASSERT_EQ(this->rhs_factory_->get_parameters().reduction_factor,
              r<TypeParam>::value);
    ASSERT_EQ(this->rhs_factory_->get_parameters().baseline,
              gko::stop::mode::rhs_norm);
    ASSERT_EQ(this->rhs_factory_->get_executor(), this->exec_);
    ASSERT_NE(this->rel_factory_, nullptr);
    ASSERT_EQ(this->rel_factory_->get_parameters().reduction_factor,
              r<TypeParam>::value);
    ASSERT_EQ(this->rel_factory_->get_parameters().baseline,
              gko::stop::mode::initial_resnorm);
    ASSERT_EQ(this->rel_factory_->get_executor(), this->exec_);
    ASSERT_NE(this->abs_factory_, nullptr);
    ASSERT_EQ(this->abs_factory_->get_parameters().reduction_factor,
              r<TypeParam>::value);
    ASSERT_EQ(this->abs_factory_->get_parameters().baseline,
              gko::stop::mode::absolute);
    ASSERT_EQ(this->abs_factory_->get_executor(), this->exec_);
}


TYPED_TEST(ResidualNorm, CannotCreateCriterionWithoutNeededInput)
{
    ASSERT_THROW(
        this->rhs_factory_->generate(nullptr, nullptr, nullptr, nullptr),
        gko::NotSupported);
    ASSERT_THROW(
        this->rel_factory_->generate(nullptr, nullptr, nullptr, nullptr),
        gko::NotSupported);
    ASSERT_THROW(
        this->abs_factory_->generate(nullptr, nullptr, nullptr, nullptr),
        gko::NotSupported);
}


TYPED_TEST(ResidualNorm, CanCreateCriterionWithNeededInput)
{
    using Mtx = typename TestFixture::Mtx;
    std::shared_ptr<gko::LinOp> scalar =
        gko::initialize<Mtx>({1.0}, this->exec_);
    auto rhs_criterion =
        this->rhs_factory_->generate(nullptr, scalar, nullptr, nullptr);
    auto rel_criterion =
        this->rel_factory_->generate(nullptr, nullptr, nullptr, scalar.get());
    auto abs_criterion =
        this->abs_factory_->generate(nullptr, scalar, nullptr, nullptr);

    ASSERT_NE(rhs_criterion, nullptr);
    ASSERT_NE(rel_criterion, nullptr);
    ASSERT_NE(abs_criterion, nullptr);
}


TYPED_TEST(ResidualNorm, WaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T_nc = gko::remove_complex<TypeParam>;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec_);
    auto rhs_criterion =
        this->rhs_factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    auto rel_criterion =
        this->rel_factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    auto abs_criterion =
        this->abs_factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    {
        auto res_norm = gko::initialize<NormVector>({10.0}, this->exec_);
        auto rhs_norm = gko::initialize<NormVector>({100.0}, this->exec_);
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_FALSE(
            rhs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0) = r<TypeParam>::value * 1.1 * rhs_norm->at(0);
        ASSERT_FALSE(
            rhs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        res_norm->at(0) = r<TypeParam>::value * 0.9 * rhs_norm->at(0);
        ASSERT_TRUE(
            rhs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
    {
        auto res_norm = gko::initialize<NormVector>({100.0}, this->exec_);
        auto init_res_val = res_norm->at(0, 0);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_FALSE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0) = r<TypeParam>::value * 1.1 * init_res_val;
        ASSERT_FALSE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        res_norm->at(0) = r<TypeParam>::value * 0.9 * init_res_val;
        ASSERT_TRUE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
    {
        auto res_norm = gko::initialize<NormVector>({100.0}, this->exec_);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_FALSE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0) = r<TypeParam>::value * 1.1;
        ASSERT_FALSE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        res_norm->at(0) = r<TypeParam>::value * 0.9;
        ASSERT_TRUE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
}


TYPED_TEST(ResidualNorm, SelfCalulatesThrowWithoutMatrix)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec_);

    T rhs_val = 10.0;
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({rhs_val}, this->exec_);
    auto rhs_criterion =
        this->rhs_factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    auto rel_criterion =
        this->rel_factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    auto abs_criterion =
        this->abs_factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    {
        auto solution = gko::initialize<Mtx>({rhs_val - T{10.0}}, this->exec_);
        auto rhs_norm = gko::initialize<NormVector>({100.0}, this->exec_);
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_THROW(
            rhs_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed),
            gko::NotSupported);
    }
    {
        T initial_norm = 100.0;
        auto solution =
            gko::initialize<Mtx>({rhs_val - initial_norm}, this->exec_);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_THROW(
            rel_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed),
            gko::NotSupported);
    }
    {
        auto solution = gko::initialize<Mtx>({rhs_val - T{100.0}}, this->exec_);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_THROW(
            abs_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed),
            gko::NotSupported);
    }
}


TYPED_TEST(ResidualNorm, RelativeSelfCalulatesThrowWithoutRhs)
{
    // only relative residual norm allows generation without rhs.
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec_);

    T rhs_val = 10.0;
    auto rel_criterion = this->rel_factory_->generate(nullptr, nullptr, nullptr,
                                                      initial_res.get());
    T initial_norm = 100.0;
    auto solution = gko::initialize<Mtx>({rhs_val - initial_norm}, this->exec_);
    constexpr gko::uint8 RelativeStoppingId{1};
    bool one_changed{};
    gko::array<gko::stopping_status> stop_status(this->exec_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_THROW(
        rel_criterion->update()
            .solution(solution.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed),
        gko::NotSupported);
}


TYPED_TEST(ResidualNorm, SelfCalulatesAndWaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec_);
    auto system_mtx = share(gko::initialize<Mtx>({1.0}, this->exec_));

    T rhs_val = 10.0;
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({rhs_val}, this->exec_);
    auto rhs_criterion = this->rhs_factory_->generate(system_mtx, rhs, nullptr,
                                                      initial_res.get());
    auto rel_criterion = this->rel_factory_->generate(system_mtx, rhs, nullptr,
                                                      initial_res.get());
    auto abs_criterion = this->abs_factory_->generate(system_mtx, rhs, nullptr,
                                                      initial_res.get());
    {
        auto solution = gko::initialize<Mtx>({rhs_val - T{10.0}}, this->exec_);
        auto rhs_norm = gko::initialize<NormVector>({100.0}, this->exec_);
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_FALSE(
            rhs_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        solution->at(0) = rhs_val - r<T>::value * T{1.1} * rhs_norm->at(0);
        ASSERT_FALSE(
            rhs_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        solution->at(0) = rhs_val - r<T>::value * T{0.9} * rhs_norm->at(0);
        ASSERT_TRUE(
            rhs_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
    {
        T initial_norm = 100.0;
        auto solution =
            gko::initialize<Mtx>({rhs_val - initial_norm}, this->exec_);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_FALSE(
            rel_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        solution->at(0) = rhs_val - r<T>::value * T{1.1} * initial_norm;
        ASSERT_FALSE(
            rel_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        solution->at(0) = rhs_val - r<T>::value * T{0.9} * initial_norm;
        ASSERT_TRUE(
            rel_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
    {
        auto solution = gko::initialize<Mtx>({rhs_val - T{100.0}}, this->exec_);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_FALSE(
            abs_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        solution->at(0) = rhs_val - r<T>::value * T{1.2};
        ASSERT_FALSE(
            abs_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        solution->at(0) = rhs_val - r<T>::value * T{0.9};
        ASSERT_TRUE(
            abs_criterion->update()
                .solution(solution.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
}


TYPED_TEST(ResidualNorm, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto res = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({I<T>{10.0, 10.0}}, this->exec_);
    auto rhs_criterion =
        this->rhs_factory_->generate(nullptr, rhs, nullptr, res.get());
    auto rel_criterion =
        this->rel_factory_->generate(nullptr, rhs, nullptr, res.get());
    auto abs_criterion =
        this->abs_factory_->generate(nullptr, rhs, nullptr, res.get());
    {
        auto res_norm =
            gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec_);
        auto rhs_norm =
            gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec_);
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
        bool one_changed{};
        constexpr gko::uint8 RelativeStoppingId{1};
        gko::array<gko::stopping_status> stop_status(this->exec_, 2);
        stop_status.get_data()[0].reset();
        stop_status.get_data()[1].reset();

        ASSERT_FALSE(
            rhs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0, 0) = r<TypeParam>::value * 0.9 * rhs_norm->at(0, 0);
        ASSERT_FALSE(
            rhs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);

        res_norm->at(0, 1) = r<TypeParam>::value * 0.9 * rhs_norm->at(0, 1);
        ASSERT_TRUE(
            rhs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
    {
        auto res_norm =
            gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec_);
        bool one_changed{};
        constexpr gko::uint8 RelativeStoppingId{1};
        gko::array<gko::stopping_status> stop_status(this->exec_, 2);
        stop_status.get_data()[0].reset();
        stop_status.get_data()[1].reset();

        ASSERT_FALSE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0, 0) = r<TypeParam>::value * 0.9 * res_norm->at(0, 0);
        ASSERT_FALSE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);

        res_norm->at(0, 1) = r<TypeParam>::value * 0.9 * res_norm->at(0, 1);
        ASSERT_TRUE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
    {
        auto res_norm =
            gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec_);
        bool one_changed{};
        constexpr gko::uint8 RelativeStoppingId{1};
        gko::array<gko::stopping_status> stop_status(this->exec_, 2);
        stop_status.get_data()[0].reset();
        stop_status.get_data()[1].reset();

        ASSERT_FALSE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0, 0) = r<TypeParam>::value * 0.9;
        ASSERT_FALSE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);

        res_norm->at(0, 1) = r<TypeParam>::value * 0.9;
        ASSERT_TRUE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
}


template <typename T>
class ResidualNormReduction : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ResidualNormReduction()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ = gko::stop::ResidualNormReduction<T>::build()
                       .with_reduction_factor(r<T>::value)
                       .on(exec_);
    }

    std::unique_ptr<typename gko::stop::ResidualNormReduction<T>::Factory>
        factory_;
    std::shared_ptr<const gko::ReferenceExecutor> exec_;
};

TYPED_TEST_SUITE(ResidualNormReduction, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(ResidualNormReduction,
           CanCreateCriterionWithMtxRhsXWithoutInitialRes)
{
    using Mtx = typename TestFixture::Mtx;
    std::shared_ptr<gko::LinOp> x = gko::initialize<Mtx>({100.0}, this->exec_);
    std::shared_ptr<gko::LinOp> mtx = gko::initialize<Mtx>({1.0}, this->exec_);
    std::shared_ptr<gko::LinOp> b = gko::initialize<Mtx>({10.0}, this->exec_);

    auto criterion = this->factory_->generate(mtx, b, x.get());

    ASSERT_NE(criterion, nullptr);
}


TYPED_TEST(ResidualNormReduction, WaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec_);
    auto res_norm = gko::initialize<NormVector>({100.0}, this->exec_);
    auto init_res_val = res_norm->at(0, 0);
    auto criterion =
        this->factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0) = r<TypeParam>::value * 1.1 * init_res_val;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    res_norm->at(0) = r<TypeParam>::value * 0.9 * init_res_val;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(ResidualNormReduction, WaitsTillResidualGoalWithoutInitialRes)
{
    using T = TypeParam;
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    T initial_res = 100;
    T rhs_val = 10;
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({rhs_val}, this->exec_);
    std::shared_ptr<Mtx> x =
        gko::initialize<Mtx>({rhs_val - initial_res}, this->exec_);
    std::shared_ptr<gko::LinOp> mtx = gko::initialize<Mtx>({1.0}, this->exec_);

    auto criterion = this->factory_->generate(mtx, rhs, x.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(criterion->update().solution(x.get()).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    x->at(0) = rhs_val - r<T>::value * T{1.1} * initial_res;
    ASSERT_FALSE(criterion->update().solution(x.get()).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    x->at(0) = rhs_val - r<T>::value * T{0.9} * initial_res;
    ASSERT_TRUE(criterion->update().solution(x.get()).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(ResidualNormReduction, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto res = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec_);
    auto res_norm =
        gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({I<T>{10.0, 10.0}}, this->exec_);
    auto criterion = this->factory_->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0, 0) = r<TypeParam>::value * 0.9 * res_norm->at(0, 0);
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    res_norm->at(0, 1) = r<TypeParam>::value * 0.9 * res_norm->at(0, 1);
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


template <typename T>
class RelativeResidualNorm : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    RelativeResidualNorm()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ = gko::stop::RelativeResidualNorm<T>::build()
                       .with_tolerance(r<T>::value)
                       .on(exec_);
    }

    std::unique_ptr<typename gko::stop::RelativeResidualNorm<T>::Factory>
        factory_;
    std::shared_ptr<const gko::Executor> exec_;
};

TYPED_TEST_SUITE(RelativeResidualNorm, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(RelativeResidualNorm, CanCreateFactory)
{
    ASSERT_NE(this->factory_, nullptr);
    ASSERT_EQ(this->factory_->get_parameters().tolerance, r<TypeParam>::value);
    ASSERT_EQ(this->factory_->get_executor(), this->exec_);
}


TYPED_TEST(RelativeResidualNorm, CannotCreateCriterionWithoutB)
{
    ASSERT_THROW(this->factory_->generate(nullptr, nullptr, nullptr, nullptr),
                 gko::NotSupported);
}


TYPED_TEST(RelativeResidualNorm, CanCreateCriterionWithB)
{
    using Mtx = typename TestFixture::Mtx;
    std::shared_ptr<gko::LinOp> scalar =
        gko::initialize<Mtx>({1.0}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, scalar, nullptr, nullptr);

    ASSERT_NE(criterion, nullptr);
}


TYPED_TEST(RelativeResidualNorm, WaitsTillResidualGoal)
{
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec_);
    auto rhs_norm = gko::initialize<NormVector>({I<T_nc>{0.0}}, this->exec_);
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
    auto res_norm = gko::initialize<NormVector>({100.0}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0) = r<TypeParam>::value * 1.1 * rhs_norm->at(0);
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    res_norm->at(0) = r<TypeParam>::value * 0.9 * rhs_norm->at(0);
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(RelativeResidualNorm, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto res = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec_);
    auto res_norm =
        gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({I<T>{10.0, 10.0}}, this->exec_);
    auto rhs_norm =
        gko::initialize<NormVector>({I<T_nc>{0.0, 0.0}}, this->exec_);
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
    auto criterion = this->factory_->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0, 0) = r<TypeParam>::value * 0.9 * rhs_norm->at(0, 0);
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    res_norm->at(0, 1) = r<TypeParam>::value * 0.9 * rhs_norm->at(0, 1);
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


template <typename T>
class ImplicitResidualNorm : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ImplicitResidualNorm()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ = gko::stop::ImplicitResidualNorm<T>::build()
                       .with_reduction_factor(r<T>::value)
                       .on(exec_);
        factory_2_ = gko::stop::ImplicitResidualNorm<T>::build()
                         .with_reduction_factor(r<T>::value)
                         .with_baseline(gko::stop::mode::initial_resnorm)
                         .on(exec_);
        factory_3_ = gko::stop::ImplicitResidualNorm<T>::build()
                         .with_reduction_factor(r<T>::value)
                         .with_baseline(gko::stop::mode::rhs_norm)
                         .on(exec_);
    }

    std::unique_ptr<typename gko::stop::ImplicitResidualNorm<T>::Factory>
        factory_;
    std::unique_ptr<typename gko::stop::ImplicitResidualNorm<T>::Factory>
        factory_2_;
    std::unique_ptr<typename gko::stop::ImplicitResidualNorm<T>::Factory>
        factory_3_;
    std::shared_ptr<const gko::Executor> exec_;
};

TYPED_TEST_SUITE(ImplicitResidualNorm, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(ImplicitResidualNorm, CanCreateFactory)
{
    ASSERT_NE(this->factory_, nullptr);
    ASSERT_EQ(this->factory_->get_parameters().reduction_factor,
              r<TypeParam>::value);
    ASSERT_EQ(this->factory_->get_parameters().baseline,
              gko::stop::mode::rhs_norm);
    ASSERT_EQ(this->factory_2_->get_parameters().baseline,
              gko::stop::mode::initial_resnorm);
    ASSERT_EQ(this->factory_3_->get_parameters().baseline,
              gko::stop::mode::rhs_norm);
    ASSERT_EQ(this->factory_->get_executor(), this->exec_);
}


TYPED_TEST(ImplicitResidualNorm, CannotCreateCriterionWithoutBAndInitRes)
{
    ASSERT_THROW(this->factory_->generate(nullptr, nullptr, nullptr, nullptr),
                 gko::NotSupported);
}


TYPED_TEST(ImplicitResidualNorm, CanCreateCriterionWithB)
{
    using Mtx = typename TestFixture::Mtx;
    std::shared_ptr<gko::LinOp> scalar =
        gko::initialize<Mtx>({1.0}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, scalar, nullptr, nullptr);

    ASSERT_NE(criterion, nullptr);
}


TYPED_TEST(ImplicitResidualNorm, CanCreateCriterionWithInitialRes)
{
    using Mtx = typename TestFixture::Mtx;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec_);
    auto criterion = this->factory_2_->generate(nullptr, nullptr, nullptr,
                                                initial_res.get());
    ASSERT_NE(criterion, nullptr);
}


TYPED_TEST(ImplicitResidualNorm, WaitsTillResidualGoal)
{
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec_);
    auto res_norm = gko::initialize<Mtx>({100.0}, this->exec_);
    auto rhs_norm = gko::initialize<NormVector>({I<T_nc>{0.0}}, this->exec_);
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
    auto criterion =
        this->factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0) = std::pow(r<TypeParam>::value * 1.1 * rhs_norm->at(0), 2);
    ASSERT_FALSE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    res_norm->at(0) = std::pow(r<TypeParam>::value * 0.9 * rhs_norm->at(0), 2);
    ASSERT_TRUE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(ImplicitResidualNorm, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto res = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec_);
    auto res_norm = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({I<T>{10.0, 10.0}}, this->exec_);
    auto rhs_norm =
        gko::initialize<NormVector>({I<T_nc>{0.0, 0.0}}, this->exec_);
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
    auto criterion = this->factory_->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();

    ASSERT_FALSE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0, 0) =
        std::pow(r<TypeParam>::value * 0.9 * rhs_norm->at(0, 0), 2);
    ASSERT_FALSE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    res_norm->at(0, 1) =
        std::pow(r<TypeParam>::value * 0.9 * rhs_norm->at(0, 1), 2);
    ASSERT_TRUE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


template <typename T>
class AbsoluteResidualNorm : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    AbsoluteResidualNorm()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ = gko::stop::AbsoluteResidualNorm<T>::build()
                       .with_tolerance(r<T>::value)
                       .on(exec_);
    }

    std::unique_ptr<typename gko::stop::AbsoluteResidualNorm<T>::Factory>
        factory_;
    std::shared_ptr<const gko::Executor> exec_;
};

TYPED_TEST_SUITE(AbsoluteResidualNorm, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(AbsoluteResidualNorm, CanCreateFactory)
{
    ASSERT_NE(this->factory_, nullptr);
    ASSERT_EQ(this->factory_->get_parameters().tolerance, r<TypeParam>::value);
    ASSERT_EQ(this->factory_->get_executor(), this->exec_);
}


TYPED_TEST(AbsoluteResidualNorm, CannotCreateCriterionWithoutB)
{
    ASSERT_THROW(this->factory_->generate(nullptr, nullptr, nullptr, nullptr),
                 gko::NotSupported);
}


TYPED_TEST(AbsoluteResidualNorm, CanCreateCriterionWithB)
{
    using Mtx = typename TestFixture::Mtx;
    std::shared_ptr<gko::LinOp> scalar =
        gko::initialize<Mtx>({1.0}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, scalar, nullptr, nullptr);

    ASSERT_NE(criterion, nullptr);
}


TYPED_TEST(AbsoluteResidualNorm, WaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec_);
    auto res_norm = gko::initialize<NormVector>({100.0}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0) = r<TypeParam>::value * 1.1;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    res_norm->at(0) = r<TypeParam>::value * 0.9;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(AbsoluteResidualNorm, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto res = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec_);
    auto res_norm =
        gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({I<T>{10.0, 10.0}}, this->exec_);
    auto criterion = this->factory_->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0, 0) = r<TypeParam>::value * 0.9;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    res_norm->at(0, 1) = r<TypeParam>::value * 0.9;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


}  // namespace
