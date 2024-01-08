// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    using ValueType = T;

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

TYPED_TEST(ResidualNorm, CheckIfResZeroConverges)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = typename TestFixture::ValueType;
    using gko::stop::mode;
    std::shared_ptr<gko::LinOp> mtx = gko::initialize<Mtx>({1.0}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({0.0}, this->exec_);
    std::shared_ptr<gko::LinOp> x = gko::initialize<Mtx>({0.0}, this->exec_);
    std::shared_ptr<gko::LinOp> res_norm =
        gko::initialize<NormVector>({0.0}, this->exec_);

    for (auto baseline :
         {mode::rhs_norm, mode::initial_resnorm, mode::absolute}) {
        gko::remove_complex<T> factor =
            (baseline == mode::absolute) ? 0.0 : r<T>::value;
        auto criterion = gko::stop::ResidualNorm<T>::build()
                             .with_reduction_factor(factor)
                             .with_baseline(baseline)
                             .on(this->exec_)
                             ->generate(mtx, rhs, x.get(), nullptr);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        EXPECT_TRUE(criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        EXPECT_TRUE(stop_status.get_data()[0].has_converged());
        EXPECT_TRUE(one_changed);
    }
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


TYPED_TEST(ResidualNorm, CanIgorneResidualNorm)
{
    using Mtx = typename TestFixture::Mtx;
    std::shared_ptr<gko::LinOp> scalar =
        gko::initialize<Mtx>({1.0}, this->exec_);
    auto criterion =
        this->rhs_factory_->generate(nullptr, scalar, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{1};
    bool one_changed{};
    gko::array<gko::stopping_status> stop_status(this->exec_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(criterion->update().ignore_residual_check(true).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_THROW(criterion->update().check(RelativeStoppingId, true,
                                           &stop_status, &one_changed),
                 gko::NotSupported);
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
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_FALSE(rhs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0) = r<TypeParam>::value * 1.1 * rhs_norm->at(0);
        ASSERT_FALSE(rhs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        res_norm->at(0) = r<TypeParam>::value * 0.9 * rhs_norm->at(0);
        ASSERT_TRUE(rhs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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

        ASSERT_FALSE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0) = r<TypeParam>::value * 1.1 * init_res_val;
        ASSERT_FALSE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        res_norm->at(0) = r<TypeParam>::value * 0.9 * init_res_val;
        ASSERT_TRUE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
    {
        auto res_norm = gko::initialize<NormVector>({100.0}, this->exec_);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_FALSE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0) = r<TypeParam>::value * 1.1;
        ASSERT_FALSE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        res_norm->at(0) = r<TypeParam>::value * 0.9;
        ASSERT_TRUE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
}


TYPED_TEST(ResidualNorm, SelfCalculatesThrowWithoutMatrix)
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
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_THROW(rhs_criterion->update().solution(solution).check(
                         RelativeStoppingId, true, &stop_status, &one_changed),
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

        ASSERT_THROW(rel_criterion->update().solution(solution).check(
                         RelativeStoppingId, true, &stop_status, &one_changed),
                     gko::NotSupported);
    }
    {
        auto solution = gko::initialize<Mtx>({rhs_val - T{100.0}}, this->exec_);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_THROW(abs_criterion->update().solution(solution).check(
                         RelativeStoppingId, true, &stop_status, &one_changed),
                     gko::NotSupported);
    }
}


TYPED_TEST(ResidualNorm, RelativeSelfCalculatesThrowWithoutRhs)
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

    ASSERT_THROW(rel_criterion->update().solution(solution).check(
                     RelativeStoppingId, true, &stop_status, &one_changed),
                 gko::NotSupported);
}


TYPED_TEST(ResidualNorm, SelfCalculatesAndWaitsTillResidualGoal)
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
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_FALSE(rhs_criterion->update().solution(solution).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        solution->at(0) = rhs_val - r<T>::value * T{1.1} * rhs_norm->at(0);
        ASSERT_FALSE(rhs_criterion->update().solution(solution).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        solution->at(0) = rhs_val - r<T>::value * T{0.5} * rhs_norm->at(0);
        ASSERT_TRUE(rhs_criterion->update().solution(solution).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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

        ASSERT_FALSE(rel_criterion->update().solution(solution).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        solution->at(0) = rhs_val - r<T>::value * T{1.1} * initial_norm;
        ASSERT_FALSE(rel_criterion->update().solution(solution).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        solution->at(0) = rhs_val - r<T>::value * T{0.5} * initial_norm;
        ASSERT_TRUE(rel_criterion->update().solution(solution).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
    {
        auto solution = gko::initialize<Mtx>({rhs_val - T{100.0}}, this->exec_);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        ASSERT_FALSE(abs_criterion->update().solution(solution).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        solution->at(0) = rhs_val - r<T>::value * T{1.2};
        ASSERT_FALSE(abs_criterion->update().solution(solution).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
        ASSERT_EQ(one_changed, false);

        solution->at(0) = rhs_val - r<T>::value * T{0.5};
        ASSERT_TRUE(abs_criterion->update().solution(solution).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
        bool one_changed{};
        constexpr gko::uint8 RelativeStoppingId{1};
        gko::array<gko::stopping_status> stop_status(this->exec_, 2);
        stop_status.get_data()[0].reset();
        stop_status.get_data()[1].reset();

        ASSERT_FALSE(rhs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0, 0) = r<TypeParam>::value * 0.9 * rhs_norm->at(0, 0);
        ASSERT_FALSE(rhs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);

        res_norm->at(0, 1) = r<TypeParam>::value * 0.9 * rhs_norm->at(0, 1);
        ASSERT_TRUE(rhs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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

        ASSERT_FALSE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0, 0) = r<TypeParam>::value * 0.9 * res_norm->at(0, 0);
        ASSERT_FALSE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);

        res_norm->at(0, 1) = r<TypeParam>::value * 0.9 * res_norm->at(0, 1);
        ASSERT_TRUE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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

        ASSERT_FALSE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        res_norm->at(0, 0) = r<TypeParam>::value * 0.9;
        ASSERT_FALSE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
        ASSERT_EQ(one_changed, true);

        res_norm->at(0, 1) = r<TypeParam>::value * 0.9;
        ASSERT_TRUE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
        ASSERT_EQ(one_changed, true);
    }
}


template <typename T>
class ResidualNormWithInitialResnorm : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ResidualNormWithInitialResnorm()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ = gko::stop::ResidualNorm<T>::build()
                       .with_baseline(gko::stop::mode::initial_resnorm)
                       .with_reduction_factor(r<T>::value)
                       .on(exec_);
    }

    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> factory_;
    std::shared_ptr<const gko::ReferenceExecutor> exec_;
};

TYPED_TEST_SUITE(ResidualNormWithInitialResnorm, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(ResidualNormWithInitialResnorm,
           CanCreateCriterionWithMtxRhsXWithoutInitialRes)
{
    using Mtx = typename TestFixture::Mtx;
    std::shared_ptr<gko::LinOp> x = gko::initialize<Mtx>({100.0}, this->exec_);
    std::shared_ptr<gko::LinOp> mtx = gko::initialize<Mtx>({1.0}, this->exec_);
    std::shared_ptr<gko::LinOp> b = gko::initialize<Mtx>({10.0}, this->exec_);

    auto criterion = this->factory_->generate(mtx, b, x.get());

    ASSERT_NE(criterion, nullptr);
}


TYPED_TEST(ResidualNormWithInitialResnorm, WaitsTillResidualGoal)
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

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0) = r<TypeParam>::value * 1.1 * init_res_val;
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    res_norm->at(0) = r<TypeParam>::value * 0.9 * init_res_val;
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(ResidualNormWithInitialResnorm,
           WaitsTillResidualGoalWithoutInitialRes)
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

    ASSERT_FALSE(criterion->update().solution(x).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    x->at(0) = rhs_val - r<T>::value * T{1.1} * initial_res;
    ASSERT_FALSE(criterion->update().solution(x).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    x->at(0) = rhs_val - r<T>::value * T{0.5} * initial_res;
    ASSERT_TRUE(criterion->update().solution(x).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(ResidualNormWithInitialResnorm, WaitsTillResidualGoalMultipleRHS)
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

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0, 0) = r<TypeParam>::value * 0.9 * res_norm->at(0, 0);
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    res_norm->at(0, 1) = r<TypeParam>::value * 0.9 * res_norm->at(0, 1);
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


template <typename T>
class ResidualNormWithRhsNorm : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ResidualNormWithRhsNorm()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ = gko::stop::ResidualNorm<T>::build()
                       .with_baseline(gko::stop::mode::rhs_norm)
                       .with_reduction_factor(r<T>::value)
                       .on(exec_);
    }

    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};

TYPED_TEST_SUITE(ResidualNormWithRhsNorm, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(ResidualNormWithRhsNorm, CanCreateFactory)
{
    ASSERT_NE(this->factory_, nullptr);
    ASSERT_EQ(this->factory_->get_parameters().reduction_factor,
              r<TypeParam>::value);
    ASSERT_EQ(this->factory_->get_executor(), this->exec_);
}


TYPED_TEST(ResidualNormWithRhsNorm, CannotCreateCriterionWithoutB)
{
    ASSERT_THROW(this->factory_->generate(nullptr, nullptr, nullptr, nullptr),
                 gko::NotSupported);
}


TYPED_TEST(ResidualNormWithRhsNorm, CanCreateCriterionWithB)
{
    using Mtx = typename TestFixture::Mtx;
    std::shared_ptr<gko::LinOp> scalar =
        gko::initialize<Mtx>({1.0}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, scalar, nullptr, nullptr);

    ASSERT_NE(criterion, nullptr);
}


TYPED_TEST(ResidualNormWithRhsNorm, WaitsTillResidualGoal)
{
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec_);
    auto rhs_norm = gko::initialize<NormVector>({I<T_nc>{0.0}}, this->exec_);
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
    auto res_norm = gko::initialize<NormVector>({100.0}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0) = r<TypeParam>::value * 1.1 * rhs_norm->at(0);
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    res_norm->at(0) = r<TypeParam>::value * 0.9 * rhs_norm->at(0);
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(ResidualNormWithRhsNorm, WaitsTillResidualGoalMultipleRHS)
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
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
    auto criterion = this->factory_->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0, 0) = r<TypeParam>::value * 0.9 * rhs_norm->at(0, 0);
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    res_norm->at(0, 1) = r<TypeParam>::value * 0.9 * rhs_norm->at(0, 1);
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


template <typename T>
class ImplicitResidualNorm : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;
    using ValueType = T;

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

TYPED_TEST(ImplicitResidualNorm, CheckIfResZeroConverges)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::ValueType;
    using gko::stop::mode;
    std::shared_ptr<gko::LinOp> mtx = gko::initialize<Mtx>({1.0}, this->exec_);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({0.0}, this->exec_);
    std::shared_ptr<gko::LinOp> x = gko::initialize<Mtx>({0.0}, this->exec_);
    std::shared_ptr<gko::LinOp> implicit_sq_res_norm =
        gko::initialize<Mtx>({0.0}, this->exec_);

    for (auto baseline :
         {mode::rhs_norm, mode::initial_resnorm, mode::absolute}) {
        gko::remove_complex<T> factor =
            (baseline == mode::absolute) ? 0.0 : r<T>::value;
        auto criterion = gko::stop::ImplicitResidualNorm<T>::build()
                             .with_reduction_factor(factor)
                             .with_baseline(baseline)
                             .on(this->exec_)
                             ->generate(mtx, rhs, x.get(), nullptr);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->exec_, 1);
        stop_status.get_data()[0].reset();

        EXPECT_TRUE(
            criterion->update()
                .implicit_sq_residual_norm(implicit_sq_res_norm)
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        EXPECT_TRUE(stop_status.get_data()[0].has_converged());
        EXPECT_TRUE(one_changed);
    }
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
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
    auto criterion =
        this->factory_->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0) = std::pow(r<TypeParam>::value * 1.1 * rhs_norm->at(0), 2);
    ASSERT_FALSE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    res_norm->at(0) = std::pow(r<TypeParam>::value * 0.9 * rhs_norm->at(0), 2);
    ASSERT_TRUE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
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
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
    auto criterion = this->factory_->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->exec_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();

    ASSERT_FALSE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0, 0) =
        std::pow(r<TypeParam>::value * 0.9 * rhs_norm->at(0, 0), 2);
    ASSERT_FALSE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    res_norm->at(0, 1) =
        std::pow(r<TypeParam>::value * 0.9 * rhs_norm->at(0, 1), 2);
    ASSERT_TRUE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


template <typename T>
class ResidualNormWithAbsolute : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ResidualNormWithAbsolute()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ = gko::stop::ResidualNorm<T>::build()
                       .with_baseline(gko::stop::mode::absolute)
                       .with_reduction_factor(r<T>::value)
                       .on(exec_);
    }

    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};

TYPED_TEST_SUITE(ResidualNormWithAbsolute, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(ResidualNormWithAbsolute, CanCreateFactory)
{
    ASSERT_NE(this->factory_, nullptr);
    ASSERT_EQ(this->factory_->get_parameters().reduction_factor,
              r<TypeParam>::value);
    ASSERT_EQ(this->factory_->get_executor(), this->exec_);
}


TYPED_TEST(ResidualNormWithAbsolute, CannotCreateCriterionWithoutB)
{
    ASSERT_THROW(this->factory_->generate(nullptr, nullptr, nullptr, nullptr),
                 gko::NotSupported);
}


TYPED_TEST(ResidualNormWithAbsolute, CanCreateCriterionWithB)
{
    using Mtx = typename TestFixture::Mtx;
    std::shared_ptr<gko::LinOp> scalar =
        gko::initialize<Mtx>({1.0}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, scalar, nullptr, nullptr);

    ASSERT_NE(criterion, nullptr);
}


TYPED_TEST(ResidualNormWithAbsolute, WaitsTillResidualGoal)
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

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0) = r<TypeParam>::value * 1.1;
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    res_norm->at(0) = r<TypeParam>::value * 0.9;
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(ResidualNormWithAbsolute, WaitsTillResidualGoalMultipleRHS)
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

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    res_norm->at(0, 0) = r<TypeParam>::value * 0.9;
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    res_norm->at(0, 1) = r<TypeParam>::value * 0.9;
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


}  // namespace
