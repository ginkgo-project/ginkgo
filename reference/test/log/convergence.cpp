// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/convergence.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Convergence : public ::testing::Test {};

TYPED_TEST_SUITE(Convergence, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Convergence, CatchesCriterionCheckCompleted)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Convergence<TypeParam>::create(
        gko::log::Logger::criterion_check_completed_mask);
    auto criterion =
        gko::stop::Iteration::build().with_max_iters(3u).on(exec)->generate(
            nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::array<gko::stopping_status> stop_status(exec, 1);
    stop_status.get_data()[0].reset();
    using Mtx = gko::matrix::Dense<TypeParam>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<TypeParam>>;
    auto residual = gko::initialize<Mtx>({1.0, 2.0, 2.0}, exec);

    logger->template on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), 1, residual.get(), nullptr, nullptr,
        RelativeStoppingId, true, &stop_status, true, true);

    ASSERT_EQ(logger->get_num_iterations(), 1);
    ASSERT_EQ(logger->has_converged(), false);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(logger->get_residual()),
                        l({1.0, 2.0, 2.0}), 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<NormVector>(logger->get_residual_norm()),
                        l({3.0}), 0.0);
}


TYPED_TEST(Convergence, CatchesCriterionCheckCompletedWithConvCheck)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Convergence<TypeParam>::create(
        gko::log::Logger::criterion_check_completed_mask);
    auto criterion =
        gko::stop::Iteration::build().with_max_iters(3u).on(exec)->generate(
            nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::array<gko::stopping_status> stop_status(exec, 1);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[0].converge(0);
    using Mtx = gko::matrix::Dense<TypeParam>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<TypeParam>>;
    auto residual = gko::initialize<Mtx>({1.0, 2.0, 2.0}, exec);

    logger->template on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), 3, residual.get(), nullptr, nullptr,
        RelativeStoppingId, true, &stop_status, true, true);

    ASSERT_EQ(logger->get_num_iterations(), 3);
    ASSERT_EQ(logger->has_converged(), true);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(logger->get_residual()),
                        l({1.0, 2.0, 2.0}), 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<NormVector>(logger->get_residual_norm()),
                        l({3.0}), 0.0);
}


TYPED_TEST(Convergence, CatchesCriterionCheckCompletedWithStopCheck)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Convergence<TypeParam>::create(
        gko::log::Logger::criterion_check_completed_mask);
    auto criterion =
        gko::stop::Iteration::build().with_max_iters(3u).on(exec)->generate(
            nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::array<gko::stopping_status> stop_status(exec, 1);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[0].stop(0);
    using Mtx = gko::matrix::Dense<TypeParam>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<TypeParam>>;
    auto residual = gko::initialize<Mtx>({1.0, 2.0, 2.0}, exec);

    logger->template on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), 3, residual.get(), nullptr, nullptr,
        RelativeStoppingId, true, &stop_status, true, true);

    ASSERT_EQ(logger->get_num_iterations(), 3);
    ASSERT_EQ(logger->has_converged(), false);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(logger->get_residual()),
                        l({1.0, 2.0, 2.0}), 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<NormVector>(logger->get_residual_norm()),
                        l({3.0}), 0.0);
}


TYPED_TEST(Convergence, CanResetConvergenceStatus)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Convergence<TypeParam>::create(
        gko::log::Logger::criterion_check_completed_mask);
    auto criterion =
        gko::stop::Iteration::build().with_max_iters(3u).on(exec)->generate(
            nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::array<gko::stopping_status> stop_status(exec, 1);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[0].converge(0);
    using Mtx = gko::matrix::Dense<TypeParam>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<TypeParam>>;
    auto residual = gko::initialize<Mtx>({1.0, 2.0, 2.0}, exec);

    logger->template on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), 3, residual.get(), nullptr, nullptr,
        RelativeStoppingId, true, &stop_status, true, true);
    ASSERT_EQ(logger->get_num_iterations(), 3);
    ASSERT_EQ(logger->has_converged(), true);

    logger->reset_convergence_status();

    ASSERT_EQ(logger->has_converged(), false);
}


TYPED_TEST(Convergence, CatchesCriterionCheckCompletedWithImplicitNorm)
{
    auto exec = gko::ReferenceExecutor::create();
    auto logger = gko::log::Convergence<TypeParam>::create(
        gko::log::Logger::criterion_check_completed_mask);
    auto criterion =
        gko::stop::Iteration::build().with_max_iters(3u).on(exec)->generate(
            nullptr, nullptr, nullptr);
    constexpr gko::uint8 RelativeStoppingId{42};
    gko::array<gko::stopping_status> stop_status(exec, 1);
    stop_status.get_data()[0].reset();
    using Mtx = gko::matrix::Dense<TypeParam>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<TypeParam>>;
    auto residual = gko::initialize<Mtx>({1.0, 2.0, 2.0}, exec);
    auto implicit_sq_resnorm = gko::initialize<Mtx>({4.0}, exec);

    logger->template on<gko::log::Logger::criterion_check_completed>(
        criterion.get(), 1, residual.get(), nullptr, implicit_sq_resnorm.get(),
        nullptr, RelativeStoppingId, true, &stop_status, true, true);

    ASSERT_EQ(logger->get_num_iterations(), 1);
    ASSERT_EQ(logger->has_converged(), false);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(logger->get_implicit_sq_resnorm()),
                        l({4.0}), 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(logger->get_residual()),
                        l({1.0, 2.0, 2.0}), 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<NormVector>(logger->get_residual_norm()),
                        l({3.0}), 0.0);
}


}  // namespace
