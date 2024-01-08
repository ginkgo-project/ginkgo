// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/convergence.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/ir.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Convergence : public ::testing::Test {
public:
    using Dense = gko::matrix::Dense<T>;
    using AbsoluteDense = gko::matrix::Dense<gko::remove_complex<T>>;

    Convergence()
    {
        status.get_data()[0].reset();
        status.get_data()[0].converge(0);
    }

    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();

    std::unique_ptr<Dense> residual = gko::initialize<Dense>({3, 4}, exec);
    std::unique_ptr<AbsoluteDense> residual_norm =
        gko::initialize<AbsoluteDense>({5}, exec);
    std::unique_ptr<AbsoluteDense> implicit_sq_resnorm =
        gko::initialize<AbsoluteDense>({6}, exec);
    std::unique_ptr<gko::LinOp> system =
        gko::solver::Ir<T>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec)
            ->generate(gko::initialize<Dense>(I<I<T>>{{1, 2}, {0, 3}}, exec));
    std::unique_ptr<Dense> rhs = gko::initialize<Dense>({15, 25}, exec);
    std::unique_ptr<Dense> solution = gko::initialize<Dense>({-2, 7}, exec);

    gko::array<gko::stopping_status> status = {exec, 1};
};

TYPED_TEST_SUITE(Convergence, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Convergence, CanGetEmptyData)
{
    auto logger = gko::log::Convergence<TypeParam>::create();

    ASSERT_EQ(logger->has_converged(), false);
    ASSERT_EQ(logger->get_num_iterations(), 0);
    ASSERT_EQ(logger->get_residual(), nullptr);
    ASSERT_EQ(logger->get_residual_norm(), nullptr);
    ASSERT_EQ(logger->get_implicit_sq_resnorm(), nullptr);
}


TYPED_TEST(Convergence, CanLogData)
{
    using Dense = gko::matrix::Dense<TypeParam>;
    using AbsoluteDense = gko::matrix::Dense<gko::remove_complex<TypeParam>>;
    auto logger = gko::log::Convergence<TypeParam>::create();

    logger->template on<gko::log::Logger::iteration_complete>(
        this->system.get(), this->rhs.get(), this->solution.get(), 100,
        this->residual.get(), this->residual_norm.get(),
        this->implicit_sq_resnorm.get(), &this->status, true);

    ASSERT_EQ(logger->has_converged(), true);
    ASSERT_EQ(logger->get_num_iterations(), 100);
    GKO_ASSERT_MTX_NEAR(gko::as<Dense>(logger->get_residual()), this->residual,
                        0);
    GKO_ASSERT_MTX_NEAR(gko::as<AbsoluteDense>(logger->get_residual_norm()),
                        this->residual_norm, 0);
    GKO_ASSERT_MTX_NEAR(
        gko::as<AbsoluteDense>(logger->get_implicit_sq_resnorm()),
        this->implicit_sq_resnorm, 0);
}


TYPED_TEST(Convergence, DoesNotLogIfNotStopped)
{
    auto logger = gko::log::Convergence<TypeParam>::create();

    logger->template on<gko::log::Logger::iteration_complete>(
        this->system.get(), this->rhs.get(), this->solution.get(), 100,
        this->residual.get(), this->residual_norm.get(),
        this->implicit_sq_resnorm.get(), &this->status, false);

    ASSERT_EQ(logger->has_converged(), false);
    ASSERT_EQ(logger->get_num_iterations(), 0);
    ASSERT_EQ(logger->get_residual(), nullptr);
    ASSERT_EQ(logger->get_residual_norm(), nullptr);
}


TYPED_TEST(Convergence, CanComputeResidualNormFromResidual)
{
    using AbsoluteDense = gko::matrix::Dense<gko::remove_complex<TypeParam>>;
    auto logger = gko::log::Convergence<TypeParam>::create();

    logger->template on<gko::log::Logger::iteration_complete>(
        this->system.get(), this->rhs.get(), this->solution.get(), 100,
        this->residual.get(), nullptr, nullptr, &this->status, true);

    GKO_ASSERT_MTX_NEAR(gko::as<AbsoluteDense>(logger->get_residual_norm()),
                        this->residual_norm, r<TypeParam>::value);
}


TYPED_TEST(Convergence, CanComputeResidualNormFromSolution)
{
    using AbsoluteDense = gko::matrix::Dense<gko::remove_complex<TypeParam>>;
    auto logger = gko::log::Convergence<TypeParam>::create();

    logger->template on<gko::log::Logger::iteration_complete>(
        this->system.get(), this->rhs.get(), this->solution.get(), 100, nullptr,
        nullptr, nullptr, &this->status, true);

    GKO_ASSERT_MTX_NEAR(gko::as<AbsoluteDense>(logger->get_residual_norm()),
                        this->residual_norm, r<TypeParam>::value);
}


}  // namespace
