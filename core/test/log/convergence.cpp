/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/log/convergence.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


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
    std::unique_ptr<Dense> solution = gko::initialize<Dense>({-2, 7}, exec);

    gko::array<gko::stopping_status> status = {exec, 1};
};

TYPED_TEST_SUITE(Convergence, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Convergence, CanGetEmptyData)
{
    auto logger = gko::log::Convergence<TypeParam>::create(
        gko::log::Logger::criterion_events_mask);

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
    auto logger = gko::log::Convergence<TypeParam>::create(
        gko::log::Logger::criterion_events_mask);

    logger->template on<gko::log::Logger::criterion_check_completed>(
        nullptr, 100, this->residual.get(), this->residual_norm.get(),
        this->implicit_sq_resnorm.get(), this->solution.get(), 0, false,
        &this->status, false, true);

    ASSERT_EQ(logger->has_converged(), true);
    ASSERT_EQ(logger->get_num_iterations(), 100);
    GKO_ASSERT_MTX_NEAR(gko::as<Dense>(logger->get_residual()),
                        this->residual.get(), 0);
    GKO_ASSERT_MTX_NEAR(gko::as<AbsoluteDense>(logger->get_residual_norm()),
                        this->residual_norm.get(), 0);
    GKO_ASSERT_MTX_NEAR(
        gko::as<AbsoluteDense>(logger->get_implicit_sq_resnorm()),
        this->implicit_sq_resnorm.get(), 0);
}


TYPED_TEST(Convergence, DoesNotLogIfNotStopped)
{
    auto logger = gko::log::Convergence<TypeParam>::create(
        gko::log::Logger::criterion_events_mask);

    logger->template on<gko::log::Logger::criterion_check_completed>(
        nullptr, 100, this->residual.get(), this->residual_norm.get(),
        this->implicit_sq_resnorm.get(), this->solution.get(), 0, false,
        &this->status, false, false);

    ASSERT_EQ(logger->has_converged(), false);
    ASSERT_EQ(logger->get_num_iterations(), 0);
    ASSERT_EQ(logger->get_residual(), nullptr);
    ASSERT_EQ(logger->get_residual_norm(), nullptr);
}


TYPED_TEST(Convergence, CanComputeResidualNorm)
{
    using AbsoluteDense = gko::matrix::Dense<gko::remove_complex<TypeParam>>;
    auto logger = gko::log::Convergence<TypeParam>::create(
        gko::log::Logger::criterion_events_mask);

    logger->template on<gko::log::Logger::criterion_check_completed>(
        nullptr, 100, this->residual.get(), nullptr, nullptr, nullptr, 0, false,
        &this->status, false, true);

    GKO_ASSERT_MTX_NEAR(gko::as<AbsoluteDense>(logger->get_residual_norm()),
                        this->residual_norm, r<TypeParam>::value);
}


}  // namespace
