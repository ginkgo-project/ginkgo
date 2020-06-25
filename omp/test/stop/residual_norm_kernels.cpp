/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include <ginkgo/core/base/math.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class ResidualNormReduction : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ResidualNormReduction()
    {
        omp_ = gko::OmpExecutor::create();
        factory_ = gko::stop::ResidualNormReduction<T>::build()
                       .with_reduction_factor(r<T>::value)
                       .on(omp_);
    }

    std::unique_ptr<typename gko::stop::ResidualNormReduction<T>::Factory>
        factory_;
    std::shared_ptr<const gko::OmpExecutor> omp_;
};

TYPED_TEST_CASE(ResidualNormReduction, gko::test::ValueTypes);


TYPED_TEST(ResidualNormReduction, WaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto scalar = gko::initialize<Mtx>({1.0}, this->omp_);
    auto norm = gko::initialize<NormVector>({1.0}, this->omp_);
    auto criterion =
        this->factory_->generate(nullptr, nullptr, nullptr, scalar.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(this->omp_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    norm->at(0) = r<TypeParam>::value * 1.0e+2;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    norm->at(0) = r<TypeParam>::value * 1.0e-2;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(ResidualNormReduction, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto mtx = gko::initialize<Mtx>({I<T>{1.0, 1.0}}, this->omp_);
    auto norm = gko::initialize<NormVector>({I<T_nc>{1.0, 1.0}}, this->omp_);
    auto criterion =
        this->factory_->generate(nullptr, nullptr, nullptr, mtx.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(this->omp_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    norm->at(0, 0) = r<TypeParam>::value * 1.0e-2;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    norm->at(0, 1) = r<TypeParam>::value * 1.0e-2;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(norm.get())
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
        omp_ = gko::OmpExecutor::create();
        factory_ = gko::stop::RelativeResidualNorm<T>::build()
                       .with_tolerance(r<T>::value)
                       .on(omp_);
    }

    std::unique_ptr<typename gko::stop::RelativeResidualNorm<T>::Factory>
        factory_;
    std::shared_ptr<const gko::OmpExecutor> omp_;
};

TYPED_TEST_CASE(RelativeResidualNorm, gko::test::ValueTypes);


TYPED_TEST(RelativeResidualNorm, WaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    std::shared_ptr<gko::LinOp> scalar =
        gko::initialize<Mtx>({1.0}, this->omp_);
    auto norm = gko::initialize<NormVector>({1.0}, this->omp_);
    auto criterion =
        this->factory_->generate(nullptr, scalar, nullptr, nullptr);
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(this->omp_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    norm->at(0) = r<TypeParam>::value * 1.0e+2;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    norm->at(0) = r<TypeParam>::value * 1.0e-2;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(norm.get())
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
    std::shared_ptr<gko::LinOp> mtx =
        gko::initialize<Mtx>({I<T>{1.0, 1.0}}, this->omp_);
    auto norm = gko::initialize<NormVector>({I<T_nc>{1.0, 1.0}}, this->omp_);
    auto criterion = this->factory_->generate(nullptr, mtx, nullptr, nullptr);
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(this->omp_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    norm->at(0, 0) = r<TypeParam>::value * 1.0e-2;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    norm->at(0, 1) = r<TypeParam>::value * 1.0e-2;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(norm.get())
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
        omp_ = gko::OmpExecutor::create();
        factory_ = gko::stop::AbsoluteResidualNorm<T>::build()
                       .with_tolerance(r<T>::value)
                       .on(omp_);
    }

    std::unique_ptr<typename gko::stop::AbsoluteResidualNorm<T>::Factory>
        factory_;
    std::shared_ptr<const gko::OmpExecutor> omp_;
};

TYPED_TEST_CASE(AbsoluteResidualNorm, gko::test::ValueTypes);


TYPED_TEST(AbsoluteResidualNorm, WaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    std::shared_ptr<gko::LinOp> scalar =
        gko::initialize<Mtx>({1.0}, this->omp_);
    auto norm = gko::initialize<NormVector>({1.0}, this->omp_);
    auto criterion =
        this->factory_->generate(nullptr, scalar, nullptr, nullptr);
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(this->omp_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    norm->at(0) = r<TypeParam>::value * 1.0e+2;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    norm->at(0) = r<TypeParam>::value * 1.0e-5;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(norm.get())
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
    std::shared_ptr<gko::LinOp> mtx =
        gko::initialize<Mtx>({I<T>{1.0, 1.0}}, this->omp_);
    auto norm = gko::initialize<NormVector>({I<T_nc>{1.0, 1.0}}, this->omp_);
    auto criterion = this->factory_->generate(nullptr, mtx, nullptr, nullptr);
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(this->omp_, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    norm->at(0, 0) = r<TypeParam>::value * 1.0e-2;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);

    norm->at(0, 1) = r<TypeParam>::value * 1.0e-2;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


}  // namespace
