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

#include <ginkgo/core/stop/residual_norm_reduction.hpp>


#include <type_traits>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


namespace {


template <typename T>
class ResidualNormReduction : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    static constexpr gko::remove_complex<T> reduction_factor =
        std::is_same<gko::remove_complex<T>, float>::value ? 1.0e-7 : 1.0e-14;

    ResidualNormReduction()
    {
        exec_ = gko::ReferenceExecutor::create();
        factory_ = gko::stop::ResidualNormReduction<T>::build()
                       .with_reduction_factor(reduction_factor)
                       .on(exec_);
    }

    std::unique_ptr<typename gko::stop::ResidualNormReduction<T>::Factory>
        factory_;
    std::shared_ptr<const gko::Executor> exec_;
};


template <typename T>
constexpr gko::remove_complex<T> ResidualNormReduction<T>::reduction_factor;


using ValueTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_CASE(ResidualNormReduction, ValueTypes);


TYPED_TEST(ResidualNormReduction, CanCreateFactory)
{
    ASSERT_NE(this->factory_, nullptr);
    ASSERT_EQ(this->factory_->get_parameters().reduction_factor,
              TestFixture::reduction_factor);
    ASSERT_EQ(this->factory_->get_executor(), this->exec_);
}


TYPED_TEST(ResidualNormReduction, CannotCreateCriterionWithoutB)
{
    ASSERT_THROW(this->factory_->generate(nullptr, nullptr, nullptr, nullptr),
                 gko::NotSupported);
}


TYPED_TEST(ResidualNormReduction, CanCreateCriterionWithB)
{
    using Mtx = typename TestFixture::Mtx;
    std::shared_ptr<gko::LinOp> scalar =
        gko::initialize<Mtx>({1.0}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, nullptr, nullptr, scalar.get());
    ASSERT_NE(criterion, nullptr);
}


TYPED_TEST(ResidualNormReduction, WaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    auto scalar = gko::initialize<Mtx>({1.0}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, nullptr, nullptr, scalar.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(this->exec_, 1);
    stop_status.get_data()[0].reset();

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(scalar.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    scalar->at(0) = TestFixture::reduction_factor * 1.0e+2;
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(scalar.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), false);
    ASSERT_EQ(one_changed, false);

    scalar->at(0) = TestFixture::reduction_factor * 1.0e-2;
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(scalar.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


TYPED_TEST(ResidualNormReduction, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    auto one = gko::one<TypeParam>();
    auto mtx = gko::initialize<Mtx>({{one, one}}, this->exec_);
    auto criterion =
        this->factory_->generate(nullptr, nullptr, nullptr, mtx.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::Array<gko::stopping_status> stop_status(this->exec_, 2);
    // Array only does malloc, it *does not* construct the object
    // therefore you get undefined values in your objects whatever you do.
    // Proper fix is not easy, we can't just call memset. We can probably not
    // call the placement constructor either
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();

    ASSERT_FALSE(criterion->update().residual_norm(mtx.get()).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    mtx->at(0, 0) = TestFixture::reduction_factor * 1.0e-2;
    ASSERT_FALSE(criterion->update().residual_norm(mtx.get()).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[0].has_converged(), true);
    ASSERT_EQ(one_changed, true);
    one_changed = false;

    mtx->at(0, 1) = TestFixture::reduction_factor * 1.0e-2;
    ASSERT_TRUE(criterion->update().residual_norm(mtx.get()).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_EQ(stop_status.get_data()[1].has_converged(), true);
    ASSERT_EQ(one_changed, true);
}


}  // namespace
