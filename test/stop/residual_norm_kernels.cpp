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

#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


template <typename Mtx>
void write(std::unique_ptr<Mtx>& mtx, gko::size_type row, gko::size_type col,
           typename Mtx::value_type val)
{
    mtx->get_executor()->copy_from(
        mtx->get_executor()->get_master().get(), 1, &val,
        mtx->get_values() + col + mtx->get_stride() * row);
}


template <typename Mtx>
typename Mtx::value_type read(std::unique_ptr<Mtx>& mtx, gko::size_type row,
                              gko::size_type col)
{
    return mtx->get_executor()->copy_val_to_host(mtx->get_values() + col +
                                                 mtx->get_stride() * row);
}


template <typename T>
class ResidualNorm : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ResidualNorm()
    {
        factory = gko::stop::ResidualNorm<T>::build()
                      .with_reduction_factor(r<T>::value)
                      .on(exec);
        rel_factory = gko::stop::ResidualNorm<T>::build()
                          .with_reduction_factor(r<T>::value)
                          .with_baseline(gko::stop::mode::initial_resnorm)
                          .on(exec);
        abs_factory = gko::stop::ResidualNorm<T>::build()
                          .with_reduction_factor(r<T>::value)
                          .with_baseline(gko::stop::mode::absolute)
                          .on(exec);
    }

    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> factory;
    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> rel_factory;
    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> abs_factory;
};

TYPED_TEST_SUITE(ResidualNorm, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(ResidualNorm, WaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec);
    auto criterion =
        this->factory->generate(nullptr, rhs, nullptr, initial_res.get());
    auto rel_criterion =
        this->rel_factory->generate(nullptr, rhs, nullptr, initial_res.get());
    auto abs_criterion =
        this->abs_factory->generate(nullptr, rhs, nullptr, initial_res.get());
    {
        auto res_norm = gko::initialize<NormVector>({10.0}, this->exec);
        auto rhs_norm = gko::initialize<NormVector>({100.0}, this->exec);
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->ref, 1);
        stop_status.get_data()[0].reset();
        stop_status.set_executor(this->exec);

        ASSERT_FALSE(
            criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 1.1 * read(res_norm, 0, 0));
        ASSERT_FALSE(
            criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_FALSE(stop_status.get_data()[0].has_converged());
        ASSERT_FALSE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(res_norm, 0, 0));
        ASSERT_TRUE(
            criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[0].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);
    }
    {
        auto res_norm = gko::initialize<NormVector>({100.0}, this->exec);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->ref, 1);
        stop_status.get_data()[0].reset();
        stop_status.set_executor(this->exec);

        ASSERT_FALSE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 1.1 * read(res_norm, 0, 0));
        ASSERT_FALSE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_FALSE(stop_status.get_data()[0].has_converged());
        ASSERT_FALSE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(res_norm, 0, 0));
        ASSERT_TRUE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[0].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);
    }
    {
        auto res_norm = gko::initialize<NormVector>({100.0}, this->exec);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->ref, 1);
        stop_status.get_data()[0].reset();
        stop_status.set_executor(this->exec);

        ASSERT_FALSE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 1.1);
        ASSERT_FALSE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_FALSE(stop_status.get_data()[0].has_converged());
        ASSERT_FALSE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9);
        ASSERT_TRUE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[0].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);
    }
}


TYPED_TEST(ResidualNorm, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto res = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec);
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({I<T>{10.0, 10.0}}, this->exec);
    auto criterion = this->factory->generate(nullptr, rhs, nullptr, res.get());
    auto rel_criterion =
        this->rel_factory->generate(nullptr, rhs, nullptr, res.get());
    auto abs_criterion =
        this->abs_factory->generate(nullptr, rhs, nullptr, res.get());
    {
        auto res_norm =
            gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec);
        auto rhs_norm =
            gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec);
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
        bool one_changed{};
        constexpr gko::uint8 RelativeStoppingId{1};
        gko::array<gko::stopping_status> stop_status(this->ref, 2);
        stop_status.get_data()[0].reset();
        stop_status.get_data()[1].reset();
        stop_status.set_executor(this->exec);

        ASSERT_FALSE(
            criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 0));
        ASSERT_FALSE(
            criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[0].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 1, r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 1));
        ASSERT_TRUE(
            criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[1].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);
    }
    {
        auto res_norm =
            gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec);
        bool one_changed{};
        constexpr gko::uint8 RelativeStoppingId{1};
        gko::array<gko::stopping_status> stop_status(this->ref, 2);
        stop_status.get_data()[0].reset();
        stop_status.get_data()[1].reset();
        stop_status.set_executor(this->exec);

        ASSERT_FALSE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(res_norm, 0, 0));
        ASSERT_FALSE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[0].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 1, r<TypeParam>::value * 0.9 * read(res_norm, 0, 1));
        ASSERT_TRUE(
            rel_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[1].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);
    }
    {
        auto res_norm =
            gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec);
        bool one_changed{};
        constexpr gko::uint8 RelativeStoppingId{1};
        gko::array<gko::stopping_status> stop_status(this->ref, 2);
        stop_status.get_data()[0].reset();
        stop_status.get_data()[1].reset();
        stop_status.set_executor(this->exec);

        ASSERT_FALSE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9);
        ASSERT_FALSE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[0].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 1, r<TypeParam>::value * 0.9);
        ASSERT_TRUE(
            abs_criterion->update()
                .residual_norm(res_norm.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[1].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);
    }
}


template <typename T>
class ResidualNormWithInitialResnorm : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ResidualNormWithInitialResnorm()
    {
        factory = gko::stop::ResidualNorm<T>::build()
                      .with_baseline(gko::stop::mode::initial_resnorm)
                      .with_reduction_factor(r<T>::value)
                      .on(exec);
    }

    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> factory;
};

TYPED_TEST_SUITE(ResidualNormWithInitialResnorm, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(ResidualNormWithInitialResnorm, WaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec);
    auto res_norm = gko::initialize<NormVector>({100.0}, this->exec);
    auto criterion =
        this->factory->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 1);
    stop_status.get_data()[0].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 1.1 * read(res_norm, 0, 0));
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    ASSERT_FALSE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(res_norm, 0, 0));
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);
}


TYPED_TEST(ResidualNormWithInitialResnorm, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto res = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec);
    auto res_norm =
        gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec);
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({I<T>{10.0, 10.0}}, this->exec);
    auto criterion = this->factory->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(res_norm, 0, 0));
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 1, r<TypeParam>::value * 0.9 * read(res_norm, 0, 1));
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[1].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);
}


template <typename T>
class ResidualNormWithRhsNorm : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ResidualNormWithRhsNorm()
    {
        factory = gko::stop::ResidualNorm<T>::build()
                      .with_baseline(gko::stop::mode::rhs_norm)
                      .with_reduction_factor(r<T>::value)
                      .on(exec);
    }

    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> factory;
};

TYPED_TEST_SUITE(ResidualNormWithRhsNorm, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(ResidualNormWithRhsNorm, WaitsTillResidualGoal)
{
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec);
    auto rhs_norm = gko::initialize<NormVector>({I<T_nc>{0.0}}, this->exec);
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
    auto res_norm = gko::initialize<NormVector>({100.0}, this->exec);
    auto criterion =
        this->factory->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 1);
    stop_status.get_data()[0].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 1.1 * read(rhs_norm, 0, 0));
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    ASSERT_FALSE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 0));
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);
}


TYPED_TEST(ResidualNormWithRhsNorm, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto res = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec);
    auto res_norm =
        gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec);
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({I<T>{10.0, 10.0}}, this->exec);
    auto rhs_norm =
        gko::initialize<NormVector>({I<T_nc>{0.0, 0.0}}, this->exec);
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
    auto criterion = this->factory->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 0));
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 1, r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 1));
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[1].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);
}


template <typename T>
class ImplicitResidualNorm : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ImplicitResidualNorm()
    {
        factory = gko::stop::ImplicitResidualNorm<T>::build()
                      .with_reduction_factor(r<T>::value)
                      .on(exec);
    }

    std::unique_ptr<typename gko::stop::ImplicitResidualNorm<T>::Factory>
        factory;
};

TYPED_TEST_SUITE(ImplicitResidualNorm, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(ImplicitResidualNorm, WaitsTillResidualGoal)
{
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec);
    auto res_norm = gko::initialize<Mtx>({100.0}, this->exec);
    auto rhs_norm = gko::initialize<NormVector>({I<T_nc>{0.0}}, this->exec);
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
    auto criterion =
        this->factory->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 1);
    stop_status.get_data()[0].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0,
          std::pow(r<TypeParam>::value * 1.1 * read(rhs_norm, 0, 0), 2));
    ASSERT_FALSE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    ASSERT_FALSE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 0,
          std::pow(r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 0), 2));
    ASSERT_TRUE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);
}


TYPED_TEST(ImplicitResidualNorm, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto res = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec);
    auto res_norm = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec);
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({I<T>{10.0, 10.0}}, this->exec);
    auto rhs_norm =
        gko::initialize<NormVector>({I<T_nc>{0.0, 0.0}}, this->exec);
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm.get());
    auto criterion = this->factory->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0,
          std::pow(r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 0), 2));
    ASSERT_FALSE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 1,
          std::pow(r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 1), 2));
    ASSERT_TRUE(
        criterion->update()
            .implicit_sq_residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[1].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);
}


template <typename T>
class ResidualNormWithAbsolute : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<T>>;

    ResidualNormWithAbsolute()
    {
        factory = gko::stop::ResidualNorm<T>::build()
                      .with_baseline(gko::stop::mode::absolute)
                      .with_reduction_factor(r<T>::value)
                      .on(exec);
    }

    std::unique_ptr<typename gko::stop::ResidualNorm<T>::Factory> factory;
};

TYPED_TEST_SUITE(ResidualNormWithAbsolute, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(ResidualNormWithAbsolute, WaitsTillResidualGoal)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec);
    auto res_norm = gko::initialize<NormVector>({100.0}, this->exec);
    auto criterion =
        this->factory->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 1);
    stop_status.get_data()[0].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 1.1);
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    ASSERT_FALSE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9);
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);
}


TYPED_TEST(ResidualNormWithAbsolute, WaitsTillResidualGoalMultipleRHS)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = TypeParam;
    using T_nc = gko::remove_complex<TypeParam>;
    auto res = gko::initialize<Mtx>({I<T>{100.0, 100.0}}, this->exec);
    auto res_norm =
        gko::initialize<NormVector>({I<T_nc>{100.0, 100.0}}, this->exec);
    std::shared_ptr<gko::LinOp> rhs =
        gko::initialize<Mtx>({I<T>{10.0, 10.0}}, this->exec);
    auto criterion = this->factory->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9);
    ASSERT_FALSE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 1, r<TypeParam>::value * 0.9);
    ASSERT_TRUE(
        criterion->update()
            .residual_norm(res_norm.get())
            .check(RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[1].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);
}
