// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    using ValueType = T;

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


TYPED_TEST(ResidualNorm, CanIgorneResidualNorm)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    auto initial_res = gko::initialize<Mtx>({100.0}, this->exec);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({10.0}, this->exec);
    auto criterion =
        this->factory->generate(nullptr, rhs, nullptr, initial_res.get());
    constexpr gko::uint8 RelativeStoppingId{1};
    bool one_changed{};
    gko::array<gko::stopping_status> stop_status(this->ref, 1);
    stop_status.get_data()[0].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(criterion->update().ignore_residual_check(true).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    ASSERT_THROW(criterion->update().check(RelativeStoppingId, true,
                                           &stop_status, &one_changed),
                 gko::NotSupported);
}

TYPED_TEST(ResidualNorm, CheckIfResZeroConverges)
{
    using Mtx = typename TestFixture::Mtx;
    using NormVector = typename TestFixture::NormVector;
    using T = typename TestFixture::ValueType;
    using mode = gko::stop::mode;
    std::shared_ptr<gko::LinOp> mtx = gko::initialize<Mtx>({1.0}, this->exec);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({0.0}, this->exec);
    std::shared_ptr<gko::LinOp> x = gko::initialize<Mtx>({0.0}, this->exec);
    std::shared_ptr<gko::LinOp> res_norm =
        gko::initialize<NormVector>({0.0}, this->exec);

    for (auto baseline :
         {mode::rhs_norm, mode::initial_resnorm, mode::absolute}) {
        gko::remove_complex<T> factor =
            (baseline == mode::absolute) ? 0.0 : r<T>::value;
        auto criterion = gko::stop::ResidualNorm<T>::build()
                             .with_reduction_factor(factor)
                             .with_baseline(baseline)
                             .on(this->exec)
                             ->generate(mtx, rhs, x.get(), nullptr);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->ref, 1);
        stop_status.get_data()[0].reset();
        stop_status.set_executor(this->exec);

        EXPECT_TRUE(criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        EXPECT_TRUE(stop_status.get_data()[0].has_converged());
        EXPECT_TRUE(one_changed);
    }
}

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
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->ref, 1);
        stop_status.get_data()[0].reset();
        stop_status.set_executor(this->exec);

        ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 1.1 * read(res_norm, 0, 0));
        ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_FALSE(stop_status.get_data()[0].has_converged());
        ASSERT_FALSE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(res_norm, 0, 0));
        ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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

        ASSERT_FALSE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 1.1 * read(res_norm, 0, 0));
        ASSERT_FALSE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_FALSE(stop_status.get_data()[0].has_converged());
        ASSERT_FALSE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(res_norm, 0, 0));
        ASSERT_TRUE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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

        ASSERT_FALSE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 1.1);
        ASSERT_FALSE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_FALSE(stop_status.get_data()[0].has_converged());
        ASSERT_FALSE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9);
        ASSERT_TRUE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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
        gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
        bool one_changed{};
        constexpr gko::uint8 RelativeStoppingId{1};
        gko::array<gko::stopping_status> stop_status(this->ref, 2);
        stop_status.get_data()[0].reset();
        stop_status.get_data()[1].reset();
        stop_status.set_executor(this->exec);

        ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 0));
        ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[0].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 1, r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 1));
        ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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

        ASSERT_FALSE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(res_norm, 0, 0));
        ASSERT_FALSE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[0].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 1, r<TypeParam>::value * 0.9 * read(res_norm, 0, 1));
        ASSERT_TRUE(rel_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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

        ASSERT_FALSE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));

        write(res_norm, 0, 0, r<TypeParam>::value * 0.9);
        ASSERT_FALSE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        ASSERT_TRUE(stop_status.get_data()[0].has_converged());
        ASSERT_TRUE(one_changed);
        stop_status.set_executor(this->exec);

        write(res_norm, 0, 1, r<TypeParam>::value * 0.9);
        ASSERT_TRUE(abs_criterion->update().residual_norm(res_norm).check(
            RelativeStoppingId, true, &stop_status, &one_changed));
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

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 1.1 * read(res_norm, 0, 0));
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    ASSERT_FALSE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(res_norm, 0, 0));
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
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

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(res_norm, 0, 0));
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 1, r<TypeParam>::value * 0.9 * read(res_norm, 0, 1));
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
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
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
    auto res_norm = gko::initialize<NormVector>({100.0}, this->exec);
    auto criterion =
        this->factory->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 1);
    stop_status.get_data()[0].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 1.1 * read(rhs_norm, 0, 0));
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    ASSERT_FALSE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 0));
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
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
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
    auto criterion = this->factory->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 0));
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 1, r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 1));
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
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
    using ValueType = T;

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


TYPED_TEST(ImplicitResidualNorm, CheckIfResZeroConverges)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::ValueType;
    using gko::stop::mode;
    std::shared_ptr<gko::LinOp> mtx = gko::initialize<Mtx>({1.0}, this->exec);
    std::shared_ptr<gko::LinOp> rhs = gko::initialize<Mtx>({0.0}, this->exec);
    std::shared_ptr<gko::LinOp> x = gko::initialize<Mtx>({0.0}, this->exec);
    std::shared_ptr<gko::LinOp> implicit_sq_res_norm =
        gko::initialize<Mtx>({0.0}, this->exec);

    for (auto baseline :
         {mode::rhs_norm, mode::initial_resnorm, mode::absolute}) {
        gko::remove_complex<T> factor =
            (baseline == mode::absolute) ? 0.0 : r<T>::value;
        auto criterion = gko::stop::ImplicitResidualNorm<T>::build()
                             .with_reduction_factor(factor)
                             .with_baseline(baseline)
                             .on(this->exec)
                             ->generate(mtx, rhs, x.get(), nullptr);
        constexpr gko::uint8 RelativeStoppingId{1};
        bool one_changed{};
        gko::array<gko::stopping_status> stop_status(this->ref, 1);
        stop_status.get_data()[0].reset();
        stop_status.set_executor(this->exec);

        EXPECT_TRUE(
            criterion->update()
                .implicit_sq_residual_norm(implicit_sq_res_norm)
                .check(RelativeStoppingId, true, &stop_status, &one_changed));
        stop_status.set_executor(this->ref);
        EXPECT_TRUE(stop_status.get_data()[0].has_converged());
        EXPECT_TRUE(one_changed);
    }
}

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
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
    auto criterion =
        this->factory->generate(nullptr, rhs, nullptr, initial_res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 1);
    stop_status.get_data()[0].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0,
          std::pow(r<TypeParam>::value * 1.1 * read(rhs_norm, 0, 0), 2));
    ASSERT_FALSE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    ASSERT_FALSE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 0,
          std::pow(r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 0), 2));
    ASSERT_TRUE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
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
    gko::as<Mtx>(rhs)->compute_norm2(rhs_norm);
    auto criterion = this->factory->generate(nullptr, rhs, nullptr, res.get());
    bool one_changed{};
    constexpr gko::uint8 RelativeStoppingId{1};
    gko::array<gko::stopping_status> stop_status(this->ref, 2);
    stop_status.get_data()[0].reset();
    stop_status.get_data()[1].reset();
    stop_status.set_executor(this->exec);

    ASSERT_FALSE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0,
          std::pow(r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 0), 2));
    ASSERT_FALSE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 1,
          std::pow(r<TypeParam>::value * 0.9 * read(rhs_norm, 0, 1), 2));
    ASSERT_TRUE(criterion->update().implicit_sq_residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
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

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 1.1);
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_FALSE(stop_status.get_data()[0].has_converged());
    ASSERT_FALSE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9);
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
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

    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));

    write(res_norm, 0, 0, r<TypeParam>::value * 0.9);
    ASSERT_FALSE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[0].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);

    write(res_norm, 0, 1, r<TypeParam>::value * 0.9);
    ASSERT_TRUE(criterion->update().residual_norm(res_norm).check(
        RelativeStoppingId, true, &stop_status, &one_changed));
    stop_status.set_executor(this->ref);
    ASSERT_TRUE(stop_status.get_data()[1].has_converged());
    ASSERT_TRUE(one_changed);
    stop_status.set_executor(this->exec);
}
