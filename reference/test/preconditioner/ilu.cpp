// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/ilu.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Ilu : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using l_solver_type = gko::solver::Bicgstab<value_type>;
    using u_solver_type = gko::solver::Bicgstab<value_type>;
    using default_ilu_prec_type = gko::preconditioner::Ilu<>;
    using ilu_prec_type =
        gko::preconditioner::Ilu<l_solver_type, u_solver_type, false>;
    using ilu_rev_prec_type =
        gko::preconditioner::Ilu<l_solver_type, u_solver_type, true>;
    using Composition = gko::Composition<value_type>;

    Ilu()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>({{2., 1., 1.}, {2., 5., 2.}, {2., 5., 5.}},
                                   exec)),
          l_factor(gko::initialize<Mtx>(
              {{1., 0., 0.}, {1., 1., 0.}, {1., 1., 1.}}, exec)),
          u_factor(gko::initialize<Mtx>(
              {{2., 1., 1.}, {0., 4., 1.}, {0., 0., 3.}}, exec)),
          l_u_composition(Composition::create(l_factor, u_factor)),
          l_factory(l_solver_type::build()
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(10u),
                            gko::stop::Time::build().with_time_limit(
                                std::chrono::seconds(6)),
                            gko::stop::ResidualNorm<value_type>::build()
                                .with_reduction_factor(r<T>::value))
                        .on(exec)),
          u_factory(u_solver_type::build()
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(10u),
                            gko::stop::Time::build().with_time_limit(
                                std::chrono::seconds(6)),
                            gko::stop::ResidualNorm<value_type>::build()
                                .with_reduction_factor(r<T>::value))
                        .on(exec)),
          ilu_pre_factory(ilu_prec_type::build()
                              .with_l_solver(l_factory)
                              .with_u_solver(u_factory)
                              .on(exec)),
          ilu_rev_pre_factory(ilu_rev_prec_type::build()
                                  .with_l_solver(l_factory)
                                  .with_u_solver(u_factory)
                                  .on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> l_factor;
    std::shared_ptr<Mtx> u_factor;
    std::shared_ptr<Composition> l_u_composition;
    std::shared_ptr<typename l_solver_type::Factory> l_factory;
    std::shared_ptr<typename u_solver_type::Factory> u_factory;
    std::shared_ptr<typename ilu_prec_type::Factory> ilu_pre_factory;
    std::shared_ptr<typename ilu_rev_prec_type::Factory> ilu_rev_pre_factory;
};

TYPED_TEST_SUITE(Ilu, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Ilu, BuildsDefaultWithoutThrowing)
{
    using ilu_prec_type = typename TestFixture::ilu_prec_type;
    auto ilu_pre_default_factory = ilu_prec_type::build().on(this->exec);

    ASSERT_NO_THROW(ilu_pre_default_factory->generate(this->l_u_composition));
}


TYPED_TEST(Ilu, BuildsCustomWithoutThrowing)
{
    ASSERT_NO_THROW(this->ilu_pre_factory->generate(this->l_u_composition));
}


TYPED_TEST(Ilu, BuildsCustomWithoutThrowing2)
{
    ASSERT_NO_THROW(this->ilu_pre_factory->generate(this->mtx));
}


TYPED_TEST(Ilu, ThrowOnWrongCompositionInput)
{
    using Composition = typename TestFixture::Composition;
    std::shared_ptr<Composition> composition =
        Composition::create(this->l_factor);

    ASSERT_THROW(this->ilu_pre_factory->generate(composition),
                 gko::NotSupported);
}


TYPED_TEST(Ilu, ThrowOnWrongCompositionInput2)
{
    using Composition = typename TestFixture::Composition;
    std::shared_ptr<Composition> composition =
        Composition::create(this->l_factor, this->u_factor, this->l_factor);

    ASSERT_THROW(this->ilu_pre_factory->generate(composition),
                 gko::NotSupported);
}


TYPED_TEST(Ilu, SetsCorrectMatrices)
{
    using Mtx = typename TestFixture::Mtx;
    auto ilu = this->ilu_pre_factory->generate(this->l_u_composition);
    auto internal_l_factor = ilu->get_l_solver()->get_system_matrix();
    auto internal_u_factor = ilu->get_u_solver()->get_system_matrix();

    // These convert steps are required since `get_system_matrix` usually
    // just returns `LinOp`, which `GKO_ASSERT_MTX_NEAR` can not use properly
    std::unique_ptr<Mtx> converted_l_factor{Mtx::create(this->exec)};
    std::unique_ptr<Mtx> converted_u_factor{Mtx::create(this->exec)};
    gko::as<gko::ConvertibleTo<Mtx>>(internal_l_factor.get())
        ->convert_to(converted_l_factor);
    gko::as<gko::ConvertibleTo<Mtx>>(internal_u_factor.get())
        ->convert_to(converted_u_factor);
    GKO_ASSERT_MTX_NEAR(converted_l_factor, this->l_factor, 0);
    GKO_ASSERT_MTX_NEAR(converted_u_factor, this->u_factor, 0);
}


TYPED_TEST(Ilu, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using ilu_prec_type = typename TestFixture::ilu_prec_type;
    using Composition = typename TestFixture::Composition;
    auto ilu = this->ilu_pre_factory->generate(this->l_u_composition);
    auto before_l_solver = ilu->get_l_solver();
    auto before_u_solver = ilu->get_u_solver();
    // The switch up of matrices is intentional, to make sure they are distinct!
    auto u_l_composition =
        gko::share(Composition::create(this->u_factor, this->l_factor));
    auto copied =
        ilu_prec_type::build().on(this->exec)->generate(u_l_composition);

    copied->copy_from(ilu);

    ASSERT_EQ(before_l_solver.get(), copied->get_l_solver().get());
    ASSERT_EQ(before_u_solver.get(), copied->get_u_solver().get());
}


TYPED_TEST(Ilu, CanBeMoved)
{
    using ilu_prec_type = typename TestFixture::ilu_prec_type;
    using Composition = typename TestFixture::Composition;
    auto ilu = this->ilu_pre_factory->generate(this->l_u_composition);
    auto before_l_solver = ilu->get_l_solver();
    auto before_u_solver = ilu->get_u_solver();
    // The switch up of matrices is intentional, to make sure they are distinct!
    auto u_l_composition =
        gko::share(Composition::create(this->u_factor, this->l_factor));
    auto moved =
        ilu_prec_type::build().on(this->exec)->generate(u_l_composition);

    moved->move_from(ilu);

    ASSERT_EQ(before_l_solver.get(), moved->get_l_solver().get());
    ASSERT_EQ(before_u_solver.get(), moved->get_u_solver().get());
}


TYPED_TEST(Ilu, CanBeCloned)
{
    auto ilu = this->ilu_pre_factory->generate(this->l_u_composition);
    auto before_l_solver = ilu->get_l_solver();
    auto before_u_solver = ilu->get_u_solver();

    auto clone = ilu->clone();

    ASSERT_EQ(before_l_solver.get(), clone->get_l_solver().get());
    ASSERT_EQ(before_u_solver.get(), clone->get_u_solver().get());
}


TYPED_TEST(Ilu, CanBeTransposed)
{
    using Ilu = typename TestFixture::ilu_prec_type;
    using Mtx = typename TestFixture::Mtx;
    auto ilu = this->ilu_pre_factory->generate(this->l_u_composition);
    auto l_ref = gko::as<Mtx>(ilu->get_l_solver()->get_system_matrix());
    auto u_ref = gko::as<Mtx>(ilu->get_u_solver()->get_system_matrix());

    auto transp = gko::as<Ilu>(ilu->transpose());

    auto l_transp = gko::as<Mtx>(
        gko::as<Mtx>(transp->get_u_solver()->get_system_matrix())->transpose());
    auto u_transp = gko::as<Mtx>(
        gko::as<Mtx>(transp->get_l_solver()->get_system_matrix())->transpose());
    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_transp);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_transp);
    GKO_ASSERT_MTX_NEAR(l_ref, l_transp, 0);
    GKO_ASSERT_MTX_NEAR(u_ref, u_transp, 0);
}


TYPED_TEST(Ilu, CanBeConjTransposed)
{
    using Ilu = typename TestFixture::ilu_prec_type;
    using Mtx = typename TestFixture::Mtx;
    auto ilu = this->ilu_pre_factory->generate(this->l_u_composition);
    auto l_ref = gko::as<Mtx>(ilu->get_l_solver()->get_system_matrix());
    auto u_ref = gko::as<Mtx>(ilu->get_u_solver()->get_system_matrix());

    auto transp = gko::as<Ilu>(ilu->conj_transpose());

    auto l_transp =
        gko::as<Mtx>(gko::as<Mtx>(transp->get_u_solver()->get_system_matrix())
                         ->conj_transpose());
    auto u_transp =
        gko::as<Mtx>(gko::as<Mtx>(transp->get_l_solver()->get_system_matrix())
                         ->conj_transpose());
    GKO_ASSERT_MTX_EQ_SPARSITY(l_ref, l_transp);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_ref, u_transp);
    GKO_ASSERT_MTX_NEAR(l_ref, l_transp, 0);
    GKO_ASSERT_MTX_NEAR(u_ref, u_transp, 0);
}


TYPED_TEST(Ilu, SolvesCustomTypeDefaultFactorySingleRhs)
{
    using ilu_prec_type = typename TestFixture::ilu_prec_type;
    using Mtx = typename TestFixture::Mtx;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b);

    auto preconditioner =
        ilu_prec_type::build().on(this->exec)->generate(this->mtx);
    preconditioner->apply(b, x);

    // Since it uses Bicgstab with default parameters, the result will not be
    // accurate
    GKO_ASSERT_MTX_NEAR(x, l({-0.125, 0.25, 1.0}), 1e-1);
}


TYPED_TEST(Ilu, SolvesSingleRhsWithParIlu)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b);
    auto par_ilu_fact =
        gko::factorization::ParIlu<value_type>::build().on(this->exec);
    auto par_ilu = gko::share(par_ilu_fact->generate(this->mtx));

    auto preconditioner = this->ilu_pre_factory->generate(par_ilu);
    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-0.125, 0.25, 1.0}), r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesSingleRhsWithComposition)
{
    using Mtx = typename TestFixture::Mtx;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b);

    auto preconditioner =
        this->ilu_pre_factory->generate(this->l_u_composition);
    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-0.125, 0.25, 1.0}), r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesSingleRhsWithMtx)
{
    using Mtx = typename TestFixture::Mtx;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b);

    auto preconditioner = this->ilu_pre_factory->generate(this->mtx);
    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-0.125, 0.25, 1.0}), r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesSingleRhsWithMixedMtx)
{
    using Mtx = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b);

    auto preconditioner = this->ilu_pre_factory->generate(this->mtx);
    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({-0.125, 0.25, 1.0}),
        (r_mixed<TypeParam, typename Mtx::value_type>()) * 1e+1);
}


TYPED_TEST(Ilu, SolvesSingleRhsWithComplexMtx)
{
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using T = typename Mtx::value_type;
    const auto b = gko::initialize<Mtx>(
        {T{1.0, 2.0}, T{3.0, 6.0}, T{6.0, 12.0}}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b);

    auto preconditioner = this->ilu_pre_factory->generate(this->mtx);
    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({T{-0.125, -0.25}, T{0.25, 0.5}, T{1.0, 2.0}}),
                        r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesSingleRhsWithMixedComplexMtx)
{
    using Mtx = gko::matrix::Dense<
        gko::to_complex<gko::next_precision<typename TestFixture::value_type>>>;
    using T = typename Mtx::value_type;
    const auto b = gko::initialize<Mtx>(
        {T{1.0, 2.0}, T{3.0, 6.0}, T{6.0, 12.0}}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b);

    auto preconditioner = this->ilu_pre_factory->generate(this->mtx);
    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x, l({T{-0.125, -0.25}, T{0.25, 0.5}, T{1.0, 2.0}}),
        (r_mixed<TypeParam, typename Mtx::value_type>()) * 1e+1);
}


TYPED_TEST(Ilu, SolvesReverseSingleRhs)
{
    using Mtx = typename TestFixture::Mtx;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b);
    auto preconditioner =
        this->ilu_rev_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-0.625, 0.875, 1.75}),
                        r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesAdvancedSingleRhs)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    const value_type alpha{2.0};
    const auto alpha_linop = gko::initialize<Mtx>({alpha}, this->exec);
    const value_type beta{-1};
    const auto beta_linop = gko::initialize<Mtx>({beta}, this->exec);
    const auto b = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0, 3.0}, this->exec);
    auto preconditioner =
        this->ilu_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(alpha_linop, b, beta_linop, x);

    GKO_ASSERT_MTX_NEAR(x, l({-7.0, 2.0, -1.0}), r<TypeParam>::value);
}


TYPED_TEST(Ilu, SolvesAdvancedSingleRhsMixed)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    const value_type alpha{2.0};
    const auto alpha_linop = gko::initialize<Mtx>({alpha}, this->exec);
    const value_type beta{-1};
    const auto beta_linop = gko::initialize<Mtx>({beta}, this->exec);
    const auto b = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0, 3.0}, this->exec);
    auto preconditioner =
        this->ilu_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(alpha_linop, b, beta_linop, x);

    GKO_ASSERT_MTX_NEAR(x, l({-7.0, 2.0, -1.0}),
                        (r_mixed<TypeParam, typename Mtx::value_type>()));
}


TYPED_TEST(Ilu, SolvesAdvancedSingleRhsComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Dense = typename TestFixture::Mtx;
    using DenseComplex = gko::to_complex<Dense>;
    const value_type alpha{2.0};
    const auto alpha_linop = gko::initialize<Dense>({alpha}, this->exec);
    const value_type beta{-1};
    const auto beta_linop = gko::initialize<Dense>({beta}, this->exec);
    const auto b = gko::initialize<DenseComplex>(
        {complex_type{-3.0, 6.0}, complex_type{6.0, -12.0},
         complex_type{9.0, -18.0}},
        this->exec);
    auto x = gko::initialize<DenseComplex>(
        {complex_type{1.0, -2.0}, complex_type{2.0, -4.0},
         complex_type{3.0, -6.0}},
        this->exec);
    auto preconditioner =
        this->ilu_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(alpha_linop, b, beta_linop, x);

    GKO_ASSERT_MTX_NEAR(x,
                        l({complex_type{-7.0, 14.0}, complex_type{2.0, -4.0},
                           complex_type{-1.0, 2.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Ilu, SolvesAdvancedSingleRhsMixedComplex)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using complex_type = gko::to_complex<value_type>;
    using MixedDense = gko::matrix::Dense<value_type>;
    using MixedDenseComplex = gko::to_complex<MixedDense>;
    const value_type alpha{2.0};
    const auto alpha_linop = gko::initialize<MixedDense>({alpha}, this->exec);
    const value_type beta{-1};
    const auto beta_linop = gko::initialize<MixedDense>({beta}, this->exec);
    const auto b = gko::initialize<MixedDenseComplex>(
        {complex_type{-3.0, 6.0}, complex_type{6.0, -12.0},
         complex_type{9.0, -18.0}},
        this->exec);
    auto x = gko::initialize<MixedDenseComplex>(
        {complex_type{1.0, -2.0}, complex_type{2.0, -4.0},
         complex_type{3.0, -6.0}},
        this->exec);
    auto preconditioner =
        this->ilu_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(alpha_linop, b, beta_linop, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({complex_type{-7.0, 14.0}, complex_type{2.0, -4.0},
           complex_type{-1.0, 2.0}}),
        (r_mixed<TypeParam, typename MixedDenseComplex::value_type>()));
}


TYPED_TEST(Ilu, SolvesAdvancedReverseSingleRhs)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    const value_type alpha{2.0};
    const auto alpha_linop = gko::initialize<Mtx>({alpha}, this->exec);
    const value_type beta{-1};
    const auto beta_linop = gko::initialize<Mtx>({beta}, this->exec);
    const auto b = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0, 3.0}, this->exec);
    auto preconditioner =
        this->ilu_rev_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(alpha_linop, b, beta_linop, x);

    GKO_ASSERT_MTX_NEAR(x, l({-7.75, 6.25, 1.5}), r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesMultipleRhs)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    const auto b = gko::initialize<Mtx>(
        {I<T>{1.0, 8.0}, I<T>{3.0, 21.0}, I<T>{6.0, 24.0}}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 2});
    x->copy_from(b);
    auto preconditioner =
        this->ilu_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{-0.125, 2.0}, {0.25, 3.0}, {1.0, 1.0}}),
                        r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesDifferentNumberOfRhs)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    const auto b1 = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, this->exec);
    auto x11 = Mtx::create(this->exec, gko::dim<2>{3, 1});
    auto x12 = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x11->copy_from(b1);
    x12->copy_from(b1);
    const auto b2 = gko::initialize<Mtx>(
        {I<T>{1.0, 8.0}, I<T>{3.0, 21.0}, I<T>{6.0, 24.0}}, this->exec);
    auto x2 = Mtx::create(this->exec, gko::dim<2>{3, 2});
    x2->copy_from(b2);
    auto preconditioner =
        this->ilu_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(b1, x11);
    preconditioner->apply(b2, x2);
    preconditioner->apply(b1, x12);

    GKO_ASSERT_MTX_NEAR(x11, l({-3.0, 2.0, 1.0}), r<TypeParam>::value * 1e+1);
    GKO_ASSERT_MTX_NEAR(x2, l({{-0.125, 2.0}, {0.25, 3.0}, {1.0, 1.0}}),
                        r<TypeParam>::value * 1e+1);
    GKO_ASSERT_MTX_NEAR(x12, x11, r<TypeParam>::value * 1e+1);
}


class DefaultIlu : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using default_ilu_prec_type = gko::preconditioner::Ilu<>;

    DefaultIlu()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>({{2., 1., 1.}, {2., 5., 2.}, {2., 5., 5.}},
                                   exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
};


TEST_F(DefaultIlu, SolvesDefaultSingleRhs)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b);

    auto preconditioner =
        default_ilu_prec_type::build().on(this->exec)->generate(this->mtx);
    preconditioner->apply(b, x);

    // Since it uses TRS per default, the result should be accurate
    GKO_ASSERT_MTX_NEAR(x, l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(DefaultIlu, CanBeUsedAsPreconditioner)
{
    auto solver =
        gko::solver::Bicgstab<>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .with_preconditioner(default_ilu_prec_type::build())
            .on(this->exec)
            ->generate(this->mtx);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    x->copy_from(b);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(DefaultIlu, CanBeUsedAsGeneratedPreconditioner)
{
    std::shared_ptr<default_ilu_prec_type> precond =
        default_ilu_prec_type::build().on(this->exec)->generate(this->mtx);
    auto solver =
        gko::solver::Bicgstab<>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .with_generated_preconditioner(precond)
            .on(this->exec)
            ->generate(this->mtx);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    x->copy_from(b);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({-0.125, 0.25, 1.0}), 1e-14);
}


}  // namespace
