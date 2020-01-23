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
#include <ginkgo/core/stop/residual_norm_reduction.hpp>
#include <ginkgo/core/stop/time.hpp>


#include <core/test/utils.hpp>


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
          l_factory(
              l_solver_type::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(10u).on(
                          exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(r<T>::value)
                          .on(exec))
                  .on(exec)),
          u_factory(
              u_solver_type::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(10u).on(
                          exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(r<T>::value)
                          .on(exec))
                  .on(exec)),
          ilu_pre_factory(ilu_prec_type::build()
                              .with_l_solver_factory(l_factory)
                              .with_u_solver_factory(u_factory)
                              .on(exec)),
          ilu_rev_pre_factory(ilu_rev_prec_type::build()
                                  .with_l_solver_factory(l_factory)
                                  .with_u_solver_factory(u_factory)
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


TYPED_TEST_CASE(Ilu, gko::test::ValueTypes);


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
        ->convert_to(converted_l_factor.get());
    gko::as<gko::ConvertibleTo<Mtx>>(internal_u_factor.get())
        ->convert_to(converted_u_factor.get());
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
    auto u_l_composition = Composition::create(this->u_factor, this->l_factor);
    auto copied = ilu_prec_type::build()
                      .on(this->exec)
                      ->generate(gko::share(u_l_composition));

    copied->copy_from(ilu.get());

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
    auto u_l_composition = Composition::create(this->u_factor, this->l_factor);
    auto moved = ilu_prec_type::build()
                     .on(this->exec)
                     ->generate(gko::share(u_l_composition));

    moved->copy_from(std::move(ilu));

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


TYPED_TEST(Ilu, SolvesCustomTypeDefaultFactorySingleRhs)
{
    using ilu_prec_type = typename TestFixture::ilu_prec_type;
    using Mtx = typename TestFixture::Mtx;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());

    auto preconditioner =
        ilu_prec_type::build().on(this->exec)->generate(this->mtx);
    preconditioner->apply(b.get(), x.get());

    // Since it uses Bicgstab with default parmeters, the result will not be
    // accurate
    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-1);
}


TYPED_TEST(Ilu, SolvesSingleRhsWithParIlu)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());
    auto par_ilu_fact =
        gko::factorization::ParIlu<value_type>::build().on(this->exec);
    auto par_ilu = par_ilu_fact->generate(this->mtx);

    auto preconditioner = this->ilu_pre_factory->generate(gko::share(par_ilu));
    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}),
                        r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesSingleRhsWithComposition)
{
    using Mtx = typename TestFixture::Mtx;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());

    auto preconditioner =
        this->ilu_pre_factory->generate(this->l_u_composition);
    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}),
                        r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesSingleRhsWithMtx)
{
    using Mtx = typename TestFixture::Mtx;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());

    auto preconditioner = this->ilu_pre_factory->generate(this->mtx);
    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}),
                        r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesReverseSingleRhs)
{
    using Mtx = typename TestFixture::Mtx;
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x->copy_from(b.get());
    auto preconditioner =
        this->ilu_rev_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.625, 0.875, 1.75}),
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

    preconditioner->apply(alpha_linop.get(), b.get(), beta_linop.get(),
                          x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-7.0, 2.0, -1.0}), r<TypeParam>::value);
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

    preconditioner->apply(alpha_linop.get(), b.get(), beta_linop.get(),
                          x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-7.75, 6.25, 1.5}),
                        r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesMultipleRhs)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    const auto b = gko::initialize<Mtx>(
        {I<T>{1.0, 8.0}, I<T>{3.0, 21.0}, I<T>{6.0, 24.0}}, this->exec);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 2});
    x->copy_from(b.get());
    auto preconditioner =
        this->ilu_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({{-0.125, 2.0}, {0.25, 3.0}, {1.0, 1.0}}),
                        r<TypeParam>::value * 1e+1);
}


TYPED_TEST(Ilu, SolvesDifferentNumberOfRhs)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    const auto b1 = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, this->exec);
    auto x11 = Mtx::create(this->exec, gko::dim<2>{3, 1});
    auto x12 = Mtx::create(this->exec, gko::dim<2>{3, 1});
    x11->copy_from(b1.get());
    x12->copy_from(b1.get());
    const auto b2 = gko::initialize<Mtx>(
        {I<T>{1.0, 8.0}, I<T>{3.0, 21.0}, I<T>{6.0, 24.0}}, this->exec);
    auto x2 = Mtx::create(this->exec, gko::dim<2>{3, 2});
    x2->copy_from(b2.get());
    auto preconditioner =
        this->ilu_pre_factory->generate(this->l_u_composition);

    preconditioner->apply(b1.get(), x11.get());
    preconditioner->apply(b2.get(), x2.get());
    preconditioner->apply(b1.get(), x12.get());

    GKO_ASSERT_MTX_NEAR(x11.get(), l({-3.0, 2.0, 1.0}),
                        r<TypeParam>::value * 1e+1);
    GKO_ASSERT_MTX_NEAR(x2.get(), l({{-0.125, 2.0}, {0.25, 3.0}, {1.0, 1.0}}),
                        r<TypeParam>::value * 1e+1);
    GKO_ASSERT_MTX_NEAR(x12.get(), x11.get(), r<TypeParam>::value * 1e+1);
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
    x->copy_from(b.get());

    auto preconditioner =
        default_ilu_prec_type::build().on(this->exec)->generate(this->mtx);
    preconditioner->apply(b.get(), x.get());

    // Since it uses TRS per default, the result should be accurate
    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(DefaultIlu, CanBeUsedAsPreconditioner)
{
    auto solver =
        gko::solver::Bicgstab<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(this->exec))
            .with_preconditioner(default_ilu_prec_type::build().on(this->exec))
            .on(this->exec)
            ->generate(this->mtx);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    x->copy_from(b.get());

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(DefaultIlu, CanBeUsedAsGeneratedPreconditioner)
{
    std::shared_ptr<default_ilu_prec_type> precond =
        default_ilu_prec_type::build().on(this->exec)->generate(this->mtx);
    auto solver =
        gko::solver::Bicgstab<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(this->exec))
            .with_generated_preconditioner(precond)
            .on(this->exec)
            ->generate(this->mtx);
    auto x = Mtx::create(this->exec, gko::dim<2>{3, 1});
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, this->exec);
    x->copy_from(b.get());

    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


}  // namespace
