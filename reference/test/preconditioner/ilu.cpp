/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include "core/test/utils/assertions.hpp"


namespace {


class Ilu : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Mtx = gko::matrix::Dense<value_type>;
    using l_solver_type = gko::solver::Bicgstab<value_type>;
    using u_solver_type = gko::solver::Bicgstab<value_type>;
    using default_ilu_prec_type = gko::preconditioner::Ilu<value_type>;
    using ilu_prec_type = gko::preconditioner::Ilu<value_type, l_solver_type,
                                                   u_solver_type, false>;
    using ilu_rev_prec_type =
        gko::preconditioner::Ilu<value_type, l_solver_type, u_solver_type,
                                 true>;

    Ilu()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>({{2, 1, 1}, {2, 5, 2}, {2, 5, 5}}, exec)),
          l_factor(
              gko::initialize<Mtx>({{1, 0, 0}, {1, 1, 0}, {1, 1, 1}}, exec)),
          u_factor(
              gko::initialize<Mtx>({{2, 1, 1}, {0, 4, 1}, {0, 0, 3}}, exec)),
          l_factory(
              l_solver_type::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(10u).on(
                          exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNormReduction<>::build()
                          .with_reduction_factor(1e-15)
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
                      gko::stop::ResidualNormReduction<>::build()
                          .with_reduction_factor(1e-15)
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
    std::shared_ptr<l_solver_type::Factory> l_factory;
    std::shared_ptr<u_solver_type::Factory> u_factory;
    std::shared_ptr<ilu_prec_type::Factory> ilu_pre_factory;
    std::shared_ptr<ilu_rev_prec_type::Factory> ilu_rev_pre_factory;
};


TEST_F(Ilu, KnowsItsExecutor)
{
    auto ilu_factory = ilu_prec_type::build().on(exec);

    ASSERT_EQ(ilu_factory->get_executor(), exec);
}


TEST_F(Ilu, CanSetLSolverFactory)
{
    auto ilu_factory =
        ilu_prec_type::build().with_l_solver_factory(l_factory).on(exec);

    ASSERT_EQ(ilu_factory->get_parameters().l_solver_factory, l_factory);
}


TEST_F(Ilu, CanSetUSolverFactory)
{
    auto ilu_factory =
        ilu_prec_type::build().with_u_solver_factory(u_factory).on(exec);

    ASSERT_EQ(ilu_factory->get_parameters().u_solver_factory, u_factory);
}


TEST_F(Ilu, BuildsDefaultWithoutThrowing)
{
    auto ilu_pre_default_factory = ilu_prec_type::build().on(exec);
    auto par_ilu_fact =
        gko::factorization::ParIlu<value_type>::build().on(exec);
    auto par_ilu = par_ilu_fact->generate(mtx);

    ASSERT_NO_THROW(ilu_pre_default_factory->generate(gko::share(par_ilu)));
}


TEST_F(Ilu, BuildsCustomWithoutThrowing)
{
    auto par_ilu_fact =
        gko::factorization::ParIlu<value_type>::build().on(exec);
    auto par_ilu = par_ilu_fact->generate(mtx);

    ASSERT_NO_THROW(ilu_pre_factory->generate(gko::share(par_ilu)));
}


TEST_F(Ilu, ThrowOnWrongInput)
{
    ASSERT_THROW(ilu_pre_factory->generate(l_factor), gko::NotSupported);
}


TEST_F(Ilu, SolvesDefaultSingleRhsWithParIlu)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    auto par_ilu_fact =
        gko::factorization::ParIlu<value_type>::build().on(exec);
    auto par_ilu = par_ilu_fact->generate(mtx);

    auto preconditioner =
        default_ilu_prec_type::build().on(exec)->generate(gko::share(par_ilu));
    preconditioner->apply(b.get(), x.get());

    // Since it is a preconditioner, the default solve is not accurate
    // Applies only to BiCGSTAB   TODO: Replace with 1e-14 when TRS is default
    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 5e-1);
}


TEST_F(Ilu, SolvesSingleRhsWithParIlu)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    auto par_ilu_fact =
        gko::factorization::ParIlu<value_type>::build().on(exec);
    auto par_ilu = par_ilu_fact->generate(mtx);

    auto preconditioner = ilu_pre_factory->generate(gko::share(par_ilu));
    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(Ilu, SolvesSingleRhsWithComposition)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    auto composition = gko::Composition<value_type>::create(l_factor, u_factor);

    auto preconditioner = ilu_pre_factory->generate(gko::share(composition));
    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(Ilu, SolvesSingleRhsWithLU)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});

    auto preconditioner = ilu_pre_factory->generate(l_factor, u_factor);
    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.125, 0.25, 1.0}), 1e-14);
}


TEST_F(Ilu, SolvesReverseSingleRhs)
{
    const auto b = gko::initialize<Mtx>({1.0, 3.0, 6.0}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 1});
    auto preconditioner = ilu_rev_pre_factory->generate(l_factor, u_factor);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-0.625, 0.875, 1.75}), 1e-14);
}


TEST_F(Ilu, SolvesAdvancedSingleRhs)
{
    const value_type alpha{2.0};
    const auto alpha_linop = gko::initialize<Mtx>({alpha}, exec);
    const value_type beta{-1};
    const auto beta_linop = gko::initialize<Mtx>({beta}, exec);
    const auto b = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0, 3.0}, exec);
    auto preconditioner = ilu_pre_factory->generate(l_factor, u_factor);

    preconditioner->apply(alpha_linop.get(), b.get(), beta_linop.get(),
                          x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-7.0, 2.0, -1.0}), 1e-14);
}


TEST_F(Ilu, SolvesAdvancedReverseSingleRhs)
{
    const value_type alpha{2.0};
    const auto alpha_linop = gko::initialize<Mtx>({alpha}, exec);
    const value_type beta{-1};
    const auto beta_linop = gko::initialize<Mtx>({beta}, exec);
    const auto b = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0, 3.0}, exec);
    auto preconditioner = ilu_rev_pre_factory->generate(l_factor, u_factor);

    preconditioner->apply(alpha_linop.get(), b.get(), beta_linop.get(),
                          x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({-7.75, 6.25, 1.5}), 1e-14);
}


TEST_F(Ilu, SolvesMultipleRhs)
{
    const auto b =
        gko::initialize<Mtx>({{1.0, 8.0}, {3.0, 21.0}, {6.0, 24.0}}, exec);
    auto x = Mtx::create(exec, gko::dim<2>{3, 2});
    auto preconditioner = ilu_pre_factory->generate(l_factor, u_factor);

    preconditioner->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x.get(), l({{-0.125, 2.0}, {0.25, 3.0}, {1.0, 1.0}}),
                        1e-14);
}


TEST_F(Ilu, SolvesDifferentNumberOfRhs)
{
    const auto b1 = gko::initialize<Mtx>({-3.0, 6.0, 9.0}, exec);
    auto x11 = Mtx::create(exec, gko::dim<2>{3, 1});
    auto x12 = Mtx::create(exec, gko::dim<2>{3, 1});
    const auto b2 =
        gko::initialize<Mtx>({{1.0, 8.0}, {3.0, 21.0}, {6.0, 24.0}}, exec);
    auto x2 = Mtx::create(exec, gko::dim<2>{3, 2});
    auto preconditioner = ilu_pre_factory->generate(l_factor, u_factor);

    preconditioner->apply(b1.get(), x11.get());
    preconditioner->apply(b2.get(), x2.get());
    preconditioner->apply(b1.get(), x12.get());

    GKO_ASSERT_MTX_NEAR(x11.get(), l({-3.0, 2.0, 1.0}), 1e-14);
    GKO_ASSERT_MTX_NEAR(x2.get(), l({{-0.125, 2.0}, {0.25, 3.0}, {1.0, 1.0}}),
                        1e-14);
    GKO_ASSERT_MTX_NEAR(x12.get(), x11.get(), 1e-14);
}


}  // namespace
