/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/solver/async_richardson.hpp>


#include <typeinfo>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class AsyncRichardson : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::AsyncRichardson<value_type>;

    AsyncRichardson()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          async_richardson_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u).on(exec),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .on(exec)),
          solver(async_richardson_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<typename Solver::Factory> async_richardson_factory;
    std::unique_ptr<gko::LinOp> solver;

    static void assert_same_matrices(const Mtx* m1, const Mtx* m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        for (gko::size_type i = 0; i < m1->get_size()[0]; ++i) {
            for (gko::size_type j = 0; j < m2->get_size()[1]; ++j) {
                EXPECT_EQ(m1->at(i, j), m2->at(i, j));
            }
        }
    }
};

TYPED_TEST_SUITE(AsyncRichardson, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(AsyncRichardson, AsyncRichardsonFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->async_richardson_factory->get_executor(), this->exec);
}


TYPED_TEST(AsyncRichardson, AsyncRichardsonFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto cg_solver = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(cg_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(cg_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(AsyncRichardson, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy =
        this->async_richardson_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx*>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(AsyncRichardson, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy =
        this->async_richardson_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx*>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(AsyncRichardson, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = static_cast<Solver*>(clone.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx*>(clone_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(AsyncRichardson, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver*>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(AsyncRichardson, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(AsyncRichardson, CanSetCriteriaAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto async_richardson_factory =
        Solver::build().with_criteria(init_crit).on(this->exec);

    ASSERT_EQ((async_richardson_factory->get_parameters().criteria).back(),
              init_crit);

    auto solver = async_richardson_factory->generate(this->mtx);
    std::shared_ptr<gko::stop::CriterionFactory> new_crit =
        gko::stop::Iteration::build().with_max_iters(5u).on(this->exec);

    solver->set_stop_criterion_factory(new_crit);
    auto new_crit_fac = solver->get_stop_criterion_factory();
    auto niter =
        static_cast<const gko::stop::Iteration::Factory*>(new_crit_fac.get())
            ->get_parameters()
            .max_iters;

    ASSERT_EQ(niter, 5);
}


TYPED_TEST(AsyncRichardson, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->async_richardson_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(AsyncRichardson, DefaultRelaxationFactor)
{
    using value_type = typename TestFixture::value_type;
    const value_type relaxation_factor{0.5};

    auto richardson =
        gko::solver::AsyncRichardson<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(r<value_type>::value)
                    .on(this->exec))
            .on(this->exec)
            ->generate(this->mtx);

    ASSERT_EQ(richardson->get_parameters().relaxation_factor, value_type{1});
}


TYPED_TEST(AsyncRichardson, UseAsRichardson)
{
    using value_type = typename TestFixture::value_type;
    const value_type relaxation_factor{0.5};

    auto richardson =
        gko::solver::AsyncRichardson<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(r<value_type>::value)
                    .on(this->exec))
            .with_relaxation_factor(relaxation_factor)
            .on(this->exec)
            ->generate(this->mtx);

    ASSERT_EQ(richardson->get_parameters().relaxation_factor, value_type{0.5});
}


}  // namespace