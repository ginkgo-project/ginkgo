// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/chebyshev.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class Chebyshev : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;

    Chebyshev() : rand_engine(30) {}

    std::unique_ptr<Mtx> gen_mtx(gko::size_type num_rows,
                                 gko::size_type num_cols, gko::size_type stride)
    {
        auto tmp_mtx = gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
        auto result = Mtx::create(ref, gko::dim<2>{num_rows, num_cols}, stride);
        result->copy_from(tmp_mtx);
        return result;
    }

    std::default_random_engine rand_engine;
};


TEST_F(Chebyshev, ApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50, 52);
    auto x = gen_mtx(50, 3, 8);
    auto b = gen_mtx(50, 3, 5);
    auto d_mtx = clone(exec, mtx);
    auto d_x = clone(exec, x);
    auto d_b = clone(exec, b);
    // Forget about accuracy - Chebyshev is not going to converge for a random
    // matrix, just check that a couple of iterations gives the same result on
    // both executors
    auto chebyshev_factory =
        gko::solver::Chebyshev<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(ref))
            .on(ref);
    auto d_chebyshev_factory =
        gko::solver::Chebyshev<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .on(exec);
    auto solver = chebyshev_factory->generate(std::move(mtx));
    auto d_solver = d_chebyshev_factory->generate(std::move(d_mtx));

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(Chebyshev, ApplyWithIterativeInnerSolverIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50, 54);
    auto x = gen_mtx(50, 3, 6);
    auto b = gen_mtx(50, 3, 10);
    auto d_mtx = clone(exec, mtx);
    auto d_x = clone(exec, x);
    auto d_b = clone(exec, b);
    auto chebyshev_factory =
        gko::solver::Chebyshev<value_type>::build()
            .with_preconditioner(
                gko::solver::Gmres<value_type>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u).on(
                            ref))
                    .on(ref))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(ref))
            .on(ref);
    auto d_chebyshev_factory =
        gko::solver::Chebyshev<value_type>::build()
            .with_preconditioner(
                gko::solver::Gmres<value_type>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u).on(
                            exec))
                    .on(exec))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .on(exec);
    auto solver = chebyshev_factory->generate(std::move(mtx));
    auto d_solver = d_chebyshev_factory->generate(std::move(d_mtx));

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    // Note: r<value_type>::value * 300 instead of r<value_type>::value, as
    // the difference in the inner gmres iteration gets amplified by the
    // difference in IR.
    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value * 300);
}


TEST_F(Chebyshev, ApplyWithGivenInitialGuessModeIsEquivalentToRef)
{
    using initial_guess_mode = gko::solver::initial_guess_mode;
    auto mtx = gko::share(gen_mtx(50, 50, 52));
    auto b = gen_mtx(50, 3, 7);
    auto d_mtx = gko::share(clone(exec, mtx));
    auto d_b = clone(exec, b);
    for (auto guess : {initial_guess_mode::provided, initial_guess_mode::rhs,
                       initial_guess_mode::zero}) {
        auto x = gen_mtx(50, 3, 4);
        auto d_x = clone(exec, x);
        auto chebyshev_factory =
            gko::solver::Chebyshev<value_type>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(2u).on(ref))
                .with_default_initial_guess(guess)
                .on(ref);
        auto d_chebyshev_factory =
            gko::solver::Chebyshev<value_type>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(2u).on(exec))
                .with_default_initial_guess(guess)
                .on(exec);
        auto solver = chebyshev_factory->generate(mtx);
        auto d_solver = d_chebyshev_factory->generate(d_mtx);

        solver->apply(b, x);
        d_solver->apply(d_b, d_x);

        GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
    }
}
