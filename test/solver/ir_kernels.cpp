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

#include "core/solver/ir_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class Ir : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;

    Ir() : rand_engine(30) {}

    std::unique_ptr<Mtx> gen_mtx(gko::size_type num_rows,
                                 gko::size_type num_cols, gko::size_type stride)
    {
        auto tmp_mtx = gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
        auto result = Mtx::create(ref, gko::dim<2>{num_rows, num_cols}, stride);
        result->copy_from(tmp_mtx.get());
        return result;
    }

    std::default_random_engine rand_engine;
};


TEST_F(Ir, InitializeIsEquivalentToRef)
{
    auto stop_status = gko::array<gko::stopping_status>(ref, 43);
    for (size_t i = 0; i < stop_status.get_num_elems(); ++i) {
        stop_status.get_data()[i].reset();
    }
    auto d_stop_status = gko::array<gko::stopping_status>(exec, stop_status);

    gko::kernels::reference::ir::initialize(ref, &stop_status);
    gko::kernels::EXEC_NAMESPACE::ir::initialize(exec, &d_stop_status);

    auto tmp = gko::array<gko::stopping_status>(ref, d_stop_status);
    for (int i = 0; i < stop_status.get_num_elems(); ++i) {
        ASSERT_EQ(stop_status.get_const_data()[i], tmp.get_const_data()[i]);
    }
}


TEST_F(Ir, ApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50, 52);
    auto x = gen_mtx(50, 3, 8);
    auto b = gen_mtx(50, 3, 5);
    auto d_mtx = clone(exec, mtx);
    auto d_x = clone(exec, x);
    auto d_b = clone(exec, b);
    // Forget about accuracy - Richardson is not going to converge for a random
    // matrix, just check that a couple of iterations gives the same result on
    // both executors
    auto ir_factory =
        gko::solver::Ir<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(ref))
            .on(ref);
    auto d_ir_factory =
        gko::solver::Ir<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .on(exec);
    auto solver = ir_factory->generate(std::move(mtx));
    auto d_solver = d_ir_factory->generate(std::move(d_mtx));

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(Ir, ApplyWithIterativeInnerSolverIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50, 54);
    auto x = gen_mtx(50, 3, 6);
    auto b = gen_mtx(50, 3, 10);
    auto d_mtx = clone(exec, mtx);
    auto d_x = clone(exec, x);
    auto d_b = clone(exec, b);

    auto ir_factory =
        gko::solver::Ir<value_type>::build()
            .with_solver(
                gko::solver::Gmres<value_type>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u).on(
                            ref))
                    .on(ref))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(ref))
            .on(ref);
    auto d_ir_factory =
        gko::solver::Ir<value_type>::build()
            .with_solver(
                gko::solver::Gmres<value_type>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u).on(
                            exec))
                    .on(exec))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .on(exec);
    auto solver = ir_factory->generate(std::move(mtx));
    auto d_solver = d_ir_factory->generate(std::move(d_mtx));

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    // Note: r<value_type>::value * 150 instead of r<value_type>::value, as
    // the difference in the inner gmres iteration gets amplified by the
    // difference in IR.
    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value * 300);
}


TEST_F(Ir, RichardsonApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50, 54);
    auto x = gen_mtx(50, 3, 4);
    auto b = gen_mtx(50, 3, 3);
    auto d_mtx = clone(exec, mtx);
    auto d_x = clone(exec, x);
    auto d_b = clone(exec, b);
    // Forget about accuracy - Richardson is not going to converge for a random
    // matrix, just check that a couple of iterations gives the same result on
    // both executors
    auto ir_factory =
        gko::solver::Ir<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(ref))
            .with_relaxation_factor(value_type{0.9})
            .on(ref);
    auto d_ir_factory =
        gko::solver::Ir<value_type>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .with_relaxation_factor(value_type{0.9})
            .on(exec);
    auto solver = ir_factory->generate(std::move(mtx));
    auto d_solver = d_ir_factory->generate(std::move(d_mtx));

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(Ir, RichardsonApplyWithIterativeInnerSolverIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50, 52);
    auto x = gen_mtx(50, 3, 4);
    auto b = gen_mtx(50, 3, 7);
    auto d_mtx = clone(exec, mtx);
    auto d_x = clone(exec, x);
    auto d_b = clone(exec, b);
    auto ir_factory =
        gko::solver::Ir<value_type>::build()
            .with_solver(
                gko::solver::Gmres<value_type>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u).on(
                            ref))
                    .on(ref))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(ref))
            .with_relaxation_factor(value_type{0.9})
            .on(ref);
    auto d_ir_factory =
        gko::solver::Ir<value_type>::build()
            .with_solver(
                gko::solver::Gmres<value_type>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u).on(
                            exec))
                    .on(exec))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .with_relaxation_factor(value_type{0.9})
            .on(exec);
    auto solver = ir_factory->generate(std::move(mtx));
    auto d_solver = d_ir_factory->generate(std::move(d_mtx));

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    // Note: r<value_type>::value * 1e2 instead of r<value_type>::value, as
    // the difference in the inner gmres iteration gets amplified by the
    // difference in IR.
    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value * 200);
}


TEST_F(Ir, ApplyWithGivenInitialGuessModeIsEquivalentToRef)
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
        auto ir_factory =
            gko::solver::Ir<value_type>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(2u).on(ref))
                .with_default_initial_guess(guess)
                .on(ref);
        auto d_ir_factory =
            gko::solver::Ir<value_type>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(2u).on(exec))
                .with_default_initial_guess(guess)
                .on(exec);
        auto solver = ir_factory->generate(mtx);
        auto d_solver = d_ir_factory->generate(d_mtx);

        solver->apply(b, x);
        d_solver->apply(d_b, d_x);

        GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
    }
}
