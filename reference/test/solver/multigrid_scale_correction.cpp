// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"


namespace {


// Exercises the multigrid `scale_correction` feature (OpenFOAM-style Rayleigh
// scaling of the pre-smooth and coarse corrections, implemented in
// MultigridState::run_cycle). These are behavioral tests on the reference
// executor: a real Pgm V-cycle multigrid is run as a stand-alone solver on an
// SPD system, so the whole scale-correction path (compute_dot -> device-side
// guarded reciprocal `safe_inv_scale` -> scaled correction) is executed.
class MultigridScaleCorrection : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using Pgm = gko::multigrid::Pgm<value_type, index_type>;
    using Ir = gko::solver::Ir<value_type>;
    using Cg = gko::solver::Cg<value_type>;
    using Jacobi = gko::preconditioner::Jacobi<value_type, index_type>;
    using Mg = gko::solver::Multigrid;

    MultigridScaleCorrection()
        : exec(gko::ReferenceExecutor::create()),
          n(300),
          mtx(gko::share(Csr::create(exec))),
          x_exact(Vec::create(exec, gko::dim<2>(n, 1))),
          b(Vec::create(exec, gko::dim<2>(n, 1)))
    {
        // 1D Laplacian tridiag(-1, 2, -1): symmetric positive definite, and
        // > min_coarse_rows so Pgm produces several levels (scale correction
        // is only active on levels above the coarsest).
        gko::matrix_data<value_type, index_type> data(gko::dim<2>(n, n));
        for (index_type i = 0; i < n; i++) {
            if (i > 0) {
                data.nonzeros.emplace_back(i, i - 1, value_type{-1.0});
            }
            data.nonzeros.emplace_back(i, i, value_type{2.0});
            if (i < n - 1) {
                data.nonzeros.emplace_back(i, i + 1, value_type{-1.0});
            }
        }
        mtx->read(data);

        x_exact->fill(value_type{1.0});
        mtx->apply(x_exact, b);  // b = A * x_exact
    }

    // Smoother / coarsest-solver factories shared by every multigrid we build.
    std::shared_ptr<Ir::Factory> smoother_factory()
    {
        return Ir::build()
            .with_solver(Jacobi::build().with_max_block_size(1u))
            .with_relaxation_factor(value_type{2.0 / 3.0})
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .on(exec);
    }

    std::shared_ptr<Cg::Factory> coarsest_factory()
    {
        return Cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(50u),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(value_type{1e-4}))
            .on(exec);
    }

    // The three scale-correction configurations exposed by the feature.
    enum class sc_mode { none, post_only, pre_and_post };

    // A single-V-cycle multigrid factory used as a preconditioner (the way the
    // OpenFOAM-style scale-corrected multigrid is deployed in production: it is
    // a *nonlinear* correction and is driven by an outer Krylov solver rather
    // than iterated stand-alone).
    std::shared_ptr<Mg::Factory> mg_precond_factory(sc_mode mode)
    {
        return Mg::build()
            .with_mg_level(Pgm::build().with_deterministic(true))
            .with_pre_smoother(smoother_factory())
            .with_post_smoother(smoother_factory())
            .with_coarsest_solver(coarsest_factory())
            .with_min_coarse_rows(16u)
            .with_scale_correction(mode != sc_mode::none)
            .with_scale_correction_pre_smooth(mode == sc_mode::pre_and_post)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);
    }

    // CG preconditioned by a (possibly scale-corrected) multigrid V-cycle.
    std::unique_ptr<gko::LinOp> build_cg_mg(sc_mode mode)
    {
        return Cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(300u),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_baseline(gko::stop::mode::rhs_norm)
                    .with_reduction_factor(value_type{1e-11}))
            .with_preconditioner(mg_precond_factory(mode))
            .on(exec)
            ->generate(mtx);
    }

    // A stand-alone V-cycle multigrid solver running a fixed number of cycles
    // (Iteration criterion only) -- used by the zero-rhs guard test, which
    // needs the scale-correction path to execute even though the residual is
    // exactly zero.
    std::unique_ptr<gko::LinOp> build_standalone_mg(bool scale_correction,
                                                    unsigned num_cycles)
    {
        return Mg::build()
            .with_mg_level(Pgm::build().with_deterministic(true))
            .with_pre_smoother(smoother_factory())
            .with_post_smoother(smoother_factory())
            .with_coarsest_solver(coarsest_factory())
            .with_min_coarse_rows(16u)
            .with_scale_correction(scale_correction)
            .with_default_initial_guess(
                gko::solver::initial_guess_mode::provided)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(num_cycles))
            .on(exec)
            ->generate(mtx);
    }

    // relative residual ||b - A x|| / ||b||
    value_type relative_residual(const Vec* x)
    {
        auto res = gko::clone(b);
        auto one = gko::initialize<Vec>({value_type{1.0}}, exec);
        auto neg_one = gko::initialize<Vec>({value_type{-1.0}}, exec);
        mtx->apply(neg_one, x, one, res);  // res = b - A x
        auto rnorm = Vec::create(exec, gko::dim<2>(1, 1));
        auto bnorm = Vec::create(exec, gko::dim<2>(1, 1));
        res->compute_norm2(rnorm);
        b->compute_norm2(bnorm);
        return rnorm->at(0, 0) / bnorm->at(0, 0);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    index_type n;
    std::shared_ptr<Csr> mtx;
    std::unique_ptr<Vec> x_exact;
    std::unique_ptr<Vec> b;
};


TEST_F(MultigridScaleCorrection, ScaleCorrectedVCyclePreconditionsCgToSolution)
{
    // A CG solve preconditioned by the scale-corrected multigrid V-cycle must
    // converge to the true solution of the SPD system. This drives every part
    // of the scale-correction code path (pre-smooth + coarse Rayleigh scaling
    // via the device-side safe_inv_scale) once per outer iteration.
    auto solver = build_cg_mg(sc_mode::pre_and_post);
    auto x = Vec::create(exec, gko::dim<2>(n, 1));
    x->fill(value_type{0.0});

    solver->apply(b, x);

    ASSERT_LT(relative_residual(x.get()), value_type{1e-9});
    GKO_ASSERT_MTX_NEAR(x, x_exact, value_type{1e-5});
}


TEST_F(MultigridScaleCorrection, PostOnlyScaleCorrectionPreconditionsCgToSolution)
{
    // The "post-only" mode (coarse-correction scaling, pre-smooth scaling
    // disabled) must also yield a correct CG solve. Exercises the do_pre_scale
    // == false path while do_scale == true.
    auto solver = build_cg_mg(sc_mode::post_only);
    auto x = Vec::create(exec, gko::dim<2>(n, 1));
    x->fill(value_type{0.0});

    solver->apply(b, x);

    ASSERT_LT(relative_residual(x.get()), value_type{1e-9});
    GKO_ASSERT_MTX_NEAR(x, x_exact, value_type{1e-5});
}


TEST_F(MultigridScaleCorrection, AllModesReachTheSameSolution)
{
    // Scale correction (in either mode) changes the multigrid convergence path
    // but not the fixed point: all three configurations drive CG to the same
    // solution of the SPD system.
    auto x_none = Vec::create(exec, gko::dim<2>(n, 1));
    auto x_post = Vec::create(exec, gko::dim<2>(n, 1));
    auto x_both = Vec::create(exec, gko::dim<2>(n, 1));
    x_none->fill(value_type{0.0});
    x_post->fill(value_type{0.0});
    x_both->fill(value_type{0.0});

    build_cg_mg(sc_mode::none)->apply(b, x_none);
    build_cg_mg(sc_mode::post_only)->apply(b, x_post);
    build_cg_mg(sc_mode::pre_and_post)->apply(b, x_both);

    GKO_ASSERT_MTX_NEAR(x_post, x_none, value_type{1e-5});
    GKO_ASSERT_MTX_NEAR(x_both, x_none, value_type{1e-5});
}


TEST_F(MultigridScaleCorrection, ZeroRhsStaysZeroWithoutNan)
{
    // Regression guard for the device-side reciprocal `safe_inv_scale`: with a
    // zero rhs and zero initial guess every correction delta is exactly zero,
    // so the Rayleigh denominator delta.A.delta is exactly zero. The guard must
    // yield a scale factor of 0 (num/(0 + eps) = 0), NOT a 0/0 NaN. Iteration-
    // only criteria force the V-cycles to actually run.
    auto solver = build_standalone_mg(true, 3u);
    auto x = Vec::create(exec, gko::dim<2>(n, 1));
    x->fill(value_type{0.0});
    auto zero = Vec::create(exec, gko::dim<2>(n, 1));
    zero->fill(value_type{0.0});

    solver->apply(zero, x);

    for (index_type i = 0; i < n; i++) {
        ASSERT_TRUE(gko::is_finite(x->at(i, 0)));
    }
    GKO_ASSERT_MTX_NEAR(x, zero, value_type{0.0});
}


}  // namespace
