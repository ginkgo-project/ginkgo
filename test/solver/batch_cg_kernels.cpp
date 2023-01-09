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

#include "core/solver/batch_cg_kernels.hpp"


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/batch_convergence.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>
#include <ginkgo/core/solver/batch_cg.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"
#include "test/utils/executor.hpp"


#ifndef GKO_COMPILING_DPCPP


namespace {


class BatchCg : public CommonTestFixture {
protected:
    using real_type = gko::remove_complex<value_type>;
    using solver_type = gko::solver::BatchCg<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using Options = gko::kernels::batch_cg::BatchCgOptions<real_type>;
    using LogData = gko::log::BatchLogData<value_type>;

    BatchCg()
        : sys_1(gko::test::get_poisson_problem<value_type>(ref, 1, nbatch))
    {
        solve_fn = [this](const Options opts, const Mtx* mtx,
                          const gko::BatchLinOp* prec, const BDense* b,
                          BDense* x, LogData& logdata) {
            gko::kernels::EXEC_NAMESPACE::batch_cg::apply<value_type>(
                this->exec, opts, mtx, prec, b, x, logdata);
        };
    }

    const real_type eps = r<value_type>::value;

    const size_t nbatch = 2;
    const int nrows = 3;
    const Options opts_1{500, static_cast<real_type>(1e2) * eps,
                         gko::stop::batch::ToleranceType::relative};

    gko::test::LinSys<value_type> sys_1;

    std::function<void(Options, const Mtx*, const gko::BatchLinOp*,
                       const BDense*, BDense*, LogData&)>
        solve_fn;

    std::unique_ptr<typename solver_type::Factory> create_factory(
        std::shared_ptr<const gko::Executor> exec, const Options& opts,
        std::shared_ptr<gko::BatchLinOpFactory> prec_factory = nullptr,
        std::shared_ptr<const BDiag> left_scale = nullptr,
        std::shared_ptr<const BDiag> right_scale = nullptr)
    {
        return solver_type::build()
            .with_default_max_iterations(opts.max_its)
            .with_default_residual_tol(opts.residual_tol)
            .with_tolerance_type(opts.tol_type)
            .with_preconditioner(prec_factory)
            .with_left_scaling_op(left_scale)
            .with_right_scaling_op(right_scale)
            .on(exec);
    }

    int single_iters_regression()
    {
        if (std::is_same<real_type, float>::value) {
            return 3;
        } else if (std::is_same<real_type, double>::value) {
            return 3;
        } else {
            return -1;
        }
    }
};


TEST_F(BatchCg, SolveIsEquivalentToReference)
{
    using mtx_type = Mtx;
    using opts_type = Options;
    constexpr bool issingle =
        std::is_same<gko::remove_complex<value_type>, float>::value;
    const float solver_restol = issingle ? 100 * eps : eps;
    const opts_type opts{500, solver_restol,
                         gko::stop::batch::ToleranceType::relative};
    auto r_sys = gko::test::generate_solvable_batch_system<mtx_type>(
        ref, nbatch, 11, 1, true);
    auto r_factory = create_factory(ref, opts);
    auto d_factory = create_factory(exec, opts);
    const double iter_tol = 0.01;
    const double res_tol = 10 * r<value_type>::value;
    const double sol_tol = 10 * solver_restol;

    gko::test::compare_with_reference<value_type, solver_type>(
        exec, r_sys, r_factory.get(), d_factory.get(), iter_tol, res_tol,
        sol_tol);
}


TEST_F(BatchCg, StencilSystemLoggerIsCorrect)
{
    using real_type = gko::remove_complex<value_type>;

    auto r_1 = gko::test::solve_poisson_uniform(
        exec, solve_fn, opts_1, sys_1, 1,
        gko::preconditioner::BatchJacobi<value_type>::build().on(exec));

    const int ref_iters = single_iters_regression();
    const int* const iter_array = r_1.logdata.iter_counts.get_const_data();
    const real_type* const res_log_array =
        r_1.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < nbatch; i++) {
        GKO_ASSERT((iter_array[i] <= ref_iters + 1) &&
                   (iter_array[i] >= ref_iters - 1));
        ASSERT_LE(res_log_array[i] / sys_1.bnorm->at(0, 0, i),
                  opts_1.residual_tol);
        ASSERT_NEAR(res_log_array[i], r_1.resnorm->get_const_values()[i],
                    10 * eps);
    }
}


TEST_F(BatchCg, CoreSolvesSystemJacobi)
{
    std::unique_ptr<typename solver_type::Factory> batchcg_factory =
        solver_type::build()
            .with_default_max_iterations(100)
            .with_default_residual_tol(1e-6f)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<value_type>::build().on(exec))
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .on(exec);
    const int nrhs_1 = 1;
    const size_t nbatch = 3;
    const auto sys =
        gko::test::get_poisson_problem<value_type>(ref, nrhs_1, nbatch);
    auto rx = gko::batch_initialize<BDense>(nbatch, {0.0, 0.0, 0.0}, ref);
    std::unique_ptr<Mtx> mtx = Mtx::create(exec);
    auto b = BDense::create(exec);
    auto x = BDense::create(exec);
    mtx->copy_from(gko::lend(sys.mtx));
    b->copy_from(gko::lend(sys.b));
    x->copy_from(gko::lend(rx));

    std::unique_ptr<solver_type> solver =
        batchcg_factory->generate(gko::give(mtx));
    solver->apply(b.get(), x.get());
    rx->copy_from(gko::lend(x));

    GKO_ASSERT_BATCH_MTX_NEAR(rx, sys.xex, 1e-5);
}


TEST_F(BatchCg, UnitScalingDoesNotChangeResult)
{
    auto left_scale =
        gko::share(gko::batch_initialize<BDiag>(nbatch, {1.0, 1.0, 1.0}, exec));
    auto right_scale =
        gko::share(gko::batch_initialize<BDiag>(nbatch, {1.0, 1.0, 1.0}, exec));
    auto factory =
        create_factory(exec, opts_1, nullptr, left_scale, right_scale);

    auto result = gko::test::solve_poisson_uniform_core<solver_type>(
        exec, factory.get(), sys_1, 1);

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, sys_1.xex, 1e2 * eps);
}


TEST_F(BatchCg, GeneralScalingDoesNotChangeResult)
{
    auto left_scale = gko::share(
        gko::batch_initialize<BDiag>(nbatch, {0.8, 1.5, 0.95}, exec));
    auto right_scale = gko::share(
        gko::batch_initialize<BDiag>(nbatch, {0.8, 1.5, 0.95}, exec));
    auto factory =
        create_factory(exec, opts_1, nullptr, left_scale, right_scale);

    auto result = gko::test::solve_poisson_uniform_core<solver_type>(
        exec, factory.get(), sys_1, 1);

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, sys_1.xex, 1e3 * eps);
}


TEST_F(BatchCg, GoodScalingImprovesConvergence)
{
    const auto eps = r<value_type>::value;
    const size_t nbatch = 3;
    const int nrows = 100;
    const int nrhs = 1;
    auto matsz = gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows));
    auto left_scale = gko::share(BDiag::create(ref, matsz));
    auto right_scale = gko::share(BDiag::create(ref, matsz));
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int i = 0; i < nrows; i++) {
            left_scale->at(ib, i) = std::sqrt(1.0 / (2.0 + i));
            right_scale->at(ib, i) = std::sqrt(1.0 / (2.0 + i));
        }
    }
    auto d_left = gko::share(gko::clone(exec, left_scale));
    auto d_right = gko::share(gko::clone(exec, right_scale));
    auto factory =
        solver_type::build()
            .with_default_max_iterations(20)
            .with_default_residual_tol(10 * eps)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .on(exec);
    auto factory_s =
        solver_type::build()
            .with_default_max_iterations(10)
            .with_default_residual_tol(10 * eps)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_left_scaling_op(d_left)
            .with_right_scaling_op(d_right)
            .on(exec);

    gko::test::test_solve_iterations_with_scaling<solver_type>(
        exec, nbatch, nrows, nrhs, factory.get(), factory_s.get());
}


TEST_F(BatchCg, CanSolveCsrSystemWithoutScaling)
{
    using T = std::complex<float>;
    using RT = typename gko::remove_complex<T>;
    using Solver = gko::solver::BatchCg<T>;
    using Csr = gko::matrix::BatchCsr<T>;
    const RT tol = 1e-5;
    const int maxits = 100;
    auto batchcg_factory =
        Solver::build()
            .with_default_max_iterations(maxits)
            .with_default_residual_tol(tol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<T>::build().on(exec))
            .on(exec);
    const int nrows = 28;
    const size_t nbatch = 3;
    const int nrhs = 1;

    gko::test::test_solve<Solver, Csr>(exec, nbatch, nrows, nrhs, tol, maxits,
                                       batchcg_factory.get(), 10);
}


}  // namespace


#endif
