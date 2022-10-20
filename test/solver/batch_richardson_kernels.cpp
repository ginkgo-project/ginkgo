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

#include "core/solver/batch_richardson_kernels.hpp"


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/batch_convergence.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>
#include <ginkgo/core/solver/batch_richardson.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "core/test/utils/batch_test_utils.hpp"
#include "test/utils/executor.hpp"


#ifndef GKO_COMPILING_DPCPP


namespace {


class BatchRich : public CommonTestFixture {
protected:
    using real_type = gko::remove_complex<value_type>;
    using solver_type = gko::solver::BatchRichardson<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using Options = gko::kernels::batch_rich::BatchRichardsonOptions<real_type>;
    using LogData = gko::log::BatchLogData<value_type>;

    BatchRich()
        : sys_1(gko::test::get_poisson_problem<value_type>(ref, 1, nbatch))
    {
        solve_fn = [this](const Options opts, const Mtx* mtx,
                          const gko::BatchLinOp* prec, const BDense* b,
                          BDense* x, LogData& logdata) {
            gko::kernels::EXEC_NAMESPACE::batch_rich::apply<value_type>(
                this->exec, opts, mtx, prec, b, x, logdata);
        };
    }

    const size_t nbatch = 2;
    const int nrows = 3;
    const Options opts_1{500, r<real_type>::value,
                         gko::stop::batch::ToleranceType::relative, 1.0};
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
            return 40;
        } else if (std::is_same<real_type, double>::value) {
            return 98;
        } else {
            return -1;
        }
    }

    std::vector<int> multiple_iters_regression()
    {
        std::vector<int> iters(2);
        if (std::is_same<real_type, float>::value) {
            iters[0] = 40;
            iters[1] = 39;
        } else if (std::is_same<real_type, double>::value) {
            iters[0] = 98;
            iters[1] = 97;
        } else {
            iters[0] = -1;
            iters[1] = -1;
        }
        return iters;
    }
};


TEST_F(BatchRich, SolvesStencilSystemJacobi)
{
    using T = value_type;
    auto r_1 = gko::test::solve_poisson_uniform(
        exec, solve_fn, opts_1, sys_1, 1,
        gko::preconditioner::BatchJacobi<T>::build().on(exec));

    for (size_t i = 0; i < nbatch; i++) {
        ASSERT_LE(r_1.resnorm->get_const_values()[i] /
                      sys_1.bnorm->get_const_values()[i],
                  opts_1.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(r_1.x, sys_1.xex, 1e-6 /*r<value_type>::value*/);
}


TEST_F(BatchRich, StencilSystemJacobiLoggerIsCorrect)
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
        ASSERT_EQ(iter_array[i], ref_iters);
        ASSERT_LE(res_log_array[i] / sys_1.bnorm->get_const_values()[i],
                  opts_1.residual_tol);
        ASSERT_NEAR(res_log_array[i] / sys_1.bnorm->get_const_values()[i],
                    r_1.resnorm->get_const_values()[i] /
                        sys_1.bnorm->get_const_values()[i],
                    10 * r<value_type>::value);
    }
}


TEST_F(BatchRich, BetterRelaxationFactorGivesBetterConvergence)
{
    using T = value_type;
    const Options opts{1000, 1e-8, gko::stop::batch::ToleranceType::relative,
                       1.0};
    const Options opts_slower{1000, 1e-8,
                              gko::stop::batch::ToleranceType::relative, 0.8};

    auto result1 = gko::test::solve_poisson_uniform(
        exec, solve_fn, opts, sys_1, 1,
        gko::preconditioner::BatchJacobi<T>::build().on(exec));
    auto result2 = gko::test::solve_poisson_uniform(
        exec, solve_fn, opts_slower, sys_1, 1,
        gko::preconditioner::BatchJacobi<T>::build().on(exec));

    const int* const iter_arr1 = result1.logdata.iter_counts.get_const_data();
    const int* const iter_arr2 = result2.logdata.iter_counts.get_const_data();
    for (size_t i = 0; i < nbatch; i++) {
        ASSERT_LE(iter_arr1[i], iter_arr2[i]);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result2.x, sys_1.xex,
                              1e-6 /*r<value_type>::value*/);
}


TEST_F(BatchRich, CoreSolvesSystemJacobi)
{
    std::unique_ptr<typename solver_type::Factory> batchrich_factory =
        solver_type::build()
            .with_default_max_iterations(100)
            .with_default_residual_tol(5e-7f)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<value_type>::build().on(exec))
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .on(exec);
    const int nrhs_1 = 1;
    auto rx = BDense::create(
        ref, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrhs_1)));
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < nrhs_1; j++) {
                rx->at(ib, i, j) = 0.0;
            }
        }
    }
    auto mtx = Mtx::create(exec);
    auto b = BDense::create(exec);
    auto x = BDense::create(exec);
    mtx->copy_from(gko::lend(sys_1.mtx));
    b->copy_from(gko::lend(sys_1.b));
    x->copy_from(gko::lend(rx));

    std::unique_ptr<solver_type> solver =
        batchrich_factory->generate(gko::give(mtx));
    solver->apply(b.get(), x.get());

    rx->copy_from(gko::lend(x));
    const auto rnorms = gko::test::compute_residual_norm(
        sys_1.mtx.get(), rx.get(), sys_1.b.get());
    for (size_t i = 0; i < nbatch; i++) {
        ASSERT_LE(
            rnorms->get_const_values()[i] / sys_1.bnorm->get_const_values()[i],
            5e-7);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(rx, sys_1.xex, 1e-5 /*r<value_type>::value*/);
}


TEST_F(BatchRich, UnitScalingDoesNotChangeResult)
{
    auto left_scale =
        gko::share(gko::batch_initialize<BDiag>(nbatch, {1.0, 1.0, 1.0}, exec));
    auto right_scale =
        gko::share(gko::batch_initialize<BDiag>(nbatch, {1.0, 1.0, 1.0}, exec));
    auto factory = create_factory(
        exec, opts_1,
        gko::preconditioner::BatchJacobi<value_type>::build().on(exec),
        left_scale, right_scale);

    auto result = gko::test::solve_poisson_uniform_core<solver_type>(
        exec, factory.get(), sys_1, 1);

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, sys_1.xex, 1e2 * r<value_type>::value);
}


TEST_F(BatchRich, GeneralScalingDoesNotChangeResult)
{
    auto left_scale =
        gko::batch_initialize<BDiag>({{0.8, 0.9, 0.95}, {1.1, 3.2, 0.9}}, exec);
    auto right_scale =
        gko::batch_initialize<BDiag>(nbatch, {1.0, 1.5, 1.05}, exec);
    auto factory = create_factory(
        exec, opts_1,
        gko::preconditioner::BatchJacobi<value_type>::build().on(exec));

    auto result = gko::test::solve_poisson_uniform_core<solver_type>(
        exec, factory.get(), sys_1, 1);

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, sys_1.xex, 1e2 * r<value_type>::value);
}


TEST_F(BatchRich, CanSolveCsrSystemWithoutScaling)
{
    using T = std::complex<float>;
    using RT = typename gko::remove_complex<T>;
    using Solver = gko::solver::BatchRichardson<T>;
    using Csr = gko::matrix::BatchCsr<T>;
    const RT tol = 1e-5;
    const int maxits = 2000;
    const int nrows = 21;
    const size_t nbatch = 3;
    const int nrhs = 1;
    auto batchrich_factory =
        Solver::build()
            .with_default_max_iterations(maxits)
            .with_default_residual_tol(tol)
            .with_relaxation_factor(RT{0.95})
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<T>::build().on(exec))
            .on(exec);

    gko::test::test_solve<Solver, Csr>(exec, nbatch, nrows, nrhs, 10 * tol,
                                       maxits, batchrich_factory.get(), 2);
}

}  // namespace


#endif
