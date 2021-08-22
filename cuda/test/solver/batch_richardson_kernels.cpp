/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/solver/batch_richardson.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/batch_convergence.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_richardson_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "core/test/utils/batch_test_utils.hpp"


namespace {


namespace gpb = gko::preconditioner::batch;


template <typename T>
class BatchRich : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using solver_type = gko::solver::BatchRichardson<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using Options = gko::kernels::batch_rich::BatchRichardsonOptions<real_type>;
    using LogData = gko::log::BatchLogData<value_type>;

    BatchRich()
        : exec(gko::ReferenceExecutor::create()),
          cuexec(gko::CudaExecutor::create(0, exec)),
          sys_1(gko::test::get_poisson_problem<value_type>(exec, 1, nbatch))
    {
        auto execp = cuexec;
        solve_fn = [execp](const Options opts, const Mtx *mtx, const BDense *b,
                           BDense *x, LogData &logdata) {
            gko::kernels::cuda::batch_rich::apply<value_type>(execp, opts, mtx,
                                                              b, x, logdata);
        };
        scale_mat = [execp](const BDense *const left, const BDense *const right,
                            Mtx *const mat) {
            gko::kernels::cuda::batch_csr::batch_scale<value_type>(execp, left,
                                                                   right, mat);
        };
        scale_vecs = [execp](const BDense *const scale, BDense *const mat) {
            gko::kernels::cuda::batch_dense::batch_scale<value_type>(
                execp, scale, mat);
        };
    }

    void TearDown()
    {
        if (cuexec != nullptr) {
            ASSERT_NO_THROW(cuexec->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::CudaExecutor> cuexec;

    const size_t nbatch = 2;
    const int nrows = 3;
    const Options opts_1{gpb::type::jacobi, 500, r<real_type>::value,
                         gko::stop::batch::ToleranceType::relative, 1.0};
    gko::test::LinSys<value_type> sys_1;

    std::function<void(Options, const Mtx *, const BDense *, BDense *,
                       LogData &)>
        solve_fn;
    std::function<void(const BDense *, const BDense *, Mtx *)> scale_mat;
    std::function<void(const BDense *, BDense *)> scale_vecs;

    std::unique_ptr<typename solver_type::Factory> create_factory(
        std::shared_ptr<const gko::Executor> exec, const Options &opts)
    {
        return solver_type::build()
            .with_max_iterations(opts.max_its)
            .with_residual_tol(opts.residual_tol)
            .with_tolerance_type(opts.tol_type)
            .with_preconditioner(opts.preconditioner)
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

TYPED_TEST_SUITE(BatchRich, gko::test::ValueTypes);

TYPED_TEST(BatchRich, SolvesStencilSystemJacobi)
{
    auto r_1 = gko::test::solve_poisson_uniform(
        this->cuexec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts_1, this->sys_1, 1);

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(r_1.resnorm->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  this->opts_1.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(r_1.x, this->sys_1.xex,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, StencilSystemJacobiLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto r_1 = gko::test::solve_poisson_uniform(
        this->cuexec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts_1, this->sys_1, 1);

    const int ref_iters = this->single_iters_regression();
    const int *const iter_array = r_1.logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        r_1.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(iter_array[i], ref_iters);
        ASSERT_LE(res_log_array[i] / this->sys_1.bnorm->get_const_values()[i],
                  this->opts_1.residual_tol);
        ASSERT_NEAR(res_log_array[i] / this->sys_1.bnorm->get_const_values()[i],
                    r_1.resnorm->get_const_values()[i] /
                        this->sys_1.bnorm->get_const_values()[i],
                    10 * r<value_type>::value);
    }
}


TYPED_TEST(BatchRich, BetterRelaxationFactorGivesBetterConvergence)
{
    using BDense = typename TestFixture::BDense;
    using Options = typename TestFixture::Options;
    const Options opts{gpb::type::jacobi, 1000, 1e-8,
                       gko::stop::batch::ToleranceType::relative, 1.0};
    const Options opts_slower{gpb::type::jacobi, 1000, 1e-8,
                              gko::stop::batch::ToleranceType::relative, 0.8};

    auto result1 = gko::test::solve_poisson_uniform(
        this->cuexec, this->solve_fn, this->scale_mat, this->scale_vecs, opts,
        this->sys_1, 1);
    auto result2 = gko::test::solve_poisson_uniform(
        this->cuexec, this->solve_fn, this->scale_mat, this->scale_vecs,
        opts_slower, this->sys_1, 1);

    const int *const iter_arr1 = result1.logdata.iter_counts.get_const_data();
    const int *const iter_arr2 = result2.logdata.iter_counts.get_const_data();
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(iter_arr1[i], iter_arr2[i]);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result2.x, this->sys_1.xex,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, CoreSolvesSystemJacobi)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using BDense = typename TestFixture::BDense;
    using Solver = gko::solver::BatchRichardson<value_type>;
    auto useexec = this->cuexec;
    std::unique_ptr<typename Solver::Factory> batchrich_factory =
        Solver::build()
            .with_max_iterations(100)
            .with_residual_tol(5e-7f)
            .with_preconditioner(gpb::type::jacobi)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .on(useexec);
    const int nrhs_1 = 1;
    const size_t nbatch = this->nbatch;
    auto rx = BDense::create(
        this->exec, gko::batch_dim<>(nbatch, gko::dim<2>(this->nrows, nrhs_1)));
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int i = 0; i < this->nrows; i++) {
            for (int j = 0; j < nrhs_1; j++) {
                rx->at(ib, i, j) = 0.0;
            }
        }
    }
    std::unique_ptr<Mtx> mtx = Mtx::create(useexec);
    auto b = BDense::create(useexec);
    auto x = BDense::create(useexec);
    mtx->copy_from(gko::lend(this->sys_1.mtx));
    b->copy_from(gko::lend(this->sys_1.b));
    x->copy_from(gko::lend(rx));

    std::unique_ptr<Solver> solver =
        batchrich_factory->generate(gko::give(mtx));
    solver->apply(b.get(), x.get());

    rx->copy_from(gko::lend(x));
    const auto rnorms = gko::test::compute_residual_norm(
        this->sys_1.mtx.get(), rx.get(), this->sys_1.b.get());
    for (size_t i = 0; i < nbatch; i++) {
        ASSERT_LE(rnorms->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  5e-7);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(rx, this->sys_1.xex,
                              1e-5 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, UnitScalingDoesNotChangeResult)
{
    using BDense = typename TestFixture::BDense;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto factory = this->create_factory(this->cuexec, this->opts_1);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->cuexec, factory.get(), this->sys_1, 1, left_scale.get(),
        right_scale.get());

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(result.resnorm->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  this->opts_1.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, 1e-6);
}


TYPED_TEST(BatchRich, GeneralScalingDoesNotChangeResult)
{
    using BDense = typename TestFixture::BDense;
    using Solver = typename TestFixture::solver_type;
    using value_type = typename TestFixture::value_type;
    auto left_scale = gko::batch_initialize<BDense>(
        {{0.8, 0.9, 0.95}, {1.1, 3.2, 0.9}}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.5, 1.05}, this->exec);
    auto factory = this->create_factory(this->cuexec, this->opts_1);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->cuexec, factory.get(), this->sys_1, 1, left_scale.get(),
        right_scale.get());

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(result.resnorm->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  3 * this->opts_1.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, 1e-5);
}

TEST(BatchRich, CanSolveWithoutScaling)
{
    using T = std::complex<float>;
    using RT = typename gko::remove_complex<T>;
    using Solver = gko::solver::BatchRichardson<T>;
    const RT tol = 1e-5;
    std::shared_ptr<gko::ReferenceExecutor> refexec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<const gko::CudaExecutor> exec =
        gko::CudaExecutor::create(0, refexec);
    const int maxits = 2000;
    const int nrows = 21;
    const size_t nbatch = 3;
    const int nrhs = 1;
    auto batchrich_factory =
        Solver::build()
            .with_max_iterations(maxits)
            .with_residual_tol(tol)
            .with_relaxation_factor(RT{0.95})
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .on(exec);
    gko::test::test_solve<Solver>(exec, nbatch, nrows, nrhs, 10 * tol, maxits,
                                  batchrich_factory.get(), 2);
}

}  // namespace
