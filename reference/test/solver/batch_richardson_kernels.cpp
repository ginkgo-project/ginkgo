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

#include <ginkgo/core/solver/batch_richardson.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/log/batch_convergence.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_richardson_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"


namespace {


template <typename T>
class BatchRich : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using solver_type = gko::solver::BatchRichardson<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using Options = gko::kernels::batch_rich::BatchRichardsonOptions<real_type>;
    using LogData = gko::log::BatchLogData<value_type>;

    BatchRich() : exec(gko::ReferenceExecutor::create())
    {
        sys_1.xex =
            gko::batch_initialize<BDense>(nbatch, {1.0, 3.0, 2.0}, exec);
        sys_1.b = gko::batch_initialize<BDense>(nbatch, {-1.0, 3.0, 1.0}, exec);
        sys_1.mtx = gko::test::create_poisson1d_batch<Mtx>(exec, nrows, nbatch);
        sys_1.bnorm = gko::batch_initialize<RBDense>(nbatch, {0.0}, exec);
        sys_1.b->compute_norm2(sys_1.bnorm.get());

        auto execp = this->exec;
        solve_fn = [execp](const Options opts, const Mtx* mtx,
                           const gko::BatchLinOp* prec, const BDense* b,
                           BDense* x, LogData& logdata) {
            gko::kernels::reference::batch_rich::apply<value_type>(
                execp, opts, mtx, prec, b, x, logdata);
        };
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;

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

    int single_iters_regression() const
    {
        if (std::is_same<real_type, float>::value) {
            return 40;
        } else if (std::is_same<real_type, double>::value) {
            return 98;
        } else {
            return -1;
        }
    }
};

TYPED_TEST_SUITE(BatchRich, gko::test::ValueTypes);


TYPED_TEST(BatchRich, SolvesStencilSystemJacobi)
{
    using T = typename TestFixture::value_type;

    auto r_1 = gko::test::solve_poisson_uniform(
        this->exec, this->solve_fn, this->opts_1, this->sys_1, 1,
        gko::preconditioner::BatchJacobi<T>::build().on(this->exec));

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(r_1.resnorm->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  this->opts_1.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(r_1.x, this->sys_1.xex,
                              1e-6 /*r<value_type>::value*/);
}

TYPED_TEST(BatchRich, StencilSystemJacobiLoggerIsSameAsBefore)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto r_1 = gko::test::solve_poisson_uniform<value_type>(
        this->exec, this->solve_fn, this->opts_1, this->sys_1, 1,
        gko::preconditioner::BatchJacobi<value_type>::build().on(this->exec));

    const int ref_iters = this->single_iters_regression();
    const int* const iter_array = r_1.logdata.iter_counts.get_const_data();
    const real_type* const res_log_array =
        r_1.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
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
    using value_type = typename TestFixture::value_type;
    using Options = typename TestFixture::Options;
    const Options opts{1000, 1e-8, gko::stop::batch::ToleranceType::relative,
                       1.0};
    const Options opts_slower{1000, 1e-8,
                              gko::stop::batch::ToleranceType::relative, 0.8};

    auto result1 = gko::test::solve_poisson_uniform<value_type>(
        this->exec, this->solve_fn, opts, this->sys_1, 1,
        gko::preconditioner::BatchJacobi<value_type>::build().on(this->exec));
    auto result2 = gko::test::solve_poisson_uniform<value_type>(
        this->exec, this->solve_fn, opts_slower, this->sys_1, 1,
        gko::preconditioner::BatchJacobi<value_type>::build().on(this->exec));

    const int* const iter_arr1 = result1.logdata.iter_counts.get_const_data();
    const int* const iter_arr2 = result2.logdata.iter_counts.get_const_data();
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(iter_arr1[i], iter_arr2[i]);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result2.x, this->sys_1.xex,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, UnitScalingDoesNotChangeResult)
{
    using value_type = typename TestFixture::value_type;
    using BDiag = typename TestFixture::BDiag;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::share(gko::batch_diagonal_initialize<value_type>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec));
    auto right_scale = gko::share(gko::batch_diagonal_initialize<value_type>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec));
    auto factory = this->create_factory(
        this->exec, this->opts_1,
        gko::preconditioner::BatchJacobi<value_type>::build().on(this->exec),
        left_scale, right_scale);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->exec, factory.get(), this->sys_1, 1);

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(result.resnorm->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  this->opts_1.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, GeneralScalingDoesNotChangeResult)
{
    using value_type = typename TestFixture::value_type;
    using BDiag = typename TestFixture::BDiag;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::share(gko::batch_initialize<BDiag>(
        this->nbatch, {0.8, 0.9, 0.95}, this->exec));
    auto right_scale = gko::share(gko::batch_initialize<BDiag>(
        this->nbatch, {1.0, 1.5, 1.05}, this->exec));
    auto factory = this->create_factory(
        this->exec, this->opts_1,
        gko::preconditioner::BatchJacobi<value_type>::build().on(this->exec),
        left_scale, right_scale);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->exec, factory.get(), this->sys_1, 1);

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(result.resnorm->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  2 * this->opts_1.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex,
                              1e-5 /*r<value_type>::value*/);
}


TEST(BatchRich, CoreCanSolveWithoutScaling)
{
    using T = std::complex<float>;
    using RT = typename gko::remove_complex<T>;
    using Solver = gko::solver::BatchRichardson<T>;
    using Mtx = gko::matrix::BatchCsr<T, int>;
    const RT tol = 200 * std::numeric_limits<RT>::epsilon();
    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
    const int maxits = 10000;
    auto batchrich_factory =
        Solver::build()
            .with_default_max_iterations(maxits)
            .with_default_residual_tol(tol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_relaxation_factor(RT{0.98})
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<T>::build().on(exec))
            .on(exec);
    const int nrows = 42;
    const size_t nbatch = 3;
    const int nrhs = 1;
    gko::test::test_solve<Solver, Mtx>(exec, nbatch, nrows, nrhs, tol, maxits,
                                       batchrich_factory.get());
}


TEST(BatchRich, CoreCanSolveWithScaling)
{
    using T = double;
    using RT = typename gko::remove_complex<T>;
    using Mtx = gko::matrix::BatchCsr<T, int>;
    using Solver = gko::solver::BatchRichardson<T>;
    const RT tol = 10000 * std::numeric_limits<RT>::epsilon();
    const int maxits = 10000;
    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
    auto batchrich_factory =
        Solver::build()
            .with_default_max_iterations(maxits)
            .with_default_residual_tol(tol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<T>::build().on(exec))
            .on(exec);
    const int nrows = 40;
    const size_t nbatch = 3;
    const int nrhs = 1;

    gko::test::test_solve<Solver, Mtx>(exec, nbatch, nrows, nrhs, 20 * tol,
                                       maxits, batchrich_factory.get(), 10,
                                       true);
}


}  // namespace
