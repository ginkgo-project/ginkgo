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

#include <ginkgo/core/solver/batch_gmres.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/log/batch_convergence.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_gmres_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"

namespace {


template <typename T>
class BatchGmres : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using solver_type = gko::solver::BatchGmres<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using Options = gko::kernels::batch_gmres::BatchGmresOptions<real_type>;
    using LogData = gko::log::BatchLogData<value_type>;

    BatchGmres()
        : exec(gko::ReferenceExecutor::create()),
          sys_1(gko::test::get_poisson_problem<T>(exec, 1, nbatch))
    {
        auto execp = exec;
        solve_fn = [execp](const Options opts, const Mtx* mtx,
                           const gko::BatchLinOp* prec, const BDense* b,
                           BDense* x, LogData& logdata) {
            gko::kernels::reference::batch_gmres::apply<value_type>(
                execp, opts, mtx, prec, b, x, logdata);
        };
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    const real_type eps = r<value_type>::value;

    const size_t nbatch = 2;
    const int nrows = 3;
    const Options opts_1{500, static_cast<real_type>(10) * eps, 2,
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
            .with_max_iterations(opts.max_its)
            .with_residual_tol(opts.residual_tol)
            .with_tolerance_type(opts.tol_type)
            .with_preconditioner(prec_factory)
            .with_restart(opts.restart_num)
            .with_left_scaling_op(left_scale)
            .with_right_scaling_op(right_scale)
            .on(exec);
    }

    int single_iters_regression() const
    {
        if (std::is_same<real_type, float>::value) {
            return 18;
        } else if (std::is_same<real_type, double>::value) {
            return 50;
        } else {
            return -1;
        }
    }
};

TYPED_TEST_SUITE(BatchGmres, gko::test::ValueTypes);


TYPED_TEST(BatchGmres, SolvesStencilSystem)
{
    auto r_1 = gko::test::solve_poisson_uniform(this->exec, this->solve_fn,
                                                this->opts_1, this->sys_1, 1);

    GKO_ASSERT_BATCH_MTX_NEAR(r_1.x, this->sys_1.xex, 1e2 * this->eps);
}

TYPED_TEST(BatchGmres, StencilSystemLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto r_1 = gko::test::solve_poisson_uniform(this->exec, this->solve_fn,
                                                this->opts_1, this->sys_1, 1);

    const int ref_iters = this->single_iters_regression();
    const int* const iter_array = r_1.logdata.iter_counts.get_const_data();
    const real_type* const res_log_array =
        r_1.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        GKO_ASSERT((iter_array[i] <= ref_iters + 1) &&
                   (iter_array[i] >= ref_iters - 1));
        ASSERT_LE(res_log_array[i] / this->sys_1.bnorm->at(i, 0, 0),
                  this->opts_1.residual_tol);
        ASSERT_NEAR(res_log_array[i], r_1.resnorm->get_const_values()[i],
                    10 * this->eps);
    }
}


TYPED_TEST(BatchGmres, UnitScalingDoesNotChangeResult)
{
    using BDiag = typename TestFixture::BDiag;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::share(gko::batch_initialize<BDiag>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec));
    auto right_scale = gko::share(gko::batch_initialize<BDiag>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec));
    auto factory = this->create_factory(this->exec, this->opts_1, nullptr,
                                        left_scale, right_scale);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->exec, factory.get(), this->sys_1, 1);

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, 1e2 * this->eps);
}


TYPED_TEST(BatchGmres, GeneralScalingDoesNotChangeResult)
{
    using BDiag = typename TestFixture::BDiag;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::share(gko::batch_initialize<BDiag>(
        this->nbatch, {0.8, 0.9, 0.95}, this->exec));
    auto right_scale = gko::share(gko::batch_initialize<BDiag>(
        this->nbatch, {1.0, 1.5, 1.05}, this->exec));
    auto factory = this->create_factory(this->exec, this->opts_1);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->exec, factory.get(), this->sys_1, 1);

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, 1e2 * this->eps);
}


TEST(BatchGmres, GoodScalingImprovesConvergence)
{
    using value_type = double;
    using real_type = gko::remove_complex<value_type>;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using Solver = gko::solver::BatchGmres<value_type>;
    const auto eps = r<value_type>::value;
    auto exec = gko::ReferenceExecutor::create();
    const size_t nbatch = 3;
    const int nrows = 100;
    const int nrhs = 1;
    auto matsz = gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows));
    auto left_scale = gko::share(BDiag::create(exec, matsz));
    auto right_scale = gko::share(BDiag::create(exec, matsz));
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int i = 0; i < nrows; i++) {
            left_scale->at(ib, i) = std::sqrt(1.0 / (2.0 + i));
            right_scale->at(ib, i) = std::sqrt(1.0 / (2.0 + i));
        }
    }
    auto factory =
        Solver::build()
            .with_restart(15)
            .with_max_iterations(30)
            .with_residual_tol(10 * eps)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .on(exec);
    auto factory_s =
        Solver::build()
            .with_max_iterations(10)
            .with_residual_tol(10 * eps)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_left_scaling_op(left_scale)
            .with_right_scaling_op(right_scale)
            .on(exec);

    gko::test::test_solve_iterations_with_scaling<Solver>(
        exec, nbatch, nrows, nrhs, factory.get(), factory_s.get());
}


TEST(BatchGmres, CanSolveDenseWithScaling)
{
    using value_type = double;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using Solver = gko::solver::BatchGmres<value_type>;
    const auto eps = r<value_type>::value;
    auto exec = gko::ReferenceExecutor::create();
    const size_t nbatch = 3;
    const int nrhs = 1;
    auto left_scale = gko::share(
        gko::batch_initialize<BDiag>(nbatch, {0.8, 0.9, 0.95}, exec));
    auto right_scale = gko::share(
        gko::batch_initialize<BDiag>(nbatch, {1.0, 1.5, 1.05}, exec));
    auto factory_s =
        Solver::build()
            .with_max_iterations(10)
            .with_residual_tol(10 * eps)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_left_scaling_op(left_scale)
            .with_right_scaling_op(right_scale)
            .on(exec);
    auto sys = gko::test::get_poisson_problem<value_type>(exec, 1, nbatch);
    const auto nrows = sys.mtx->get_size().at(0)[0];
    auto A_dense = gko::share(BDense::create(exec));
    dynamic_cast<const Mtx*>(sys.mtx.get())->convert_to(A_dense.get());
    auto solver = factory_s->generate(A_dense);
    auto x = gko::clone(sys.xex);
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (size_t i = 0; i < nrows; i++) {
            x->at(ib, i) = 0.0;
        }
    }

    solver->apply(sys.b.get(), x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(x, sys.xex, 1e2 * eps);
}


TEST(BatchGmres, CanSolveWithoutScaling)
{
    using T = std::complex<double>;
    using RT = typename gko::remove_complex<T>;
    using Solver = gko::solver::BatchGmres<T>;
    using Dense = gko::matrix::BatchDense<T>;
    using RDense = gko::matrix::BatchDense<RT>;
    using Mtx = typename gko::matrix::BatchCsr<T>;
    using BDiag = gko::matrix::BatchDiagonal<T>;
    const RT tol = 1e-4;
    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
    const int nrows = 41;
    const size_t nbatch = 3;
    const int nrhs = 1;
    auto matsz = gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows));
    auto left_scale = gko::share(BDiag::create(exec, matsz));
    auto right_scale = gko::share(BDiag::create(exec, matsz));
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int i = 0; i < nrows; i++) {
            left_scale->at(ib, i) = std::sqrt(1.0 / (2.0 + i));
            right_scale->at(ib, i) = std::sqrt(1.0 / (2.0 + i));
        }
    }
    auto factory =
        Solver::build()
            .with_max_iterations(1000)
            .with_residual_tol(tol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<T>::build().on(exec))
            .with_restart(2)
            .on(exec);
    auto factory_s =
        Solver::build()
            .with_max_iterations(1000)
            .with_residual_tol(tol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(
                gko::preconditioner::BatchJacobi<T>::build().on(exec))
            .with_restart(2)
            .with_left_scaling_op(left_scale)
            .with_right_scaling_op(right_scale)
            .on(exec);

    gko::test::test_solve_iterations_with_scaling<Solver>(
        exec, nbatch, nrows, nrhs, factory.get(), factory_s.get());
}


}  // namespace
