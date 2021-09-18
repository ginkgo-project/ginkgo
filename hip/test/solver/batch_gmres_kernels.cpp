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

#include <ginkgo/core/solver/batch_gmres.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/batch_convergence.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_gmres_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "core/test/utils/batch_test_utils.hpp"

namespace {


template <typename T>
class BatchGmres : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using solver_type = gko::solver::BatchGmres<value_type>;
    using Options = gko::kernels::batch_gmres::BatchGmresOptions<real_type>;
    using LogData = gko::log::BatchLogData<value_type>;

    BatchGmres()
        : exec(gko::ReferenceExecutor::create()),
          d_exec(gko::HipExecutor::create(0, exec)),
          sys_1(gko::test::get_poisson_problem<T>(exec, 1, nbatch))
    {
        auto execp = d_exec;
        solve_fn = [execp](const Options opts, const Mtx* mtx, const BDense* b,
                           BDense* x, LogData& logdata) {
            gko::kernels::hip::batch_gmres::apply<value_type>(execp, opts, mtx,
                                                              b, x, logdata);
        };
        scale_mat = [execp](const BDense* const left, const BDense* const right,
                            Mtx* const mat, BDense* const b) {
            gko::kernels::hip::batch_csr::pre_diag_scale_system<value_type>(
                execp, left, right, mat, b);
        };
        scale_vecs = [execp](const BDense* const scale, BDense* const mat) {
            gko::kernels::hip::batch_dense::batch_scale<value_type>(execp,
                                                                    scale, mat);
        };
    }

    void TearDown()
    {
        if (d_exec != nullptr) {
            ASSERT_NO_THROW(d_exec->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::HipExecutor> d_exec;

    const real_type eps = r<value_type>::value;

    const size_t nbatch = 2;
    const int nrows = 3;

    const Options opts_1{gko::preconditioner::batch::type::none, 500,
                         static_cast<real_type>(10) * eps, 10,
                         gko::stop::batch::ToleranceType::relative};

    gko::test::LinSys<T> sys_1;

    std::function<void(Options, const Mtx*, const BDense*, BDense*, LogData&)>
        solve_fn;
    std::function<void(const BDense*, const BDense*, Mtx*, BDense*)> scale_mat;
    std::function<void(const BDense*, BDense*)> scale_vecs;

    std::unique_ptr<typename solver_type::Factory> create_factory(
        std::shared_ptr<const gko::Executor> exec, const Options& opts)
    {
        return solver_type::build()
            .with_max_iterations(opts.max_its)
            .with_residual_tol(opts.residual_tol)
            .with_tolerance_type(opts.tol_type)
            .with_preconditioner(opts.preconditioner)
            .with_restart(opts.restart_num)
            .on(exec);
    }

    int single_iters_regression() { return 3; }
};

TYPED_TEST_SUITE(BatchGmres, gko::test::ValueTypes);


TYPED_TEST(BatchGmres, SolvesSystemEquivalentToReference)
{
    using value_type = typename TestFixture::value_type;
    using solver_type = gko::solver::BatchGmres<value_type>;
    using opts_type = typename TestFixture::Options;
    const opts_type opts{gko::preconditioner::batch::type::none, 700, this->eps,
                         10, gko::stop::batch::ToleranceType::relative};
    auto r_sys = gko::test::generate_solvable_batch_system<value_type>(
        this->exec, this->nbatch, 7, 1, false);
    auto r_factory = this->create_factory(this->exec, opts);
    const double iter_tol = 0.01;
    const double res_tol = 10 * r<value_type>::value;
    const double sol_tol = 100 * res_tol;

    gko::test::compare_with_reference<value_type, solver_type>(
        this->d_exec, r_sys, r_factory.get(), false, iter_tol, res_tol,
        sol_tol);
}


TYPED_TEST(BatchGmres, StencilSystemLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto r_1 = gko::test::solve_poisson_uniform(
        this->d_exec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts_1, this->sys_1, 1);

    const int ref_iters = this->single_iters_regression();
    const int* const iter_array = r_1.logdata.iter_counts.get_const_data();
    const real_type* const res_log_array =
        r_1.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        GKO_ASSERT((iter_array[i] <= ref_iters + 1) &&
                   (iter_array[i] >= ref_iters - 1));
        ASSERT_LE(res_log_array[i] / this->sys_1.bnorm->at(0, 0, i),
                  this->opts_1.residual_tol);
        ASSERT_NEAR(res_log_array[i], r_1.resnorm->get_const_values()[i],
                    10 * this->eps);
    }
}


TYPED_TEST(BatchGmres, CoreSolvesSystemJacobi)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using BDense = typename TestFixture::BDense;
    using Solver = gko::solver::BatchGmres<value_type>;
    auto useexec = this->d_exec;
    std::unique_ptr<typename Solver::Factory> batchgmres_factory =
        Solver::build()
            .with_max_iterations(100)
            .with_residual_tol(1e-6f)
            .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .with_restart(2)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .on(useexec);
    const int nrhs_1 = 1;
    const size_t nbatch = 3;
    const auto sys =
        gko::test::get_poisson_problem<value_type>(this->exec, nrhs_1, nbatch);
    auto rx =
        gko::batch_initialize<BDense>(nbatch, {0.0, 0.0, 0.0}, this->exec);
    std::unique_ptr<Mtx> mtx = Mtx::create(useexec);
    auto b = BDense::create(useexec);
    auto x = BDense::create(useexec);
    mtx->copy_from(gko::lend(sys.mtx));
    b->copy_from(gko::lend(sys.b));
    x->copy_from(gko::lend(rx));

    std::unique_ptr<Solver> solver =
        batchgmres_factory->generate(gko::give(mtx));
    solver->apply(b.get(), x.get());
    rx->copy_from(gko::lend(x));

    GKO_ASSERT_BATCH_MTX_NEAR(rx, sys.xex, 1e-5);
}


TYPED_TEST(BatchGmres, UnitScalingDoesNotChangeResult)
{
    using BDense = typename TestFixture::BDense;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto factory = this->create_factory(this->d_exec, this->opts_1);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->d_exec, factory.get(), this->sys_1, 1, left_scale.get(),
        right_scale.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, 1e2 * this->eps);
}


TYPED_TEST(BatchGmres, GeneralScalingDoesNotChangeResult)
{
    using BDense = typename TestFixture::BDense;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {0.8, 0.9, 0.95}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.5, 1.05}, this->exec);
    auto factory = this->create_factory(this->d_exec, this->opts_1);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->d_exec, factory.get(), this->sys_1, 1, left_scale.get(),
        right_scale.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, 1e2 * this->eps);
}


TEST(BatchGmres, GoodScalingImprovesConvergence)
{
    using value_type = double;
    using real_type = gko::remove_complex<value_type>;
    using Solver = gko::solver::BatchGmres<value_type>;
    const auto eps = r<value_type>::value;
    auto refexec = gko::ReferenceExecutor::create();
    std::shared_ptr<const gko::HipExecutor> d_exec =
        gko::HipExecutor::create(0, refexec);
    const size_t nbatch = 3;
    const int nrows = 100;
    const int nrhs = 1;
    auto factory =
        Solver::build()
            .with_max_iterations(20)
            .with_residual_tol(10 * eps)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(gko::preconditioner::batch::type::none)
            .on(d_exec);

    gko::test::test_solve_iterations_with_scaling<Solver>(d_exec, nbatch, nrows,
                                                          nrhs, factory.get());
}

}  // namespace