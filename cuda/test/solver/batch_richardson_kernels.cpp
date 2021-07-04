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
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using Options = gko::kernels::batch_rich::BatchRichardsonOptions<real_type>;

    BatchRich()
        : exec(gko::ReferenceExecutor::create()),
          cuexec(gko::CudaExecutor::create(0, exec)),
          sys_1(get_poisson_problem(1, nbatch)),
          sys_m(get_poisson_problem(nrhs, nbatch)),
          r_1(solve_poisson_uniform_1(opts_1)),
          r_m(solve_poisson_uniform_mult())
    {}

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
    const Options opts_1{gpb::type::jacobi, 500, r<real_type>::value, 1.0};
    const int nrhs = 2;
    const Options opts_m{gpb::type::jacobi, 500, r<real_type>::value, 1.0};

    struct LinSys {
        std::unique_ptr<Mtx> mtx;
        std::unique_ptr<BDense> b;
        std::unique_ptr<RBDense> bnorm;
        std::unique_ptr<BDense> xex;
    };
    LinSys sys_1;
    LinSys sys_m;
    struct Result {
        std::shared_ptr<BDense> x;
        std::shared_ptr<BDense> residual;
        std::shared_ptr<RBDense> resnorm;
        gko::log::BatchLogData<value_type> logdata;
    };
    Result r_1;
    Result r_m;

    LinSys get_poisson_problem(const int nrhs, const size_t nbatches)
    {
        LinSys sys;
        sys.mtx = gko::test::create_poisson1d_batch<value_type>(exec, nrows,
                                                                nbatches);
        if (nrhs == 1) {
            sys.b =
                gko::batch_initialize<BDense>(nbatches, {-1.0, 3.0, 1.0}, exec);
            sys.xex =
                gko::batch_initialize<BDense>(nbatches, {1.0, 3.0, 2.0}, exec);
        } else if (nrhs == 2) {
            sys.b = gko::batch_initialize<BDense>(
                nbatches,
                std::initializer_list<std::initializer_list<value_type>>{
                    {-1.0, 4.0}, {3.0, -5.0}, {1.0, 2.0}},
                exec);
            sys.xex = gko::batch_initialize<BDense>(
                nbatches,
                std::initializer_list<std::initializer_list<value_type>>{
                    {1.0, 1.0}, {3.0, -2.0}, {2.0, 0.0}},
                exec);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
        const gko::batch_dim<> normdim(nbatches, gko::dim<2>(1, nrhs));
        sys.bnorm = RBDense::create(exec, normdim);
        sys.b->compute_norm2(sys.bnorm.get());
        return sys;
    }

    std::unique_ptr<RBDense> compute_residual_norm(const Mtx *const rmtx,
                                                   const BDense *const x,
                                                   const BDense *const b)
    {
        std::unique_ptr<BDense> res = b->clone();
        const size_t nbatches = x->get_num_batch_entries();
        const int xnrhs = x->get_size().at(0)[1];
        const gko::batch_stride stride(nbatches, xnrhs);
        const gko::batch_dim<> normdim(nbatches, gko::dim<2>(1, xnrhs));
        auto normsr = RBDense::create(exec, normdim);
        auto alpha =
            gko::batch_initialize<BDense>(nbatches, {-1.0}, this->exec);
        auto beta = gko::batch_initialize<BDense>(nbatches, {1.0}, this->exec);
        rmtx->apply(alpha.get(), x, beta.get(), res.get());
        res->compute_norm2(normsr.get());
        return normsr;
    }

    // Uses sys_1
    Result solve_poisson_uniform_1(const Options opts,
                                   const BDense *const left_scale = nullptr,
                                   const BDense *const right_scale = nullptr)
    {
        const int nrhs_1 = 1;
        Result res;
        std::vector<gko::dim<2>> sizes(nbatch, gko::dim<2>(nrows, nrhs_1));
        res.x = BDense::create(exec, sizes);
        value_type *const xvalsinit = res.x->get_values();
        for (size_t i = 0; i < nbatch * nrows * nrhs_1; i++) {
            xvalsinit[i] = gko::zero<value_type>();
        }

        std::vector<gko::dim<2>> normsizes(nbatch, gko::dim<2>(1, nrhs_1));
        gko::log::BatchLogData<value_type> logdata;
        logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->cuexec, normsizes);
        logdata.iter_counts.set_executor(this->cuexec);
        logdata.iter_counts.resize_and_reset(nrhs_1 * nbatch);

        auto mtx = Mtx::create(this->cuexec);
        auto b = BDense::create(this->cuexec);
        auto x = BDense::create(this->cuexec);
        mtx->copy_from(gko::lend(sys_1.mtx));
        b->copy_from(gko::lend(sys_1.b));
        x->copy_from(gko::lend(res.x));
        auto d_left = BDense::create(cuexec);
        auto d_right = BDense::create(cuexec);
        if (left_scale) {
            d_left->copy_from(left_scale);
        }
        if (right_scale) {
            d_right->copy_from(right_scale);
        }
        auto d_left_ptr = left_scale ? d_left.get() : nullptr;
        auto d_right_ptr = right_scale ? d_right.get() : nullptr;

        gko::kernels::cuda::batch_rich::apply<value_type>(
            this->cuexec, opts, mtx.get(), d_left_ptr, d_right_ptr, b.get(),
            x.get(), logdata);

        res.x->copy_from(gko::lend(x));
        auto rnorms =
            compute_residual_norm(sys_1.mtx.get(), res.x.get(), sys_1.b.get());

        res.logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->exec, sizes);
        res.logdata.iter_counts.set_executor(this->exec);
        res.logdata.res_norms->copy_from(logdata.res_norms.get());
        res.logdata.iter_counts = logdata.iter_counts;

        res.resnorm = std::move(rnorms);
        return std::move(res);
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

    Result solve_poisson_uniform_mult()
    {
        Result res;
        res.x = gko::batch_initialize<BDense>(
            nbatch,
            std::initializer_list<std::initializer_list<value_type>>{
                {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
            this->exec);

        std::vector<gko::dim<2>> sizes(nbatch, gko::dim<2>(1, nrhs));
        gko::log::BatchLogData<value_type> logdata;
        logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->cuexec, sizes);
        logdata.iter_counts.set_executor(this->cuexec);
        logdata.iter_counts.resize_and_reset(nrhs * nbatch);

        auto mtx = Mtx::create(this->cuexec);
        auto b = BDense::create(this->cuexec);
        auto x = BDense::create(this->cuexec);
        mtx->copy_from(gko::lend(sys_m.mtx));
        b->copy_from(gko::lend(sys_m.b));
        x->copy_from(gko::lend(res.x));

        gko::kernels::cuda::batch_rich::apply<value_type>(
            this->cuexec, opts_m, mtx.get(), nullptr, nullptr, b.get(), x.get(),
            logdata);

        res.x->copy_from(gko::lend(x));
        auto rnorms =
            compute_residual_norm(sys_m.mtx.get(), res.x.get(), sys_m.b.get());

        res.logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->exec, sizes);
        res.logdata.iter_counts.set_executor(this->exec);
        res.logdata.res_norms->copy_from(logdata.res_norms.get());
        res.logdata.iter_counts = logdata.iter_counts;

        res.resnorm = std::move(rnorms);
        return res;
    }

    std::vector<int> multiple_iters_regression()
    {
        std::vector<int> iters(2);
        if (std::is_same<real_type, float>::value) {
            iters[0] = 40;
            iters[1] = 48;
        } else if (std::is_same<real_type, double>::value) {
            iters[0] = 98;
            iters[1] = 106;
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
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(this->r_1.resnorm->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  this->opts_1.rel_residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(this->r_1.x, this->sys_1.xex,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, StencilSystemJacobiLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    const int ref_iters = this->single_iters_regression();

    const int *const iter_array =
        this->r_1.logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->r_1.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        ASSERT_EQ(iter_array[i], ref_iters);
        ASSERT_LE(res_log_array[i] / this->sys_1.bnorm->get_const_values()[i],
                  this->opts_1.rel_residual_tol);
        // The following is satisfied for float but not for double - why?
        ASSERT_NEAR(res_log_array[i] / this->sys_1.bnorm->get_const_values()[i],
                    this->r_1.resnorm->get_const_values()[i] /
                        this->sys_1.bnorm->get_const_values()[i],
                    10 * r<value_type>::value);
    }
}


TYPED_TEST(BatchRich, SolvesStencilMultipleSystemJacobi)
{
    for (size_t i = 0; i < this->nbatch; i++) {
        for (size_t j = 0; j < this->nrhs; j++) {
            ASSERT_LE(
                this->r_m.resnorm->get_const_values()[i * this->nrhs + j] /
                    this->sys_m.bnorm->get_const_values()[i * this->nrhs + j],
                this->opts_m.rel_residual_tol);
        }
    }
    GKO_ASSERT_BATCH_MTX_NEAR(this->r_m.x, this->sys_m.xex,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, StencilMultipleSystemJacobiLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    const std::vector<int> ref_iters = this->multiple_iters_regression();

    const int *const iter_array =
        this->r_m.logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->r_m.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        for (size_t j = 0; j < this->nrhs; j++) {
            ASSERT_EQ(iter_array[i * this->nrhs + j], ref_iters[j]);
            ASSERT_LE(
                res_log_array[i * this->nrhs + j] /
                    this->sys_m.bnorm->get_const_values()[i * this->nrhs + j],
                this->opts_m.rel_residual_tol);
            ASSERT_NEAR(
                res_log_array[i] / this->sys_m.bnorm->get_const_values()[i],
                this->r_m.resnorm->get_const_values()[i] /
                    this->sys_m.bnorm->get_const_values()[i],
                10 * r<value_type>::value);
        }
    }
}


TYPED_TEST(BatchRich, BetterRelaxationFactorGivesBetterConvergence)
{
    using Result = typename TestFixture::Result;
    using BDense = typename TestFixture::BDense;
    using Options = typename TestFixture::Options;
    const Options opts{gpb::type::jacobi, 1000, 1e-8, 1.0};
    const Options opts_slower{gpb::type::jacobi, 1000, 1e-8, 0.8};

    Result result1 = this->solve_poisson_uniform_1(opts);
    Result result2 = this->solve_poisson_uniform_1(opts_slower);

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
    using LinSys = typename TestFixture::LinSys;
    auto useexec = this->cuexec;
    std::unique_ptr<typename Solver::Factory> batchrich_factory =
        Solver::build()
            .with_max_iterations(100)
            .with_rel_residual_tol(5e-7f)
            .with_preconditioner(gpb::type::jacobi)
            .on(useexec);
    const int nrhs_1 = 1;
    const size_t nbatch = 3;
    const LinSys sys = this->get_poisson_problem(nrhs_1, nbatch);
    auto rx =
        gko::batch_initialize<BDense>(nbatch, {0.0, 0.0, 0.0}, this->exec);
    std::unique_ptr<Mtx> mtx = Mtx::create(useexec);
    auto b = BDense::create(useexec);
    auto x = BDense::create(useexec);
    mtx->copy_from(gko::lend(sys.mtx));
    b->copy_from(gko::lend(sys.b));
    x->copy_from(gko::lend(rx));

    std::unique_ptr<Solver> solver =
        batchrich_factory->generate(gko::give(mtx));
    solver->apply(b.get(), x.get());

    rx->copy_from(gko::lend(x));
    const auto rnorms =
        this->compute_residual_norm(sys.mtx.get(), rx.get(), sys.b.get());
    for (size_t i = 0; i < nbatch; i++) {
        ASSERT_LE(
            rnorms->get_const_values()[i] / sys.bnorm->get_const_values()[i],
            5e-7);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(rx, sys.xex, 1e-5 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, UnitScalingDoesNotChangeResult)
{
    using Result = typename TestFixture::Result;
    using BDense = typename TestFixture::BDense;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);

    Result result = this->solve_poisson_uniform_1(
        this->opts_1, left_scale.get(), right_scale.get());

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(result.resnorm->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  this->opts_1.rel_residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, GeneralScalingDoesNotChangeResult)
{
    using Result = typename TestFixture::Result;
    using BDense = typename TestFixture::BDense;
    using value_type = typename TestFixture::value_type;
    auto left_scale = gko::batch_initialize<BDense>(
        {{0.8, 0.9, 0.95}, {1.1, 3.2, 0.9}}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.5, 1.05}, this->exec);

    Result result = this->solve_poisson_uniform_1(
        this->opts_1, left_scale.get(), right_scale.get());

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_LE(result.resnorm->get_const_values()[i] /
                      this->sys_1.bnorm->get_const_values()[i],
                  3 * this->opts_1.rel_residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex,
                              1e-5 /*r<value_type>::value*/);
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
    const int maxits = 10000;
    const int nrows = 31;
    const size_t nbatch = 3;
    const int nrhs = 3;
    auto batchrich_factory = Solver::build()
                                 .with_max_iterations(maxits)
                                 .with_rel_residual_tol(tol)
                                 .with_relaxation_factor(RT{0.95})
                                 .on(exec);
    gko::test::test_solve<Solver>(exec, nbatch, nrows, nrhs, tol, maxits,
                                  batchrich_factory.get(), 2);
}

}  // namespace
