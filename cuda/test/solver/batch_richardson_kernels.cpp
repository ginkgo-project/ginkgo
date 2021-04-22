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

#include "core/solver/batch_richardson_kernels.hpp"


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/batch_richardson.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchRich : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    BatchRich()
        : exec(gko::ReferenceExecutor::create()),
          cuexec(gko::CudaExecutor::create(0, exec))
    {
        solve_poisson_uniform_1();
        cuexec->synchronize();
        solve_poisson_uniform_mult();
        cuexec->synchronize();
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::CudaExecutor> cuexec;

    const size_t nbatch = 2;
    const int nrows = 3;
    std::unique_ptr<const BDense> x_1;
    std::unique_ptr<const BDense> xex_1;
    std::unique_ptr<const RBDense> resnorm_1;
    std::unique_ptr<const RBDense> bnorm_1;
    gko::log::BatchLogData<value_type> logdata_1;
    const gko::kernels::batch_rich::BatchRichardsonOptions<real_type> opts_1{
        "jacobi", 500, 1e-6, 1.0};

    const int nrhs = 2;
    std::unique_ptr<const BDense> x_m;
    std::unique_ptr<const BDense> xex_m;
    std::unique_ptr<const RBDense> resnorm_m;
    std::unique_ptr<const RBDense> bnorm_m;
    gko::log::BatchLogData<value_type> logdata_m;
    const gko::kernels::batch_rich::BatchRichardsonOptions<real_type> opts_m{
        "jacobi", 100, 1e-6, 1.0};

    struct LinSys {
        std::unique_ptr<Mtx> mtx;
        std::unique_ptr<BDense> b;
        std::unique_ptr<BDense> xex;
    };

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
                    {-1.0, 2.0}, {3.0, -1.0}, {1.0, 0.0}},
                exec);
            sys.xex = gko::batch_initialize<BDense>(
                nbatches,
                std::initializer_list<std::initializer_list<value_type>>{
                    {1.0, 1.0}, {3.0, 0.0}, {2.0, 0.0}},
                exec);
        } else {
            GKO_NOT_IMPLEMENTED;
        }

        return sys;
    }

    struct Norms {
        std::unique_ptr<RBDense> r;
        std::unique_ptr<RBDense> b;
    };

    Norms compute_residual_norm(const Mtx *const rmtx, const BDense *const x,
                                const BDense *const b)
    {
        std::unique_ptr<BDense> res = b->clone();
        const size_t nbatches = x->get_num_batches();
        const int xnrhs = x->get_size().at(0)[1];
        const gko::batch_stride stride(nbatches, xnrhs);
        const gko::batch_dim normdim(nbatches, gko::dim<2>(1, xnrhs));
        Norms norms;
        norms.b = RBDense::create(exec, normdim);
        b->compute_norm2(norms.b.get());
        norms.r = RBDense::create(exec, normdim);
        auto alpha =
            gko::batch_initialize<BDense>(nbatches, {-1.0}, this->exec);
        auto beta = gko::batch_initialize<BDense>(nbatches, {1.0}, this->exec);
        rmtx->apply(alpha.get(), x, beta.get(), res.get());
        res->compute_norm2(norms.r.get());
        return norms;
    }

    void solve_poisson_uniform_1()
    {
        const int nrhs_1 = 1;
        auto rx =
            gko::batch_initialize<BDense>(nbatch, {0.0, 0.0, 0.0}, this->exec);
        const LinSys sys = get_poisson_problem(nrhs_1, nbatch);

        std::vector<gko::dim<2>> sizes(nbatch, gko::dim<2>(1, nrhs_1));
        gko::log::BatchLogData<value_type> logdata;
        logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->cuexec, sizes);
        logdata.iter_counts.set_executor(this->cuexec);
        logdata.iter_counts.resize_and_reset(nrhs_1 * nbatch);

        auto mtx = Mtx::create(this->cuexec);
        auto b = BDense::create(this->cuexec);
        auto x = BDense::create(this->cuexec);
        mtx->copy_from(gko::lend(sys.mtx));
        b->copy_from(gko::lend(sys.b));
        x->copy_from(gko::lend(rx));

        gko::kernels::cuda::batch_rich::apply<value_type>(
            this->cuexec, opts_1, mtx.get(), nullptr, nullptr, b.get(), x.get(),
            logdata);

        rx->copy_from(gko::lend(x));
        Norms norms =
            compute_residual_norm(sys.mtx.get(), rx.get(), sys.b.get());

        logdata_1.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->exec, sizes);
        logdata_1.iter_counts.set_executor(this->exec);
        logdata_1.res_norms->copy_from(logdata.res_norms.get());
        logdata_1.iter_counts = logdata.iter_counts;

        x_1 = std::move(rx);
        xex_1 = sys.xex->clone();
        resnorm_1 = std::move(norms.r);
        bnorm_1 = std::move(norms.b);
    }

    int single_iters_regression()
    {
        if (std::is_same<real_type, float>::value) {
            return 50;
        } else if (std::is_same<real_type, double>::value) {
            return 80;
        } else {
            return -1;
        }
    }

    void solve_poisson_uniform_mult()
    {
        auto rx = gko::batch_initialize<BDense>(
            nbatch,
            std::initializer_list<std::initializer_list<value_type>>{
                {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
            this->exec);
        const LinSys sys = get_poisson_problem(nrhs, nbatch);

        const gko::kernels::batch_rich::BatchRichardsonOptions<real_type> opts{
            "jacobi", 100, 1e-6, 1.0};
        std::vector<gko::dim<2>> sizes(nbatch, gko::dim<2>(1, nrhs));
        gko::log::BatchLogData<value_type> logdata;
        logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->cuexec, sizes);
        logdata.iter_counts.set_executor(this->cuexec);
        logdata.iter_counts.resize_and_reset(nrhs * nbatch);

        auto mtx = Mtx::create(this->cuexec);
        auto b = BDense::create(this->cuexec);
        auto x = BDense::create(this->cuexec);
        mtx->copy_from(gko::lend(sys.mtx));
        b->copy_from(gko::lend(sys.b));
        x->copy_from(gko::lend(rx));

        gko::kernels::cuda::batch_rich::apply<value_type>(
            this->cuexec, opts_m, mtx.get(), nullptr, nullptr, b.get(), x.get(),
            logdata);

        rx->copy_from(gko::lend(x));
        Norms norms =
            compute_residual_norm(sys.mtx.get(), rx.get(), sys.b.get());

        logdata_m.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->exec, sizes);
        logdata_m.iter_counts.set_executor(this->exec);
        logdata_m.res_norms->copy_from(logdata.res_norms.get());
        logdata_m.iter_counts = logdata.iter_counts;

        x_m = std::move(rx);
        xex_m = sys.xex->clone();
        resnorm_m = std::move(norms.r);
        bnorm_m = std::move(norms.b);
    }

    std::vector<int> multiple_iters_regression()
    {
        std::vector<int> iters(2);
        if (std::is_same<real_type, float>::value) {
            iters[0] = 50;
            iters[1] = 63;
        } else if (std::is_same<real_type, double>::value) {
            iters[0] = 80;
            iters[1] = 79;
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
        ASSERT_LE(this->resnorm_1->get_const_values()[i] /
                      this->bnorm_1->get_const_values()[i],
                  this->opts_1.rel_residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(this->x_1, this->xex_1,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, StencilSystemJacobiLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    const int ref_iters = this->single_iters_regression();

    const int *const iter_array = this->logdata_1.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->logdata_1.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        ASSERT_EQ(iter_array[i], ref_iters);
        ASSERT_LE(res_log_array[i] / this->bnorm_1->get_const_values()[i],
                  this->opts_1.rel_residual_tol);
        // The following is satisfied for float but not for double - why?
        // ASSERT_NEAR(res_log_array[i]/this->bnorm_1->get_const_values()[i],
        // 			this->resnorm_1->get_const_values()[i]/this->bnorm_1->get_const_values()[i],
        // 10*r<value_type>::value);
    }
}


TYPED_TEST(BatchRich, SolvesStencilMultipleSystemJacobi)
{
    for (size_t i = 0; i < this->nbatch; i++) {
        for (size_t j = 0; j < this->nrhs; j++) {
            ASSERT_LE(this->resnorm_m->get_const_values()[i * this->nrhs + j] /
                          this->bnorm_m->get_const_values()[i * this->nrhs + j],
                      this->opts_m.rel_residual_tol);
        }
    }
    GKO_ASSERT_BATCH_MTX_NEAR(this->x_m, this->xex_m,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, StencilMultipleSystemJacobiLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    const std::vector<int> ref_iters = this->multiple_iters_regression();

    const int *const iter_array = this->logdata_m.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->logdata_m.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        for (size_t j = 0; j < this->nrhs; j++) {
            ASSERT_EQ(iter_array[i * this->nrhs + j], ref_iters[j]);
            ASSERT_LE(res_log_array[i * this->nrhs + j] /
                          this->bnorm_m->get_const_values()[i * this->nrhs + j],
                      this->opts_m.rel_residual_tol);
            // The following is satisfied for float but not for double - why?
            // ASSERT_NEAR(res_log_array[i]/bnorm->get_const_values()[i],
            // 			rnorm->get_const_values()[i]/bnorm->get_const_values()[i],
            // 10*r<value_type>::value);
        }
    }
}


TYPED_TEST(BatchRich, CoreSolvesSystemJacobi)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using BDense = typename TestFixture::BDense;
    using Solver = gko::solver::BatchRichardson<value_type>;
    using LinSys = typename TestFixture::LinSys;
    using Norms = typename TestFixture::Norms;
    auto useexec = this->cuexec;
    std::unique_ptr<typename Solver::Factory> batchrich_factory =
        Solver::build()
            .with_max_iterations(100)
            .with_rel_residual_tol(1e-6f)
            .with_preconditioner("jacobi")
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
    const Norms norms =
        this->compute_residual_norm(sys.mtx.get(), rx.get(), sys.b.get());
    for (size_t i = 0; i < nbatch; i++) {
        ASSERT_LE(
            norms.r->get_const_values()[i] / norms.b->get_const_values()[i],
            1e-6);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(rx, sys.xex, 1e-5 /*r<value_type>::value*/);
}


}  // namespace
