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

#include "core/solver/batch_bicgstab_kernels.hpp"


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchBicgstab : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    BatchBicgstab()
        : exec(gko::ReferenceExecutor::create()),
          cuexec(gko::CudaExecutor::create(0, exec))
    {}

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::CudaExecutor> cuexec;

    const size_t nbatch = 1;

    std::unique_ptr<const BDense> x_1;
    std::unique_ptr<const BDense> xex_1;
    std::unique_ptr<const RBDense> resnorm_1;
    std::unique_ptr<const RBDense> bnorm_1;
    gko::log::BatchLogData<value_type> logdata_1;
    const gko::kernels::batch_bicgstab::BatchBicgstabOptions<real_type> opts_1{
        "none", 10, 1e-6, 1e-11, gko::stop::batch::ToleranceType::absolute};

    const int nrhs = 2;
    std::unique_ptr<const BDense> x_m;
    std::unique_ptr<const BDense> xex_m;
    std::unique_ptr<const RBDense> resnorm_m;
    std::unique_ptr<const RBDense> bnorm_m;
    gko::log::BatchLogData<value_type> logdata_m;
    const gko::kernels::batch_bicgstab::BatchBicgstabOptions<real_type> opts_m{
        "none", 10, 1e-6, 1e-11, gko::stop::batch::ToleranceType::absolute};

    void solve_poisson_uniform_1()
    {
        const int nrhs_1 = 1;
        const int nrows = 3;
        auto rmtx = gko::test::create_poisson1d_batch<value_type>(
            this->exec, nrows, nbatch);
        auto rb =
            gko::batch_initialize<BDense>(nbatch, {-1.0, 3.0, 1.0}, this->exec);
        auto rx =
            gko::batch_initialize<BDense>(nbatch, {0.0, 0.0, 0.0}, this->exec);
        xex_1 =
            gko::batch_initialize<BDense>(nbatch, {1.0, 3.0, 2.0}, this->exec);

        std::vector<gko::dim<2>> sizes(nbatch, gko::dim<2>(1, nrhs_1));
        gko::log::BatchLogData<value_type> logdata;
        logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->cuexec, sizes);
        logdata.iter_counts.set_executor(this->cuexec);
        logdata.iter_counts.resize_and_reset(nrhs_1 * nbatch);
        // for(int i = 0; i < nbatch*nrhs_1; i++) {
        // 	logdata.iter_counts.get_data()[i] = -1;
        // }

        auto mtx = Mtx::create(this->cuexec);
        auto b = BDense::create(this->cuexec);
        auto x = BDense::create(this->cuexec);
        mtx->copy_from(gko::lend(rmtx));
        b->copy_from(gko::lend(rb));
        x->copy_from(gko::lend(rx));

        gko::kernels::cuda::batch_bicgstab::apply<value_type>(
            this->cuexec, opts_1, mtx.get(), b.get(), x.get(), logdata);

        rx->copy_from(gko::lend(x));
        std::unique_ptr<BDense> res = rb->clone();
        auto bnorm = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
            nbatch, {0.0}, this->exec);
        rb->compute_norm2(bnorm.get());
        auto rnorm = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
            nbatch, {0.0}, this->exec);
        auto alpha = gko::batch_initialize<BDense>(nbatch, {-1.0}, this->exec);
        auto beta = gko::batch_initialize<BDense>(nbatch, {1.0}, this->exec);
        rmtx->apply(alpha.get(), rx.get(), beta.get(), res.get());
        res->compute_norm2(rnorm.get());

        logdata_1.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->exec, sizes);
        logdata_1.iter_counts.set_executor(this->exec);
        logdata_1.res_norms->copy_from(logdata.res_norms.get());
        logdata_1.iter_counts = logdata.iter_counts;

        x_1 = std::move(rx);
        resnorm_1 = std::move(rnorm);
        bnorm_1 = std::move(bnorm);
    }

    int single_iters_regression()
    {
        if (std::is_same<real_type, float>::value) {
            return 4;
        } else if (std::is_same<real_type, double>::value) {
            return 2;
        } else {
            return -1;
        }
    }

    void solve_poisson_uniform_mult()
    {
        const int nrows = 3;
        auto rmtx = gko::test::create_poisson1d_batch<value_type>(
            this->exec, nrows, nbatch);
        auto rb = gko::batch_initialize<BDense>(
            nbatch,
            std::initializer_list<std::initializer_list<value_type>>{
                {-1.0, 2.0}, {3.0, -1.0}, {1.0, 0.0}},
            this->exec);
        auto rx = gko::batch_initialize<BDense>(
            nbatch,
            std::initializer_list<std::initializer_list<value_type>>{
                {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
            this->exec);
        xex_m = gko::batch_initialize<BDense>(
            nbatch,
            std::initializer_list<std::initializer_list<value_type>>{
                {1.0, 1.0}, {3.0, 0.0}, {2.0, 0.0}},
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
        mtx->copy_from(gko::lend(rmtx));
        b->copy_from(gko::lend(rb));
        x->copy_from(gko::lend(rx));

        gko::kernels::cuda::batch_bicgstab::apply<value_type>(
            this->cuexec, opts_m, mtx.get(), b.get(), x.get(), logdata);

        rx->copy_from(gko::lend(x));
        std::unique_ptr<BDense> res = rb->clone();
        auto bnorm = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
            nbatch, {{0.0, 0.0}}, this->exec);
        rb->compute_norm2(bnorm.get());
        auto rnorm = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
            nbatch, {{0.0, 0.0}}, this->exec);
        auto alpha = gko::batch_initialize<BDense>(nbatch, {-1.0}, this->exec);
        auto beta = gko::batch_initialize<BDense>(nbatch, {1.0}, this->exec);
        rmtx->apply(alpha.get(), rx.get(), beta.get(), res.get());
        res->compute_norm2(rnorm.get());

        logdata_m.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->exec, sizes);
        logdata_m.iter_counts.set_executor(this->exec);
        logdata_m.res_norms->copy_from(logdata.res_norms.get());
        logdata_m.iter_counts = logdata.iter_counts;

        x_m = std::move(rx);
        resnorm_m = std::move(rnorm);
        bnorm_m = std::move(bnorm);
    }

    std::vector<int> multiple_iters_regression()
    {
        std::vector<int> iters(2);
        if (std::is_same<real_type, float>::value) {
            iters[0] = 4;
            iters[1] = 5;
        } else if (std::is_same<real_type, double>::value) {
            iters[0] = 2;
            iters[1] = 2;
        } else {
            iters[0] = -1;
            iters[1] = -1;
        }
        return iters;
    }
};

TYPED_TEST_SUITE(BatchBicgstab, gko::test::ValueTypes);


TYPED_TEST(BatchBicgstab, SolvesStencilSystem)
{
    this->solve_poisson_uniform_1();
    this->cuexec->synchronize();

    GKO_ASSERT_BATCH_MTX_NEAR(this->x_1, this->xex_1,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchBicgstab, StencilSystemLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    this->solve_poisson_uniform_1();
    this->cuexec->synchronize();

    const int ref_iters = this->single_iters_regression();

    const int *const iter_array = this->logdata_1.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->logdata_1.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        ASSERT_EQ(iter_array[i], ref_iters);
        ASSERT_LE(res_log_array[i], this->opts_1.abs_residual_tol);
    }
}


TYPED_TEST(BatchBicgstab, SolvesStencilMultipleSystem)
{
    this->solve_poisson_uniform_mult();
    this->cuexec->synchronize();

    GKO_ASSERT_BATCH_MTX_NEAR(this->x_m, this->xex_m,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchBicgstab, StencilMultipleSystemLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;


    this->solve_poisson_uniform_mult();
    this->cuexec->synchronize();

    const std::vector<int> ref_iters = this->multiple_iters_regression();

    const int *const iter_array = this->logdata_m.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->logdata_m.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        for (size_t j = 0; j < this->nrhs; j++) {
            ASSERT_EQ(iter_array[i * this->nrhs + j], ref_iters[j]);
            ASSERT_LE(res_log_array[i * this->nrhs + j],
                      this->opts_m.abs_residual_tol);
        }
    }
}


}  // namespace
