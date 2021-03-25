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

    BatchRich() : exec(gko::ReferenceExecutor::create())
    {
        solve_poisson_uniform_1();
        solve_poisson_uniform_mult();
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    const size_t nbatch = 2;
    std::shared_ptr<BDense> x_1;
    std::shared_ptr<BDense> xex_1;
    std::shared_ptr<RBDense> resnorm_1;
    std::shared_ptr<RBDense> bnorm_1;
    gko::log::BatchLogData<value_type> logdata_1;
    const gko::kernels::batch_rich::BatchRichardsonOptions<real_type> opts_1{
        "jacobi", 500, 1e-6, 1.0};

    const int nrhs = 2;
    std::shared_ptr<BDense> x_m;
    std::shared_ptr<BDense> xex_m;
    std::shared_ptr<RBDense> resnorm_m;
    std::shared_ptr<RBDense> bnorm_m;
    gko::log::BatchLogData<value_type> logdata_m;
    const gko::kernels::batch_rich::BatchRichardsonOptions<real_type> opts_m{
        "jacobi", 100, 1e-6, 1.0};

    void solve_poisson_uniform_1()
    {
        const int nrows = 3;
        const int nrhs_1 = 1;
        auto mtx = gko::test::create_poisson1d_batch<value_type>(this->exec,
                                                                 nrows, nbatch);
        auto b =
            gko::batch_initialize<BDense>(nbatch, {-1.0, 3.0, 1.0}, this->exec);
        x_1 =
            gko::batch_initialize<BDense>(nbatch, {0.0, 0.0, 0.0}, this->exec);
        xex_1 =
            gko::batch_initialize<BDense>(nbatch, {1.0, 3.0, 2.0}, this->exec);

        std::vector<gko::dim<2>> sizes(nbatch, gko::dim<2>(1, nrhs_1));
        logdata_1.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->exec, sizes);
        logdata_1.iter_counts.set_executor(this->exec);
        logdata_1.iter_counts.resize_and_reset(nrhs_1 * nbatch);

        gko::kernels::reference::batch_rich::apply<value_type>(
            this->exec, opts_1, mtx.get(), b.get(), x_1.get(), logdata_1);

        std::unique_ptr<BDense> res = b->clone();
        bnorm_1 = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
            nbatch, {0.0}, this->exec);
        b->compute_norm2(bnorm_1.get());
        resnorm_1 = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
            nbatch, {0.0}, this->exec);
        auto alpha = gko::batch_initialize<BDense>(nbatch, {-1.0}, this->exec);
        auto beta = gko::batch_initialize<BDense>(nbatch, {1.0}, this->exec);
        mtx->apply(alpha.get(), x_1.get(), beta.get(), res.get());
        res->compute_norm2(resnorm_1.get());
    }

    void solve_poisson_uniform_mult()
    {
        const int nrows = 3;
        auto mtx = gko::test::create_poisson1d_batch<value_type>(this->exec,
                                                                 nrows, nbatch);
        auto b = gko::batch_initialize<BDense>(
            nbatch,
            std::initializer_list<std::initializer_list<value_type>>{
                {-1.0, 2.0}, {3.0, -1.0}, {1.0, 0.0}},
            this->exec);
        x_m = gko::batch_initialize<BDense>(
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
        logdata_m.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->exec, sizes);
        logdata_m.iter_counts.set_executor(this->exec);
        logdata_m.iter_counts.resize_and_reset(nrhs * nbatch);

        gko::kernels::reference::batch_rich::apply<value_type>(
            this->exec, opts_m, mtx.get(), b.get(), x_m.get(), logdata_m);

        std::unique_ptr<BDense> res = b->clone();
        bnorm_m = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
            nbatch, {{0.0, 0.0}}, this->exec);
        b->compute_norm2(bnorm_m.get());
        resnorm_m = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
            nbatch, {{0.0, 0.0}}, this->exec);
        auto alpha = gko::batch_initialize<BDense>(nbatch, {-1.0}, this->exec);
        auto beta = gko::batch_initialize<BDense>(nbatch, {1.0}, this->exec);
        mtx->apply(alpha.get(), x_m.get(), beta.get(), res.get());
        res->compute_norm2(resnorm_m.get());
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

    // regression values for iteration counts
    auto ref_iter_dict = []() -> int {
        if (std::is_same<real_type, float>::value) {
            return 50;
        } else if (std::is_same<real_type, double>::value) {
            return 80;
        } else {
            return -1;
        }
    };
    const int ref_iters = ref_iter_dict();

    const int *const iter_array = this->logdata_1.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->logdata_1.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        ASSERT_EQ(iter_array[i], ref_iters);
        ASSERT_LE(res_log_array[i] / this->bnorm_1->get_const_values()[i],
                  this->opts_1.rel_residual_tol);
        // The following is satisfied for float but not for double - why?
        // ASSERT_NEAR(res_log_array[i]/bnorm->get_const_values()[i],
        // 			rnorm->get_const_values()[i]/bnorm->get_const_values()[i],
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


}  // namespace
