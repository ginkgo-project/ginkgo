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
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    BatchRich() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
};

TYPED_TEST_SUITE(BatchRich, gko::test::ValueTypes);


TYPED_TEST(BatchRich, SolvesStencilSystemJacobi)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    using BDense = typename TestFixture::BDense;
    const int nbatch = 2;
    const int nrows = 3;
    auto mtx = gko::test::create_poisson1d_batch<value_type>(this->exec, nrows,
                                                             nbatch);
    auto b =
        gko::batch_initialize<BDense>(nbatch, {-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::batch_initialize<BDense>(nbatch, {0.0, 0.0, 0.0}, this->exec);
    auto xex =
        gko::batch_initialize<BDense>(nbatch, {1.0, 3.0, 2.0}, this->exec);

    const gko::kernels::batch_rich::BatchRichardsonOptions<real_type> opts{
        "jacobi", 500, 1e-6, 1.0};

    gko::kernels::reference::batch_rich::apply<value_type>(
        this->exec, opts, mtx.get(), b.get(), x.get());

    std::unique_ptr<BDense> res = b->clone();
    auto bnorm = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
        nbatch, {0.0}, this->exec);
    b->compute_norm2(bnorm.get());
    auto rnorm = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
        nbatch, {0.0}, this->exec);
    auto alpha = gko::batch_initialize<BDense>(nbatch, {-1.0}, this->exec);
    auto beta = gko::batch_initialize<BDense>(nbatch, {1.0}, this->exec);
    mtx->apply(alpha.get(), x.get(), beta.get(), res.get());
    res->compute_norm2(rnorm.get());

    for (size_t i = 0; i < mtx->get_num_batches(); i++) {
        ASSERT_LE(rnorm->get_const_values()[i] / bnorm->get_const_values()[i],
                  opts.rel_residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(x, xex, 1e-4 /*r<value_type>::value*/);
}


TYPED_TEST(BatchRich, SolvesStencilMultipleSystemJacobi)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    using BDense = typename TestFixture::BDense;
    const gko::size_type nbatch = 2;
    const int nrhs = 2;
    const int nrows = 3;
    auto mtx = gko::test::create_poisson1d_batch<value_type>(this->exec, nrows,
                                                             nbatch);
    auto b = gko::batch_initialize<BDense>(
        nbatch,
        std::initializer_list<std::initializer_list<value_type>>{
            {-1.0, 2.0}, {3.0, -1.0}, {1.0, 0.0}},
        this->exec);
    auto x = gko::batch_initialize<BDense>(
        nbatch,
        std::initializer_list<std::initializer_list<value_type>>{
            {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        this->exec);
    auto xex = gko::batch_initialize<BDense>(
        nbatch,
        std::initializer_list<std::initializer_list<value_type>>{
            {1.0, 1.0}, {3.0, 0.0}, {2.0, 0.0}},
        this->exec);

    const gko::kernels::batch_rich::BatchRichardsonOptions<real_type> opts{
        "jacobi", 100, 1e-6, 1.0};

    gko::kernels::reference::batch_rich::apply<value_type>(
        this->exec, opts, mtx.get(), b.get(), x.get());

    std::unique_ptr<BDense> res = b->clone();
    auto bnorm = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
        nbatch, {{0.0, 0.0}}, this->exec);
    b->compute_norm2(bnorm.get());
    auto rnorm = gko::batch_initialize<gko::matrix::BatchDense<real_type>>(
        nbatch, {{0.0, 0.0}}, this->exec);
    auto alpha = gko::batch_initialize<BDense>(nbatch, {-1.0}, this->exec);
    auto beta = gko::batch_initialize<BDense>(nbatch, {1.0}, this->exec);
    mtx->apply(alpha.get(), x.get(), beta.get(), res.get());
    res->compute_norm2(rnorm.get());

    for (size_t i = 0; i < mtx->get_num_batches(); i++) {
        for (size_t j = 0; j < nrhs; j++) {
            ASSERT_LE(rnorm->get_const_values()[i * nrhs + j] /
                          bnorm->get_const_values()[i * nrhs + j],
                      opts.rel_residual_tol);
        }
    }
    GKO_ASSERT_BATCH_MTX_NEAR(x, xex, 1e-6 /*r<value_type>::value*/);
}


}  // namespace
