/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/matrix/csr.hpp>


#include <random>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/test/utils.hpp>


namespace {


class Csr : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;
    using ComplexMtx = gko::matrix::Csr<std::complex<double>>;

    Csr() : rand_engine(42) {}

    void SetUp()
    {
        ASSERT_GT(gko::GpuExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        gpu = gko::GpuExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (gpu != nullptr) {
            ASSERT_NO_THROW(gpu->synchronize());
        }
    }

    std::unique_ptr<Vec> gen_mtx(int num_rows, int num_cols, int min_nnz_row)
    {
        return gko::test::generate_random_matrix<Vec>(
            ref, num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine);
    }

    void set_up_apply_data()
    {
        mtx = Mtx::create(ref);
        mtx->copy_from(gen_mtx(532, 231, 1));
        expected = gen_mtx(532, 1, 1);
        y = gen_mtx(231, 1, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(gpu);
        dmtx->copy_from(mtx.get());
        dresult = Vec::create(gpu);
        dresult->copy_from(expected.get());
        dy = Vec::create(gpu);
        dy->copy_from(y.get());
        dalpha = Vec::create(gpu);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(gpu);
        dbeta->copy_from(beta.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::GpuExecutor> gpu;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Csr, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, TransposeIsEquivalentToRef)
{
    set_up_apply_data();

    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    auto dense_result = gko::matrix::Dense<>::create(mtx->get_executor());
    auto dense_trans = gko::matrix::Dense<>::create(mtx->get_executor());

    auto result = Mtx::create(ref);
    result->copy_from(d_trans.get());

    auto trans_as_csr = static_cast<Mtx *>(trans.get());

    result->convert_to(dense_result.get());
    trans_as_csr->convert_to(dense_trans.get());
    ASSERT_MTX_NEAR(dense_result, dense_trans, 0);
}


TEST_F(Csr, ConjugateTransposeIsEquivalentToRef)
{
    set_up_apply_data();

    auto trans = mtx->conj_transpose();
    auto d_trans = dmtx->conj_transpose();

    auto dense_result = gko::matrix::Dense<>::create(mtx->get_executor());
    auto dense_trans = gko::matrix::Dense<>::create(mtx->get_executor());

    auto result = Mtx::create(ref);
    result->copy_from(d_trans.get());

    auto trans_as_csr = static_cast<Mtx *>(trans.get());

    result->convert_to(dense_result.get());
    trans_as_csr->convert_to(dense_trans.get());
    ASSERT_MTX_NEAR(dense_result, dense_trans, 0);
}


}  // namespace
