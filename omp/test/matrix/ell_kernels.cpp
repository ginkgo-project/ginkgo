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

#include <ginkgo/core/matrix/ell.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/ell_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class Ell : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Ell<>;
    using Vec = gko::matrix::Dense<>;
    using Vec2 = gko::matrix::Dense<float>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;

    Ell() : rand_engine(42) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown()
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    template <typename MtxType = Vec>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(int max_nonzeros_per_row = 0, int stride = 0,
                           int num_vectors = 1)
    {
        mtx = Mtx::create(ref, gko::dim<2>{}, max_nonzeros_per_row, stride);
        mtx->copy_from(gen_mtx(532, 231, 1));
        expected = gen_mtx(532, num_vectors, 1);
        expected2 = Vec2::create(ref);
        expected2->copy_from(expected.get());
        y = gen_mtx(231, num_vectors, 1);
        y2 = Vec2::create(ref);
        y2->copy_from(y.get());
        alpha = gko::initialize<Vec>({2.0}, ref);
        alpha2 = gko::initialize<Vec2>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        beta2 = gko::initialize<Vec2>({-1.0}, ref);
        dmtx = Mtx::create(omp);
        dmtx->copy_from(mtx.get());
        dresult = Vec::create(omp);
        dresult->copy_from(expected.get());
        dresult2 = Vec2::create(omp);
        dresult2->copy_from(expected2.get());
        dy = Vec::create(omp);
        dy->copy_from(y.get());
        dy2 = Vec2::create(omp);
        dy2->copy_from(y2.get());
        dalpha = Vec::create(omp);
        dalpha->copy_from(alpha.get());
        dalpha2 = Vec2::create(omp);
        dalpha2->copy_from(alpha2.get());
        dbeta = Vec::create(omp);
        dbeta->copy_from(beta.get());
        dbeta2 = Vec2::create(omp);
        dbeta2->copy_from(beta2.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec2> expected2;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec2> y2;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec2> alpha2;
    std::unique_ptr<Vec> beta;
    std::unique_ptr<Vec2> beta2;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec2> dresult2;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec2> dy2;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec2> dalpha2;
    std::unique_ptr<Vec> dbeta;
    std::unique_ptr<Vec2> dbeta2;
};


TEST_F(Ell, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyIsEquivalentToRef1)
{
    set_up_apply_data();

    mtx->apply(y2.get(), expected2.get());
    dmtx->apply(dy2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyIsEquivalentToRef2)
{
    set_up_apply_data();

    mtx->apply(y2.get(), expected.get());
    dmtx->apply(dy2.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyIsEquivalentToRef3)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected2.get());
    dmtx->apply(dy.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyIsEquivalentToRef1)
{
    set_up_apply_data();

    mtx->apply(alpha2.get(), y2.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyIsEquivalentToRef2)
{
    set_up_apply_data();

    mtx->apply(alpha2.get(), y2.get(), beta.get(), expected.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyIsEquivalentToRef3)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, SimpleApplyWithPaddingIsEquivalentToRef)
{
    set_up_apply_data(300, 600);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyWithPaddingIsEquivalentToRef1)
{
    set_up_apply_data(300, 600);

    mtx->apply(y2.get(), expected2.get());
    dmtx->apply(dy2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyWithPaddingIsEquivalentToRef2)
{
    set_up_apply_data(300, 600);

    mtx->apply(y2.get(), expected.get());
    dmtx->apply(dy2.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyWithPaddingIsEquivalentToRef3)
{
    set_up_apply_data(300, 600);

    mtx->apply(y.get(), expected2.get());
    dmtx->apply(dy.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, AdvancedApplyWithPaddingIsEquivalentToRef)
{
    set_up_apply_data(300, 600);
    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyWithPaddingIsEquivalentToRef1)
{
    set_up_apply_data(300, 600);

    mtx->apply(alpha2.get(), y2.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyWithPaddingIsEquivalentToRef2)
{
    set_up_apply_data(300, 600);

    mtx->apply(alpha2.get(), y2.get(), beta.get(), expected.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyWithPaddingIsEquivalentToRef3)
{
    set_up_apply_data(300, 600);

    mtx->apply(alpha.get(), y.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(0, 0, 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyToDenseMatrixIsEquivalentToRef1)
{
    set_up_apply_data(0, 0, 4);

    mtx->apply(y2.get(), expected2.get());
    dmtx->apply(dy2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyToDenseMatrixIsEquivalentToRef2)
{
    set_up_apply_data(0, 0, 5);

    mtx->apply(y2.get(), expected.get());
    dmtx->apply(dy2.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyToDenseMatrixIsEquivalentToRef3)
{
    set_up_apply_data(0, 0, 6);

    mtx->apply(y.get(), expected2.get());
    dmtx->apply(dy.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, AdvancedApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(0, 0, 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyToDenseMatrixIsEquivalentToRef1)
{
    set_up_apply_data(0, 0, 4);

    mtx->apply(alpha2.get(), y2.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyToDenseMatrixIsEquivalentToRef2)
{
    set_up_apply_data(0, 0, 5);

    mtx->apply(alpha2.get(), y2.get(), beta.get(), expected.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyToDenseMatrixIsEquivalentToRef3)
{
    set_up_apply_data(0, 0, 6);

    mtx->apply(alpha.get(), y.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, SimpleApplyWithPaddingToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(300, 600, 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyWithPaddingToDenseMatrixIsEquivalentToRef1)
{
    set_up_apply_data(300, 600, 3);

    mtx->apply(y2.get(), expected2.get());
    dmtx->apply(dy2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyWithPaddingToDenseMatrixIsEquivalentToRef2)
{
    set_up_apply_data(300, 600, 3);

    mtx->apply(y2.get(), expected.get());
    dmtx->apply(dy2.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyWithPaddingToDenseMatrixIsEquivalentToRef3)
{
    set_up_apply_data(300, 600, 3);

    mtx->apply(y.get(), expected2.get());
    dmtx->apply(dy.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, AdvancedApplyWithPaddingToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(300, 600, 3);
    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyWithPaddingToDenseMatrixIsEquivalentToRef1)
{
    set_up_apply_data(300, 600, 3);

    mtx->apply(alpha2.get(), y2.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyWithPaddingToDenseMatrixIsEquivalentToRef2)
{
    set_up_apply_data(300, 600, 3);

    mtx->apply(alpha2.get(), y2.get(), beta.get(), expected.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyWithPaddingToDenseMatrixIsEquivalentToRef3)
{
    set_up_apply_data(300, 600, 3);

    mtx->apply(alpha.get(), y.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Ell, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data(300, 600);
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx->apply(complex_b.get(), complex_x.get());
    dmtx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Ell, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data(300, 600);
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Ell, CountNonzerosIsEquivalentToRef)
{
    set_up_apply_data();

    gko::size_type nnz;
    gko::size_type dnnz;

    gko::kernels::reference::ell::count_nonzeros(ref, mtx.get(), &nnz);
    gko::kernels::omp::ell::count_nonzeros(omp, dmtx.get(), &dnnz);

    ASSERT_EQ(nnz, dnnz);
}


TEST_F(Ell, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data();

    auto diag = mtx->extract_diagonal();
    auto ddiag = dmtx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Ell, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->compute_absolute_inplace();
    dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 1e-14);
}


TEST_F(Ell, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_mtx = mtx->compute_absolute();
    auto dabs_mtx = dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, 1e-14);
}


}  // namespace
