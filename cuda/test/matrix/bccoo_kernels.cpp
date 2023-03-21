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

#include <ginkgo/core/matrix/bccoo.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/bccoo_kernels.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "cuda/test/utils.hpp"


namespace {


class Bccoo : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Bccoo<>;
    using Vec = gko::matrix::Dense<>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;

    Bccoo() : rand_engine(42) {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    template <typename MtxType = Vec>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols, std::uniform_int_distribution<>(1, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(int num_vectors = 1)
    {
        mtx = Mtx::create(ref);
        mtx->copy_from(gen_mtx(532, 231));
        expected = gen_mtx(532, num_vectors);
        y = gen_mtx(231, num_vectors);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(cuda);
        dmtx->copy_from(mtx.get());
        dresult = Vec::create(cuda);
        dresult->copy_from(expected.get());
        dy = Vec::create(cuda);
        dy->copy_from(y.get());
        dalpha = Vec::create(cuda);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(cuda);
        dbeta->copy_from(beta.get());
    }

    void set_up_apply_data_blk(int num_vectors = 1)
    {
        mtx_blk = Mtx::create(ref, 0, gko::matrix::bccoo::compression::block);
        mtx_blk->copy_from(gen_mtx(532, 231));
        expected = gen_mtx(532, num_vectors);
        y = gen_mtx(231, num_vectors);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        //        dmtx_blk = Mtx::create(cuda, 32,
        //        gko::matrix::bccoo::compression::block);
        dmtx_blk =
            Mtx::create(cuda, 128, gko::matrix::bccoo::compression::block);
        dmtx_blk->copy_from(mtx_blk.get());
        dresult = Vec::create(cuda);
        dresult->copy_from(expected.get());
        dy = Vec::create(cuda);
        dy->copy_from(y.get());
        dalpha = Vec::create(cuda);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(cuda);
        dbeta->copy_from(beta.get());
    }
    /*
        void unsort_mtx()
        {
            gko::test::unsort_matrix(mtx.get(), rand_engine);
            dmtx->copy_from(mtx.get());
        }
    */
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx_blk;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Mtx> dmtx_blk;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Bccoo, SimpleApplyIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk();

    mtx_blk->apply(y.get(), expected.get());
    dmtx_blk->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}

/*
TEST_F(Bccoo, SimpleApplyIsEquivalentToRefUnsorted)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    set_up_apply_data();
//    unsort_mtx();
//
//    mtx->apply(y.get(), expected.get());
//    dmtx->apply(dy.get(), dresult.get());
//
//    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
//}
*/

TEST_F(Bccoo, AdvancedApplyIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk();

    mtx_blk->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_blk->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk();

    mtx_blk->apply2(y.get(), expected.get());
    dmtx_blk->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk();

    mtx_blk->apply2(alpha.get(), y.get(), expected.get());
    dmtx_blk->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyToDenseMatrixIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk(3);

    mtx_blk->apply(y.get(), expected.get());
    dmtx_blk->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyToDenseMatrixIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk(3);

    mtx_blk->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_blk->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddToDenseMatrixIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk(3);

    mtx_blk->apply2(y.get(), expected.get());
    dmtx_blk->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddToLargeDenseMatrixIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk(33);

    mtx_blk->apply2(y.get(), expected.get());
    dmtx_blk->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddToDenseMatrixIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk(3);

    mtx_blk->apply2(alpha.get(), y.get(), expected.get());
    dmtx_blk->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddToLargeDenseMatrixIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk(33);

    mtx_blk->apply2(y.get(), expected.get());
    dmtx_blk->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, ApplyToComplexIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk();
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = ComplexVec::create(cuda);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = ComplexVec::create(cuda);
    dcomplex_x->copy_from(complex_x.get());

    mtx_blk->apply(complex_b.get(), complex_x.get());
    dmtx_blk->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);

    //    set_up_apply_data();
    //    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    //    auto dcomplex_b = ComplexVec::create(cuda);
    //    dcomplex_b->copy_from(complex_b.get());
    //    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    //    auto dcomplex_x = ComplexVec::create(cuda);
    //    dcomplex_x->copy_from(complex_x.get());
    //
    //    mtx->apply(complex_b.get(), complex_x.get());
    //    dmtx->apply(dcomplex_b.get(), dcomplex_x.get());
    //
    //    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyToComplexIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk();
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = ComplexVec::create(cuda);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = ComplexVec::create(cuda);
    dcomplex_x->copy_from(complex_x.get());

    mtx_blk->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx_blk->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(),
                    dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);

    //    set_up_apply_data();
    //    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    //    auto dcomplex_b = ComplexVec::create(cuda);
    //    dcomplex_b->copy_from(complex_b.get());
    //    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    //    auto dcomplex_x = ComplexVec::create(cuda);
    //    dcomplex_x->copy_from(complex_x.get());
    //
    //    mtx->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    //    dmtx->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(),
    //    dcomplex_x.get());
    //
    //    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, ApplyAddToComplexIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    set_up_apply_data_blk();
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = ComplexVec::create(cuda);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = ComplexVec::create(cuda);
    dcomplex_x->copy_from(complex_x.get());

    mtx_blk->apply2(alpha.get(), complex_b.get(), complex_x.get());
    dmtx_blk->apply2(dalpha.get(), dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);

    //    set_up_apply_data();
    //    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    //    auto dcomplex_b = ComplexVec::create(cuda);
    //    dcomplex_b->copy_from(complex_b.get());
    //    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    //    auto dcomplex_x = ComplexVec::create(cuda);
    //    dcomplex_x->copy_from(complex_x.get());
    //
    //    mtx->apply2(alpha.get(), complex_b.get(), complex_x.get());
    //    dmtx->apply2(dalpha.get(), dcomplex_b.get(), dcomplex_x.get());
    //
    //    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, ConvertToDenseIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    //    set_up_apply_data();
    set_up_apply_data_blk();
    auto dense_mtx_blk = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx_blk = gko::matrix::Dense<>::create(cuda);

    mtx_blk->convert_to(dense_mtx_blk.get());
    dmtx_blk->convert_to(ddense_mtx_blk.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx_blk.get(), ddense_mtx_blk.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToCooIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    //    set_up_apply_data();
    set_up_apply_data_blk();

    auto dense_mtx_blk = gko::matrix::Dense<>::create(ref);
    auto coo_mtx_blk = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx_blk = gko::matrix::Coo<>::create(cuda);

    mtx_blk->convert_to(dense_mtx_blk.get());
    dense_mtx_blk->convert_to(coo_mtx_blk.get());
    dmtx_blk->convert_to(dcoo_mtx_blk.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx_blk.get(), dcoo_mtx_blk.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToCsrIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    //    set_up_apply_data();
    set_up_apply_data_blk();

    auto dense_mtx_blk = gko::matrix::Dense<>::create(ref);
    auto csr_mtx_blk = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx_blk = gko::matrix::Csr<>::create(cuda);

    mtx_blk->convert_to(dense_mtx_blk.get());
    dense_mtx_blk->convert_to(csr_mtx_blk.get());
    dmtx_blk->convert_to(dcsr_mtx_blk.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx_blk.get(), dcsr_mtx_blk.get(), 1e-14);
}


TEST_F(Bccoo, ExtractDiagonalIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    //    set_up_apply_data();
    set_up_apply_data_blk();

    auto diag = mtx_blk->extract_diagonal();
    auto ddiag = dmtx_blk->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Bccoo, InplaceAbsoluteMatrixIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    //    set_up_apply_data();
    set_up_apply_data_blk();

    //		printf ("InPlace_00\n");
    mtx_blk->compute_absolute_inplace();
    //		printf ("InPlace_01\n");
    dmtx_blk->compute_absolute_inplace();
    //		printf ("InPlace_02\n");

    GKO_ASSERT_MTX_NEAR(mtx_blk, dmtx_blk, 1e-14);
    //		printf ("InPlace_03\n");
}


TEST_F(Bccoo, OutplaceAbsoluteMatrixIsEquivalentToRef)
// GKO_NOT_IMPLEMENTED;
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    //    set_up_apply_data();
    set_up_apply_data_blk();

    auto abs_mtx_blk = mtx_blk->compute_absolute();
    auto dabs_mtx_blk = dmtx_blk->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_blk, dabs_mtx_blk, 1e-14);
}


}  // namespace
