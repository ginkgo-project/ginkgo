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
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/bccoo_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"


namespace {


class Bccoo : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Bccoo<>;
    using Coo = gko::matrix::Coo<>;
    using Vec = gko::matrix::Dense<>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;

    Bccoo() : mtx_size(532, 231), rand_engine(42) {}

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
    /*
        void set_up_apply_data(int num_vectors = 1)
        {
            mtx = Mtx::create(ref);
            mtx->copy_from(gen_mtx(mtx_size[0], mtx_size[1], 1));
            expected = gen_mtx(mtx_size[0], num_vectors, 1);
            y = gen_mtx(mtx_size[1], num_vectors, 1);
            alpha = gko::initialize<Vec>({2.0}, ref);
            beta = gko::initialize<Vec>({-1.0}, ref);
            dmtx = gko::clone(omp, mtx);
            dresult = gko::clone(omp, expected);
            dy = gko::clone(omp, y);
            dalpha = gko::clone(omp, alpha);
            dbeta = gko::clone(omp, beta);
        }
    */
    void set_up_apply_data_elm(int num_vectors = 1)
    {
        mtx_elm = Mtx::create(ref, 0, gko::matrix::bccoo::compression::element);
        mtx_elm->copy_from(gen_mtx(mtx_size[0], mtx_size[1], 1));
        expected = gen_mtx(mtx_size[0], num_vectors, 1);
        y = gen_mtx(mtx_size[1], num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx_elm = gko::clone(omp, mtx_elm);
        dresult = gko::clone(omp, expected);
        dy = gko::clone(omp, y);
        dalpha = gko::clone(omp, alpha);
        dbeta = gko::clone(omp, beta);
    }

    void set_up_apply_data_blk(int num_vectors = 1)
    {
        mtx_blk = Mtx::create(ref, 0, gko::matrix::bccoo::compression::block);
        mtx_blk->copy_from(gen_mtx(mtx_size[0], mtx_size[1], 1));
        expected = gen_mtx(mtx_size[0], num_vectors, 1);
        y = gen_mtx(mtx_size[1], num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx_blk = gko::clone(omp, mtx_blk);
        dresult = gko::clone(omp, expected);
        dy = gko::clone(omp, y);
        dalpha = gko::clone(omp, alpha);
        dbeta = gko::clone(omp, beta);
    }

    struct matrix_pair {
        std::unique_ptr<Mtx> ref;
        std::unique_ptr<Mtx> omp;
    };

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx_elm;
    std::unique_ptr<Mtx> mtx_blk;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Mtx> dmtx_elm;
    std::unique_ptr<Mtx> dmtx_blk;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};

/*
TEST_F(Bccoo, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}
*/

TEST_F(Bccoo, SimpleApplyIsEquivalentToRefElm)
{
    set_up_apply_data_elm();

    mtx_elm->apply(y.get(), expected.get());
    dmtx_elm->apply(dy.get(), dresult.get());

    if (mtx_elm->use_block_compression()) {
        std::cout << "Elm_ref_error: " << std::endl;
    }
    if (dmtx_elm->use_block_compression()) {
        std::cout << "Elm__omp_error: " << std::endl;
    }
    // std::cout << "Elm_ref: " << mtx_elm->get_block_size() << std::endl;
    // std::cout << "Elm_omp: " << dmtx_elm->get_block_size() << std::endl;

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();

    mtx_blk->apply(y.get(), expected.get());
    dmtx_blk->apply(dy.get(), dresult.get());

    if (mtx_blk->use_element_compression()) {
        std::cout << "Blk_ref_error: " << std::endl;
    }
    if (dmtx_blk->use_element_compression()) {
        std::cout << "Blk__omp_error: " << std::endl;
    }
    // std::cout << "Blk_ref: " << mtx_blk->get_block_size() << std::endl;
    // std::cout << "Blk_omp: " << dmtx_blk->get_block_size() << std::endl;

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyIsEquivalentToRefElm)
{
    set_up_apply_data_elm();

    mtx_elm->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_elm->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();

    mtx_blk->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_blk->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddIsEquivalentToRefElm)
{
    set_up_apply_data_elm();

    mtx_elm->apply2(y.get(), expected.get());
    dmtx_elm->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();

    mtx_blk->apply2(y.get(), expected.get());
    dmtx_blk->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddIsEquivalentToRefElm)
{
    set_up_apply_data_elm();

    mtx_elm->apply2(alpha.get(), y.get(), expected.get());
    dmtx_elm->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();

    mtx_blk->apply2(alpha.get(), y.get(), expected.get());
    dmtx_blk->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyToDenseMatrixIsEquivalentToRefElm)
{
    set_up_apply_data_elm(3);

    mtx_elm->apply(y.get(), expected.get());
    dmtx_elm->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyToDenseMatrixIsEquivalentToRefBlk)
{
    set_up_apply_data_blk(3);

    mtx_blk->apply(y.get(), expected.get());
    dmtx_blk->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyToDenseMatrixIsEquivalentToRefElm)
{
    set_up_apply_data_elm(3);

    mtx_elm->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_elm->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyToDenseMatrixIsEquivalentToRefBlk)
{
    set_up_apply_data_blk(3);

    mtx_blk->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_blk->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddToDenseMatrixIsEquivalentToRefElm)
{
    set_up_apply_data_elm(3);

    mtx_elm->apply2(y.get(), expected.get());
    dmtx_elm->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddToDenseMatrixIsEquivalentToRefBlk)
{
    set_up_apply_data_blk(3);

    mtx_blk->apply2(y.get(), expected.get());
    dmtx_blk->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddToDenseMatrixIsEquivalentToRefElm)
{
    set_up_apply_data_elm(3);

    mtx_elm->apply2(alpha.get(), y.get(), expected.get());
    dmtx_elm->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddToDenseMatrixIsEquivalentToRefBlk)
{
    set_up_apply_data_blk(3);

    mtx_blk->apply2(alpha.get(), y.get(), expected.get());
    dmtx_blk->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, ApplyToComplexIsEquivalentToRefElm)
{
    set_up_apply_data_elm();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_elm->apply(complex_b.get(), complex_x.get());
    dmtx_elm->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, ApplyToComplexIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_blk->apply(complex_b.get(), complex_x.get());
    dmtx_blk->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyToComplexIsEquivalentToRefElm)
{
    set_up_apply_data_elm();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_elm->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx_elm->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(),
                    dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyToComplexIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_blk->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx_blk->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(),
                    dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, ApplyAddToComplexIsEquivalentToRefElm)
{
    set_up_apply_data_elm();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_elm->apply2(alpha.get(), complex_b.get(), complex_x.get());
    dmtx_elm->apply2(dalpha.get(), dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, ApplyAddToComplexIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_blk->apply2(alpha.get(), complex_b.get(), complex_x.get());
    dmtx_blk->apply2(dalpha.get(), dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, ConvertToDenseIsEquivalentToRefElm)
{
    set_up_apply_data_elm();
    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<>::create(omp);

    mtx_elm->convert_to(dense_mtx.get());
    dmtx_elm->convert_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToDenseIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();
    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<>::create(omp);

    mtx_blk->convert_to(dense_mtx.get());
    dmtx_blk->convert_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToPrecisionIsEquivalentToRefElm)
{
    using ValueType = Mtx::value_type;
    using IndexType = Mtx::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto tmp = OtherBccoo::create(ref);
    auto dtmp = OtherBccoo::create(omp);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    set_up_apply_data_elm();

    mtx_elm->move_to(tmp.get());
    dmtx_elm->move_to(dtmp.get());

    GKO_ASSERT_MTX_NEAR(tmp, dtmp, residual);
}


TEST_F(Bccoo, ConvertToPrecisionIsEquivalentToRefBlk)
{
    using ValueType = Mtx::value_type;
    using IndexType = Mtx::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Bccoo = Mtx;
    using OtherBccoo = gko::matrix::Bccoo<OtherType, IndexType>;
    auto tmp = OtherBccoo::create(ref);
    auto dtmp = OtherBccoo::create(omp);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    set_up_apply_data_blk();

    mtx_blk->move_to(tmp.get());
    dmtx_blk->move_to(dtmp.get());

    GKO_ASSERT_MTX_NEAR(tmp, dtmp, residual);
}


TEST_F(Bccoo, ConvertToCooIsEquivalentToRefElm)
{
    set_up_apply_data_elm();
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(omp);

    mtx_elm->convert_to(coo_mtx.get());
    dmtx_elm->convert_to(dcoo_mtx.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToCooIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(omp);

    mtx_blk->convert_to(coo_mtx.get());
    dmtx_blk->convert_to(dcoo_mtx.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToCsrIsEquivalentToRefElm)
{
    set_up_apply_data_elm();
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(omp);

    mtx_elm->convert_to(csr_mtx.get());
    dmtx_elm->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToCsrIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(omp);

    mtx_blk->convert_to(csr_mtx.get());
    dmtx_blk->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ExtractDiagonalIsEquivalentToRefElm)
{
    set_up_apply_data_elm();

    auto diag = mtx_elm->extract_diagonal();
    auto ddiag = dmtx_elm->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Bccoo, ExtractDiagonalIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();

    auto diag = mtx_blk->extract_diagonal();
    auto ddiag = dmtx_blk->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Bccoo, InplaceAbsoluteMatrixIsEquivalentToRefElm)
{
    set_up_apply_data_elm();

    mtx_elm->compute_absolute_inplace();
    dmtx_elm->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx_elm, dmtx_elm, 1e-14);
}


TEST_F(Bccoo, InplaceAbsoluteMatrixIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();

    mtx_blk->compute_absolute_inplace();
    dmtx_blk->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx_blk, dmtx_blk, 1e-14);
}


TEST_F(Bccoo, OutplaceAbsoluteMatrixIsEquivalentToRefElm)
{
    set_up_apply_data_elm();

    auto abs_mtx_elm = mtx_elm->compute_absolute();
    auto dabs_mtx_elm = dmtx_elm->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_elm, dabs_mtx_elm, 1e-14);
}


TEST_F(Bccoo, OutplaceAbsoluteMatrixIsEquivalentToRefBlk)
{
    set_up_apply_data_blk();

    auto abs_mtx_blk = mtx_blk->compute_absolute();
    auto dabs_mtx_blk = dmtx_blk->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_blk, dabs_mtx_blk, 1e-14);
}


}  // namespace
