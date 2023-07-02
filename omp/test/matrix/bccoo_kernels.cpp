/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

    void set_up_apply_data_ind(int num_vectors = 1)
    {
        mtx_ind =
            Mtx::create(ref, 0, gko::matrix::bccoo::compression::individual);
        mtx_ind->move_from(gen_mtx(mtx_size[0], mtx_size[1], 1));
        expected = gen_mtx(mtx_size[0], num_vectors, 1);
        y = gen_mtx(mtx_size[1], num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx_ind = gko::clone(omp, mtx_ind);
        dresult = gko::clone(omp, expected);
        dy = gko::clone(omp, y);
        dalpha = gko::clone(omp, alpha);
        dbeta = gko::clone(omp, beta);
    }

    void set_up_apply_data_grp(int num_vectors = 1)
    {
        mtx_grp = Mtx::create(ref, 0, gko::matrix::bccoo::compression::group);
        mtx_grp->move_from(gen_mtx(mtx_size[0], mtx_size[1], 1));
        expected = gen_mtx(mtx_size[0], num_vectors, 1);
        y = gen_mtx(mtx_size[1], num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx_grp = gko::clone(omp, mtx_grp);
        dresult = gko::clone(omp, expected);
        dy = gko::clone(omp, y);
        dalpha = gko::clone(omp, alpha);
        dbeta = gko::clone(omp, beta);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx_ind;
    std::unique_ptr<Mtx> mtx_grp;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx_ind;
    std::unique_ptr<Mtx> dmtx_grp;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Bccoo, SimpleApplyIsEquivalentToRefInd)
{
    set_up_apply_data_ind();

    mtx_ind->apply(y.get(), expected.get());
    dmtx_ind->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();

    mtx_grp->apply(y.get(), expected.get());
    dmtx_grp->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyIsEquivalentToRefInd)
{
    set_up_apply_data_ind();

    mtx_ind->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_ind->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();

    mtx_grp->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_grp->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddIsEquivalentToRefInd)
{
    set_up_apply_data_ind();

    mtx_ind->apply2(y.get(), expected.get());
    dmtx_ind->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();

    mtx_grp->apply2(y.get(), expected.get());
    dmtx_grp->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddIsEquivalentToRefInd)
{
    set_up_apply_data_ind();

    mtx_ind->apply2(alpha.get(), y.get(), expected.get());
    dmtx_ind->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();

    mtx_grp->apply2(alpha.get(), y.get(), expected.get());
    dmtx_grp->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyToDenseMatrixIsEquivalentToRefInd)
{
    set_up_apply_data_ind(3);

    mtx_ind->apply(y.get(), expected.get());
    dmtx_ind->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyToDenseMatrixIsEquivalentToRefGrp)
{
    set_up_apply_data_grp(3);

    mtx_grp->apply(y.get(), expected.get());
    dmtx_grp->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyToDenseMatrixIsEquivalentToRefInd)
{
    set_up_apply_data_ind(3);

    mtx_ind->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_ind->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyToDenseMatrixIsEquivalentToRefGrp)
{
    set_up_apply_data_grp(3);

    mtx_grp->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_grp->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddToDenseMatrixIsEquivalentToRefInd)
{
    set_up_apply_data_ind(3);

    mtx_ind->apply2(y.get(), expected.get());
    dmtx_ind->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, SimpleApplyAddToDenseMatrixIsEquivalentToRefGrp)
{
    set_up_apply_data_grp(3);

    mtx_grp->apply2(y.get(), expected.get());
    dmtx_grp->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddToDenseMatrixIsEquivalentToRefInd)
{
    set_up_apply_data_ind(3);

    mtx_ind->apply2(alpha.get(), y.get(), expected.get());
    dmtx_ind->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyAddToDenseMatrixIsEquivalentToRefGrp)
{
    set_up_apply_data_grp(3);

    mtx_grp->apply2(alpha.get(), y.get(), expected.get());
    dmtx_grp->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Bccoo, ApplyToComplexIsEquivalentToRefInd)
{
    set_up_apply_data_ind();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_ind->apply(complex_b.get(), complex_x.get());
    dmtx_ind->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, ApplyToComplexIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_grp->apply(complex_b.get(), complex_x.get());
    dmtx_grp->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyToComplexIsEquivalentToRefInd)
{
    set_up_apply_data_ind();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_ind->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx_ind->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(),
                    dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, AdvancedApplyToComplexIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_grp->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx_grp->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(),
                    dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, ApplyAddToComplexIsEquivalentToRefInd)
{
    set_up_apply_data_ind();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_ind->apply2(alpha.get(), complex_b.get(), complex_x.get());
    dmtx_ind->apply2(dalpha.get(), dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, ApplyAddToComplexIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    mtx_grp->apply2(alpha.get(), complex_b.get(), complex_x.get());
    dmtx_grp->apply2(dalpha.get(), dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Bccoo, ConvertToDenseIsEquivalentToRefInd)
{
    set_up_apply_data_ind();
    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<>::create(omp);

    mtx_ind->convert_to(dense_mtx.get());
    dmtx_ind->convert_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToDenseIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();
    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<>::create(omp);

    mtx_grp->convert_to(dense_mtx.get());
    dmtx_grp->convert_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToPrecisionIsEquivalentToRefInd)
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

    set_up_apply_data_ind();

    mtx_ind->move_to(tmp.get());
    dmtx_ind->move_to(dtmp.get());

    GKO_ASSERT_MTX_NEAR(tmp, dtmp, residual);
}


TEST_F(Bccoo, ConvertToPrecisionIsEquivalentToRefGrp)
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

    set_up_apply_data_grp();

    mtx_grp->move_to(tmp.get());
    dmtx_grp->move_to(dtmp.get());

    GKO_ASSERT_MTX_NEAR(tmp, dtmp, residual);
}


TEST_F(Bccoo, ConvertToCooIsEquivalentToRefInd)
{
    set_up_apply_data_ind();
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(omp);

    mtx_ind->convert_to(coo_mtx.get());
    dmtx_ind->convert_to(dcoo_mtx.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToCooIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(omp);

    mtx_grp->convert_to(coo_mtx.get());
    dmtx_grp->convert_to(dcoo_mtx.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToCsrIsEquivalentToRefInd)
{
    set_up_apply_data_ind();
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(omp);

    mtx_ind->convert_to(csr_mtx.get());
    dmtx_ind->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ConvertToCsrIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(omp);

    mtx_grp->convert_to(csr_mtx.get());
    dmtx_grp->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 1e-14);
}


TEST_F(Bccoo, ExtractDiagonalIsEquivalentToRefInd)
{
    set_up_apply_data_ind();

    auto diag = mtx_ind->extract_diagonal();
    auto ddiag = dmtx_ind->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Bccoo, ExtractDiagonalIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();

    auto diag = mtx_grp->extract_diagonal();
    auto ddiag = dmtx_grp->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Bccoo, InplaceAbsoluteMatrixIsEquivalentToRefInd)
{
    set_up_apply_data_ind();

    mtx_ind->compute_absolute_inplace();
    dmtx_ind->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx_ind, dmtx_ind, 1e-14);
}


TEST_F(Bccoo, InplaceAbsoluteMatrixIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();

    mtx_grp->compute_absolute_inplace();
    dmtx_grp->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx_grp, dmtx_grp, 1e-14);
}


TEST_F(Bccoo, OutplaceAbsoluteMatrixIsEquivalentToRefInd)
{
    set_up_apply_data_ind();

    auto abs_mtx_ind = mtx_ind->compute_absolute();
    auto dabs_mtx_ind = dmtx_ind->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_ind, dabs_mtx_ind, 1e-14);
}


TEST_F(Bccoo, OutplaceAbsoluteMatrixIsEquivalentToRefGrp)
{
    set_up_apply_data_grp();

    auto abs_mtx_grp = mtx_grp->compute_absolute();
    auto dabs_mtx_grp = dmtx_grp->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_grp, dabs_mtx_grp, 1e-14);
}


}  // namespace
