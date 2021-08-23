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
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/ell_kernels.hpp"
#include "core/test/utils.hpp"
#include "dpcpp/test/utils.hpp"


namespace {


class Ell : public ::testing::Test {
protected:
#if GINKGO_DPCPP_SINGLE_MODE
    using vtype = float;
#else
    using vtype = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE
    using Mtx = gko::matrix::Ell<vtype>;
    using Vec = gko::matrix::Dense<vtype>;
    using Vec2 = gko::matrix::Dense<float>;
    using ComplexVec = gko::matrix::Dense<std::complex<vtype>>;

    Ell()
        : rand_engine(42), size{532, 231}, num_els_rowwise{300}, ell_stride{600}
    {}

    void SetUp()
    {
        ASSERT_GT(gko::DpcppExecutor::get_num_devices("all"), 0);
        ref = gko::ReferenceExecutor::create();
        dpcpp = gko::DpcppExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (dpcpp != nullptr) {
            ASSERT_NO_THROW(dpcpp->synchronize());
        }
    }

    template <typename MtxType = Vec>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols, std::uniform_int_distribution<>(1, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(int num_rows = 532, int num_cols = 231,
                           int num_vectors = 1,
                           int num_stored_elements_per_row = 0, int stride = 0)
    {
        mtx = Mtx::create(ref, gko::dim<2>{}, num_stored_elements_per_row,
                          stride);
        mtx->copy_from(gen_mtx(num_rows, num_cols));
        expected = gen_mtx(num_rows, num_vectors);
        expected2 = Vec2::create(ref);
        expected2->copy_from(expected.get());
        y = gen_mtx(num_cols, num_vectors);
        y2 = Vec2::create(ref);
        y2->copy_from(y.get());
        alpha = gko::initialize<Vec>({2.0}, ref);
        alpha2 = gko::initialize<Vec2>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        beta2 = gko::initialize<Vec2>({-1.0}, ref);
        dmtx = Mtx::create(dpcpp);
        dmtx->copy_from(mtx.get());
        dresult = Vec::create(dpcpp);
        dresult->copy_from(expected.get());
        dresult2 = Vec2::create(dpcpp);
        dresult2->copy_from(expected2.get());
        dy = Vec::create(dpcpp);
        dy->copy_from(y.get());
        dy2 = Vec2::create(dpcpp);
        dy2->copy_from(y2.get());
        dalpha = Vec::create(dpcpp);
        dalpha->copy_from(alpha.get());
        dalpha2 = Vec2::create(dpcpp);
        dalpha2->copy_from(alpha2.get());
        dbeta = Vec::create(dpcpp);
        dbeta->copy_from(beta.get());
        dbeta2 = Vec2::create(dpcpp);
        dbeta2->copy_from(beta2.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::DpcppExecutor> dpcpp;

    std::ranlux48 rand_engine;
    gko::dim<2> size;
    gko::size_type num_els_rowwise;
    gko::size_type ell_stride;

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

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, MixedSimpleApplyIsEquivalentToRef1)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data();

    mtx->apply(y2.get(), expected2.get());
    dmtx->apply(dy2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, MixedSimpleApplyIsEquivalentToRef2)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data();

    mtx->apply(y2.get(), expected.get());
    dmtx->apply(dy2.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyIsEquivalentToRef3)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data();

    mtx->apply(y.get(), expected2.get());
    dmtx->apply(dy.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, MixedAdvancedApplyIsEquivalentToRef1)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data();

    mtx->apply(alpha2.get(), y2.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, MixedAdvancedApplyIsEquivalentToRef2)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data();

    mtx->apply(alpha2.get(), y2.get(), beta.get(), expected.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyIsEquivalentToRef3)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, SimpleApplyWithStrideIsEquivalentToRef)
{
    set_up_apply_data(size[0], size[1], 1, num_els_rowwise, ell_stride);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, MixedSimpleApplyWithStrideIsEquivalentToRef1)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 1, num_els_rowwise, ell_stride);

    mtx->apply(y2.get(), expected2.get());
    dmtx->apply(dy2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, MixedSimpleApplyWithStrideIsEquivalentToRef2)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 1, num_els_rowwise, ell_stride);

    mtx->apply(y2.get(), expected.get());
    dmtx->apply(dy2.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyWithStrideIsEquivalentToRef3)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 1, num_els_rowwise, ell_stride);

    mtx->apply(y.get(), expected2.get());
    dmtx->apply(dy.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, AdvancedApplyWithStrideIsEquivalentToRef)
{
    set_up_apply_data(size[0], size[1], 1, num_els_rowwise, ell_stride);
    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, MixedAdvancedApplyWithStrideIsEquivalentToRef1)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 1, num_els_rowwise, ell_stride);

    mtx->apply(alpha2.get(), y2.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, MixedAdvancedApplyWithStrideIsEquivalentToRef2)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 1, num_els_rowwise, ell_stride);

    mtx->apply(alpha2.get(), y2.get(), beta.get(), expected.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyWithStrideIsEquivalentToRef3)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 1, num_els_rowwise, ell_stride);

    mtx->apply(alpha.get(), y.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, SimpleApplyWithStrideToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(size[0], size[1], 3, num_els_rowwise, ell_stride);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, MixedSimpleApplyWithStrideToDenseMatrixIsEquivalentToRef1)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 3, num_els_rowwise, ell_stride);

    mtx->apply(y2.get(), expected2.get());
    dmtx->apply(dy2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, MixedSimpleApplyWithStrideToDenseMatrixIsEquivalentToRef2)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 3, num_els_rowwise, ell_stride);

    mtx->apply(y2.get(), expected.get());
    dmtx->apply(dy2.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedSimpleApplyWithStrideToDenseMatrixIsEquivalentToRef3)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 3, num_els_rowwise, ell_stride);

    mtx->apply(y.get(), expected2.get());
    dmtx->apply(dy.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, AdvancedApplyWithStrideToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(size[0], size[1], 3, num_els_rowwise, ell_stride);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, MixedAdvancedApplyWithStrideToDenseMatrixIsEquivalentToRef1)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 3, num_els_rowwise, ell_stride);

    mtx->apply(alpha2.get(), y2.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, MixedAdvancedApplyWithStrideToDenseMatrixIsEquivalentToRef2)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 3, num_els_rowwise, ell_stride);

    mtx->apply(alpha2.get(), y2.get(), beta.get(), expected.get());
    dmtx->apply(dalpha2.get(), dy2.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, MixedAdvancedApplyWithStrideToDenseMatrixIsEquivalentToRef3)
{
    SKIP_IF_SINGLE_MODE;
    set_up_apply_data(size[0], size[1], 3, num_els_rowwise, ell_stride);

    mtx->apply(alpha.get(), y.get(), beta2.get(), expected2.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta2.get(), dresult2.get());

    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-6);
}


TEST_F(Ell, SimpleApplyByAtomicIsEquivalentToRef)
{
    set_up_apply_data(10, 10000);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, AdvancedByAtomicApplyIsEquivalentToRef)
{
    set_up_apply_data(10, 10000);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, SimpleApplyByAtomicToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(10, 10000, 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, AdvancedByAtomicToDenseMatrixApplyIsEquivalentToRef)
{
    set_up_apply_data(10, 10000, 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, SimpleApplyOnSmallMatrixIsEquivalentToRef)
{
    set_up_apply_data(1, 10);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 5 * r<vtype>::value);
}


TEST_F(Ell, AdvancedApplyOnSmallMatrixToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(1, 10, 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, SimpleApplyOnSmallMatrixToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(1, 10, 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Ell, AdvancedApplyOnSmallMatrixIsEquivalentToRef)
{
    set_up_apply_data(1, 10);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value * 5);
}


TEST_F(Ell, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexVec>(size[1], 3);
    auto dcomplex_b = ComplexVec::create(dpcpp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(size[0], 3);
    auto dcomplex_x = ComplexVec::create(dpcpp);
    dcomplex_x->copy_from(complex_x.get());

    mtx->apply(complex_b.get(), complex_x.get());
    dmtx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<vtype>::value);
}


TEST_F(Ell, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexVec>(size[1], 3);
    auto dcomplex_b = ComplexVec::create(dpcpp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexVec>(size[0], 3);
    auto dcomplex_x = ComplexVec::create(dpcpp);
    dcomplex_x->copy_from(complex_x.get());

    mtx->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<vtype>::value);
}


TEST_F(Ell, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_data();

    auto dense_mtx = gko::matrix::Dense<vtype>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<vtype>::create(dpcpp);

    mtx->convert_to(dense_mtx.get());
    dmtx->convert_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), r<vtype>::value);
}


TEST_F(Ell, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data();

    auto csr_mtx = gko::matrix::Csr<vtype>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<vtype>::create(dpcpp);

    mtx->convert_to(csr_mtx.get());
    dmtx->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), r<vtype>::value);
}


TEST_F(Ell, CalculateNNZPerRowIsEquivalentToRef)
{
    set_up_apply_data();

    gko::Array<gko::size_type> nnz_per_row;
    nnz_per_row.set_executor(ref);
    nnz_per_row.resize_and_reset(mtx->get_size()[0]);

    gko::Array<gko::size_type> dnnz_per_row;
    dnnz_per_row.set_executor(dpcpp);
    dnnz_per_row.resize_and_reset(dmtx->get_size()[0]);

    gko::kernels::reference::ell::calculate_nonzeros_per_row(ref, mtx.get(),
                                                             &nnz_per_row);
    gko::kernels::dpcpp::ell::calculate_nonzeros_per_row(dpcpp, dmtx.get(),
                                                         &dnnz_per_row);

    auto tmp = gko::Array<gko::size_type>(ref, dnnz_per_row);
    for (auto i = 0; i < nnz_per_row.get_num_elems(); i++) {
        ASSERT_EQ(nnz_per_row.get_const_data()[i], tmp.get_const_data()[i]);
    }
}


TEST_F(Ell, CountNNZIsEquivalentToRef)
{
    set_up_apply_data();

    gko::size_type nnz;
    gko::size_type dnnz;

    gko::kernels::reference::ell::count_nonzeros(ref, mtx.get(), &nnz);
    gko::kernels::dpcpp::ell::count_nonzeros(dpcpp, dmtx.get(), &dnnz);

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

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<vtype>::value);
}


TEST_F(Ell, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_mtx = mtx->compute_absolute();
    auto dabs_mtx = dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, r<vtype>::value);
}


}  // namespace
