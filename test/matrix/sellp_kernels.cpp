// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sellp_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class Sellp : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Sellp<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using ComplexVec = gko::matrix::Dense<std::complex<value_type>>;

    Sellp() : rand_engine(42) {}

    template <typename MtxType = Vec>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols, std::uniform_int_distribution<>(1, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_matrix(
        int total_cols = 1, int slice_size = gko::matrix::default_slice_size,
        int stride_factor = gko::matrix::default_stride_factor)
    {
        mtx = gen_mtx<Mtx>(532, 231);
        empty = Mtx::create(ref);
        expected = gen_mtx(532, total_cols);
        y = gen_mtx(231, total_cols);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = gko::clone(exec, mtx);
        dempty = Mtx::create(exec);
        dresult = gko::clone(exec, expected);
        dy = gko::clone(exec, y);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
    }

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> empty;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Mtx> dempty;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Sellp, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_matrix();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Sellp, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_matrix();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Sellp, SimpleApplyWithSliceSizeAndStrideFactorIsEquivalentToRef)
{
    set_up_apply_matrix(1, 32, 2);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Sellp, AdvancedApplyWithSliceSizeAndStrideFActorIsEquivalentToRef)
{
    set_up_apply_matrix(1, 32, 2);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Sellp, SimpleApplyMultipleRHSIsEquivalentToRef)
{
    set_up_apply_matrix(3);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Sellp, AdvancedApplyMultipleRHSIsEquivalentToRef)
{
    set_up_apply_matrix(4);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Sellp,
       SimpleApplyMultipleRHSWithSliceSizeAndStrideFactorIsEquivalentToRef)
{
    set_up_apply_matrix(5, 2);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Sellp,
       AdvancedApplyMultipleRHSWithSliceSizeAndStrideFActorIsEquivalentToRef)
{
    set_up_apply_matrix(6, 2);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Sellp, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_matrix(64);
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = gko::clone(exec, complex_x);

    mtx->apply(complex_b, complex_x);
    dmtx->apply(dcomplex_b, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<value_type>::value);
}


TEST_F(Sellp, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_matrix(64);
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = gko::clone(exec, complex_x);

    mtx->apply(alpha, complex_b, beta, complex_x);
    dmtx->apply(dalpha, dcomplex_b, dbeta, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<value_type>::value);
}


TEST_F(Sellp, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_matrix(64);
    auto dense_mtx = gko::matrix::Dense<value_type>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<value_type>::create(exec);

    mtx->convert_to(dense_mtx);
    dmtx->convert_to(ddense_mtx);

    GKO_ASSERT_MTX_NEAR(dense_mtx, ddense_mtx, 0);
}


TEST_F(Sellp, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_matrix(64);
    auto csr_mtx = gko::matrix::Csr<value_type>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<value_type>::create(exec);

    mtx->convert_to(csr_mtx);
    dmtx->convert_to(dcsr_mtx);

    GKO_ASSERT_MTX_NEAR(csr_mtx, dcsr_mtx, 0);
}


TEST_F(Sellp, ConvertEmptyToDenseIsEquivalentToRef)
{
    set_up_apply_matrix(64);
    auto dense_mtx = gko::matrix::Dense<value_type>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<value_type>::create(exec);

    empty->convert_to(dense_mtx);
    dempty->convert_to(ddense_mtx);

    GKO_ASSERT_MTX_NEAR(dense_mtx, ddense_mtx, 0);
}


TEST_F(Sellp, ConvertEmptyToCsrIsEquivalentToRef)
{
    set_up_apply_matrix(64);
    auto csr_mtx = gko::matrix::Csr<value_type>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<value_type>::create(exec);

    empty->convert_to(csr_mtx);
    dempty->convert_to(dcsr_mtx);

    GKO_ASSERT_MTX_NEAR(csr_mtx, dcsr_mtx, 0);
}


TEST_F(Sellp, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_matrix(64);

    auto diag = mtx->extract_diagonal();
    auto ddiag = dmtx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 0);
}


TEST_F(Sellp, ExtractDiagonalWithSliceSizeAndStrideFactorIsEquivalentToRef)
{
    set_up_apply_matrix(64, 32, 2);

    auto diag = mtx->extract_diagonal();
    auto ddiag = dmtx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 0);
}


TEST_F(Sellp, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_matrix(64, 32, 2);

    mtx->compute_absolute_inplace();
    dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<value_type>::value);
}


TEST_F(Sellp, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_matrix(64, 32, 2);

    auto abs_mtx = mtx->compute_absolute();
    auto dabs_mtx = dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, r<value_type>::value);
}
