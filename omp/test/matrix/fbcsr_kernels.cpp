// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/fbcsr.hpp>


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/fbcsr_kernels.hpp"
#include "core/test/matrix/fbcsr_sample.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"


namespace {


class Fbcsr : public ::testing::Test {
protected:
    using real_type = double;
    using index_type = int;
    using Arr = gko::array<index_type>;
    using Mtx = gko::matrix::Fbcsr<real_type, index_type>;
    using Vec = gko::matrix::Dense<real_type>;
    using ComplexVec = gko::matrix::Dense<std::complex<real_type>>;
    using ComplexMtx = gko::matrix::Fbcsr<std::complex<real_type>>;

    Fbcsr() : rand_engine(42) {}

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


    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(int num_vectors = 1)
    {
        mtx = gko::test::generate_random_fbcsr<real_type>(
            ref, num_brows, num_bcols, blk_sz, false, false, rand_engine);
        complex_mtx = gko::test::generate_random_fbcsr<std::complex<real_type>>(
            ref, num_brows, num_bcols, blk_sz, false, false, rand_engine);
        square_mtx = gko::test::generate_random_fbcsr<real_type>(
            ref, num_brows, num_brows, blk_sz, false, false, rand_engine);
        dmtx = Mtx::create(omp);
        dmtx->copy_from(mtx);
        complex_dmtx = ComplexMtx::create(omp);
        complex_dmtx->copy_from(complex_mtx);
        square_dmtx = Mtx::create(omp);
        square_dmtx->copy_from(square_mtx);
        expected = gen_mtx<Vec>(num_brows * blk_sz, num_vectors, 1);
        y = gen_mtx<Vec>(num_bcols * blk_sz, num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dresult = Vec::create(omp);
        dresult->copy_from(expected);
        dy = Vec::create(omp);
        dy->copy_from(y);
        dalpha = Vec::create(omp);
        dalpha->copy_from(alpha);
        dbeta = Vec::create(omp);
        dbeta->copy_from(beta);
    }

    struct matrix_pair {
        std::unique_ptr<Mtx> ref;
        std::unique_ptr<Mtx> omp;
    };

    matrix_pair gen_unsorted_mtx()
    {
        constexpr int min_nnz_per_row{2};
        auto local_mtx_ref = gko::test::generate_random_fbcsr<real_type>(
            ref, num_brows, num_bcols, blk_sz, false, true, rand_engine);

        auto local_mtx_omp = Mtx::create(omp);
        local_mtx_omp->copy_from(local_mtx_ref);

        return matrix_pair{std::move(local_mtx_ref), std::move(local_mtx_omp)};
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    const index_type num_brows = 112;
    const index_type num_bcols = 31;
    const int blk_sz = 3;
    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<ComplexMtx> complex_mtx;
    std::unique_ptr<Mtx> square_mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<ComplexMtx> complex_dmtx;
    std::unique_ptr<Mtx> square_dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Fbcsr, CanWriteFromMatrixOnDevice)
{
    using value_type = Mtx::value_type;
    using index_type = Mtx::index_type;
    using MatData = gko::matrix_data<value_type, index_type>;
    gko::testing::FbcsrSample<value_type, index_type> sample(ref);
    auto refmat = sample.generate_fbcsr();
    auto ompmat = gko::clone(omp, refmat);
    MatData refdata;
    MatData ompdata;

    refmat->write(refdata);
    ompmat->write(ompdata);

    ASSERT_TRUE(refdata.nonzeros == ompdata.nonzeros);
}


TEST_F(Fbcsr, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Fbcsr, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Fbcsr, SimpleApplyToDenseMatrixIsEquivalentToRefUnsorted)
{
    set_up_apply_data(3);
    auto pair = gen_unsorted_mtx();

    pair.ref->apply(y, expected);
    pair.omp->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Fbcsr, AdvancedApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Fbcsr, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data(3);
    auto complex_b = gen_mtx<ComplexVec>(num_bcols * blk_sz, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b);
    auto complex_x = gen_mtx<ComplexVec>(num_brows * blk_sz, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x);

    mtx->apply(complex_b, complex_x);
    dmtx->apply(dcomplex_b, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Fbcsr, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data(3);
    auto complex_b = gen_mtx<ComplexVec>(num_bcols * blk_sz, 3, 1);
    auto dcomplex_b = ComplexVec::create(omp);
    dcomplex_b->copy_from(complex_b);
    auto complex_x = gen_mtx<ComplexVec>(num_brows * blk_sz, 3, 1);
    auto dcomplex_x = ComplexVec::create(omp);
    dcomplex_x->copy_from(complex_x);

    mtx->apply(alpha, complex_b, beta, complex_x);
    dmtx->apply(dalpha, dcomplex_b, dbeta, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Fbcsr, TransposeIsEquivalentToRef)
{
    set_up_apply_data();

    auto trans = gko::as<Mtx>(mtx->transpose());
    auto d_trans = gko::as<Mtx>(dmtx->transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Fbcsr, ConjugateTransposeIsEquivalentToRef)
{
    set_up_apply_data();

    auto trans = gko::as<ComplexMtx>(complex_mtx->conj_transpose());
    auto d_trans = gko::as<ComplexMtx>(complex_dmtx->conj_transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Fbcsr, RecognizeSortedMatrix)
{
    set_up_apply_data();
    bool is_sorted_omp{};

    is_sorted_omp = dmtx->is_sorted_by_column_index();

    ASSERT_TRUE(is_sorted_omp);
}


TEST_F(Fbcsr, RecognizeUnsortedMatrix)
{
    auto uns_mtx = gen_unsorted_mtx();
    bool is_sorted_omp{};

    is_sorted_omp = uns_mtx.omp->is_sorted_by_column_index();

    ASSERT_FALSE(is_sorted_omp);
}


TEST_F(Fbcsr, SortSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
}


TEST_F(Fbcsr, SortUnsortedMatrixIsEquivalentToRef)
{
    auto uns_mtx = gen_unsorted_mtx();

    uns_mtx.ref->sort_by_column_index();
    uns_mtx.omp->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(uns_mtx.ref, uns_mtx.omp, 0);
    ASSERT_TRUE(uns_mtx.omp->is_sorted_by_column_index());
}


TEST_F(Fbcsr, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data();

    auto diag = mtx->extract_diagonal();
    auto ddiag = dmtx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 0);
}


TEST_F(Fbcsr, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->compute_absolute_inplace();
    dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 1e-14);
}


TEST_F(Fbcsr, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_mtx = mtx->compute_absolute();
    auto dabs_mtx = dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, 1e-14);
}


TEST_F(Fbcsr, InplaceAbsoluteComplexMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    complex_mtx->compute_absolute_inplace();
    complex_dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(complex_mtx, complex_dmtx, 1e-14);
}


TEST_F(Fbcsr, OutplaceAbsoluteComplexMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_mtx = complex_mtx->compute_absolute();
    auto dabs_mtx = complex_dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, 1e-14);
}


}  // namespace
