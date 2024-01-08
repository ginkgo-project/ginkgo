// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/hybrid_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class Hybrid : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Hybrid<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using ComplexVec = gko::matrix::Dense<std::complex<value_type>>;

    Hybrid() : rand_engine(42) {}

    template <typename MtxType = Vec>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gen_mtx<MtxType>(num_rows, num_cols, min_nnz_row, num_cols);
    }

    template <typename MtxType = Vec>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row, int max_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, max_nnz_row),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(int num_vectors = 1,
                           std::shared_ptr<Mtx::strategy_type> strategy =
                               std::make_shared<Mtx::automatic>())
    {
        mtx = Mtx::create(ref, strategy);
        mtx->move_from(gen_mtx(532, 231, 1));
        expected = gen_mtx(532, num_vectors, 1);
        y = gen_mtx(231, num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(exec, strategy);
        dmtx->copy_from(mtx);
        dresult = gko::clone(exec, expected);
        dy = gko::clone(exec, y);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
    }

    std::default_random_engine rand_engine;

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


TEST_F(Hybrid, SubMatrixExecutorAfterCopyIsEquivalentToExcutor)
{
    set_up_apply_data();

    auto coo_mtx = dmtx->get_coo();
    auto ell_mtx = dmtx->get_ell();

    ASSERT_EQ(coo_mtx->get_executor(), exec);
    ASSERT_EQ(ell_mtx->get_executor(), exec);
    ASSERT_EQ(dmtx->get_executor(), exec);
}


TEST_F(Hybrid, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Hybrid, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Hybrid, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Hybrid, AdvancedApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Hybrid, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = gko::clone(exec, complex_x);

    mtx->apply(complex_b, complex_x);
    dmtx->apply(dcomplex_b, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<value_type>::value);
}


TEST_F(Hybrid, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexVec>(231, 3, 1);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3, 1);
    auto dcomplex_x = gko::clone(exec, complex_x);

    mtx->apply(alpha, complex_b, beta, complex_x);
    dmtx->apply(dalpha, dcomplex_b, dbeta, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<value_type>::value);
}


TEST_F(Hybrid, ConvertEmptyCooToCsrIsEquivalentToRef)
{
    auto balanced_mtx =
        Mtx::create(ref, std::make_shared<Mtx::column_limit>(4));
    balanced_mtx->move_from(gen_mtx(400, 200, 4, 4));
    auto dbalanced_mtx =
        Mtx::create(exec, std::make_shared<Mtx::column_limit>(4));
    dbalanced_mtx->copy_from(balanced_mtx);
    auto csr_mtx = gko::matrix::Csr<value_type>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<value_type>::create(exec);

    balanced_mtx->convert_to(csr_mtx);
    dbalanced_mtx->convert_to(dcsr_mtx);

    GKO_ASSERT_MTX_NEAR(csr_mtx, dcsr_mtx, 1e-14);
}


TEST_F(Hybrid, ConvertWithEmptyFirstAndLastRowToCsrIsEquivalentToRef)
{
    // create a dense matrix for easier manipulation
    auto dense_mtx = gen_mtx(400, 200, 0, 4);
    // set first and last row to zero
    for (gko::size_type col = 0; col < dense_mtx->get_size()[1]; col++) {
        dense_mtx->at(0, col) = gko::zero<value_type>();
        dense_mtx->at(dense_mtx->get_size()[0] - 1, col) =
            gko::zero<value_type>();
    }
    // now convert them to hybrid matrices
    auto balanced_mtx = gko::clone(ref, dense_mtx);
    auto dbalanced_mtx = gko::clone(exec, balanced_mtx);
    auto csr_mtx = gko::matrix::Csr<value_type>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<value_type>::create(exec);

    balanced_mtx->convert_to(csr_mtx);
    dbalanced_mtx->convert_to(dcsr_mtx);

    GKO_ASSERT_MTX_NEAR(csr_mtx, dcsr_mtx, 1e-14);
}


TEST_F(Hybrid, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data(1, std::make_shared<Mtx::column_limit>(2));
    auto csr_mtx = gko::matrix::Csr<value_type>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<value_type>::create(exec);

    mtx->convert_to(csr_mtx);
    dmtx->convert_to(dcsr_mtx);

    GKO_ASSERT_MTX_NEAR(csr_mtx, dcsr_mtx, 0);
}


TEST_F(Hybrid, MoveToCsrIsEquivalentToRef)
{
    set_up_apply_data(1, std::make_shared<Mtx::column_limit>(2));
    auto csr_mtx = gko::matrix::Csr<value_type>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<value_type>::create(exec);

    mtx->move_to(csr_mtx);
    dmtx->move_to(dcsr_mtx);

    GKO_ASSERT_MTX_NEAR(csr_mtx, dcsr_mtx, 0);
}


TEST_F(Hybrid, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data();

    auto diag = mtx->extract_diagonal();
    auto ddiag = dmtx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 0);
}


TEST_F(Hybrid, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->compute_absolute_inplace();
    dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<value_type>::value);
}


TEST_F(Hybrid, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data(1, std::make_shared<Mtx::column_limit>(2));
    using AbsMtx = gko::remove_complex<Mtx>;

    auto abs_mtx = mtx->compute_absolute();
    auto dabs_mtx = dmtx->compute_absolute();
    auto abs_strategy = gko::as<AbsMtx::column_limit>(abs_mtx->get_strategy());
    auto dabs_strategy =
        gko::as<AbsMtx::column_limit>(dabs_mtx->get_strategy());

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, r<value_type>::value);
    GKO_ASSERT_EQ(abs_strategy->get_num_columns(),
                  dabs_strategy->get_num_columns());
    GKO_ASSERT_EQ(abs_strategy->get_num_columns(), 2);
}
