// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/csr.hpp>


#include <random>
#include <stdexcept>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


class Csr : public CommonTestFixture {
protected:
    using Arr = gko::array<int>;
    using Vec = gko::matrix::Dense<value_type>;
    using Vec2 = gko::matrix::Dense<gko::next_precision<value_type>>;
    using Mtx = gko::matrix::Csr<value_type>;
    using ComplexVec = gko::matrix::Dense<std::complex<value_type>>;
    using ComplexMtx = gko::matrix::Csr<std::complex<value_type>>;
    using Perm = gko::matrix::Permutation<index_type>;
    using ScaledPerm = gko::matrix::ScaledPermutation<value_type, index_type>;

    Csr()
#ifdef GINKGO_FAST_TESTS
        : mtx_size(152, 231),
#else
        : mtx_size(532, 231),
#endif
          rand_engine(42)
    {}

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row, int max_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, max_nnz_row),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gen_mtx<MtxType>(num_rows, num_cols, min_nnz_row, num_cols);
    }

    void set_up_mat_data()
    {
        mtx2 = Mtx::create(ref);
        mtx2->move_from(gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 5));
        dmtx2 = Mtx::create(exec);
        dmtx2->copy_from(mtx2);
    }

    template <typename Mtx>
    void set_up_strategy(std::shared_ptr<typename Mtx::automatical>& strategy)
    {
#ifdef GKO_COMPILING_OMP
        throw std::runtime_error{"We shouldn't be testing this"};
#else
        strategy = std::make_shared<typename Mtx::automatical>(exec);
#endif
    }

    template <typename Mtx>
    void set_up_strategy(std::shared_ptr<typename Mtx::sparselib>& strategy)
    {
        strategy = std::make_shared<typename Mtx::sparselib>();
    }

    template <typename Mtx>
    void set_up_strategy(std::shared_ptr<typename Mtx::load_balance>& strategy)
    {
#ifdef GKO_COMPILING_OMP
        throw std::runtime_error{"We shouldn't be testing this"};
#else
        strategy = std::make_shared<typename Mtx::load_balance>(exec);
#endif
    }

    template <typename Mtx>
    void set_up_strategy(std::shared_ptr<typename Mtx::classical>& strategy)
    {
        strategy = std::make_shared<typename Mtx::classical>();
    }

    template <typename Mtx>
    void set_up_strategy(std::shared_ptr<typename Mtx::merge_path>& strategy)
    {
        strategy = std::make_shared<typename Mtx::merge_path>();
    }

    template <typename StrategyType>
    void set_up_apply_data(int num_vectors = 1)
    {
        std::shared_ptr<StrategyType> strategy;
        set_up_strategy<Mtx>(strategy);
        mtx = Mtx::create(ref, strategy);
        mtx->move_from(gen_mtx<Vec>(mtx_size[0], mtx_size[1], 1));
        square_mtx = Mtx::create(ref, strategy);
        square_mtx->move_from(gen_mtx<Vec>(mtx_size[0], mtx_size[0], 1));
        expected = gen_mtx<Vec>(mtx_size[0], num_vectors, 1);
        expected2 = Vec2::create(ref);
        expected2->copy_from(expected);
        y = gen_mtx<Vec>(mtx_size[1], num_vectors, 1);
        y2 = Vec2::create(ref);
        y2->copy_from(y);
        alpha = gko::initialize<Vec>({2.0}, ref);
        alpha2 = gko::initialize<Vec2>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        beta2 = gko::initialize<Vec2>({-1.0}, ref);
        dmtx = Mtx::create(exec, strategy);
        dmtx->copy_from(mtx);
        dsquare_mtx = Mtx::create(exec, strategy);
        dsquare_mtx->copy_from(square_mtx);
        dresult = gko::clone(exec, expected);
        dresult2 = gko::clone(exec, expected2);
        dy = gko::clone(exec, y);
        dy2 = gko::clone(exec, y2);
        dalpha = gko::clone(exec, alpha);
        dalpha2 = gko::clone(exec, alpha2);
        dbeta = gko::clone(exec, beta);
        dbeta2 = gko::clone(exec, beta2);

        std::vector<int> tmp(mtx->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<int> tmp2(mtx->get_size()[1], 0);
        std::iota(tmp2.begin(), tmp2.end(), 0);
        std::shuffle(tmp2.begin(), tmp2.end(), rng);
        std::vector<value_type> scale(mtx->get_size()[0]);
        std::vector<value_type> scale2(mtx->get_size()[1]);
        std::uniform_real_distribution<value_type> dist(1, 2);
        auto gen = [&] { return dist(rng); };
        std::generate(scale.begin(), scale.end(), gen);
        std::generate(scale2.begin(), scale2.end(), gen);
        rpermute_idxs = std::make_unique<Arr>(ref, tmp.begin(), tmp.end());
        cpermute_idxs = std::make_unique<Arr>(ref, tmp2.begin(), tmp2.end());
        rpermutation = Perm::create(ref, *rpermute_idxs);
        cpermutation = Perm::create(ref, *cpermute_idxs);
        srpermutation = ScaledPerm::create(
            ref, gko::array<value_type>(ref, scale.begin(), scale.end()),
            *rpermute_idxs);
        scpermutation = ScaledPerm::create(
            ref, gko::array<value_type>(ref, scale2.begin(), scale2.end()),
            *cpermute_idxs);
    }

    template <typename StrategyType>
    void set_up_apply_complex_data()
    {
        std::shared_ptr<StrategyType> strategy;
        set_up_strategy<ComplexMtx>(strategy);
        complex_mtx = ComplexMtx::create(ref, strategy);
        complex_mtx->move_from(
            gen_mtx<ComplexVec>(mtx_size[0], mtx_size[1], 1));
        dcomplex_mtx = ComplexMtx::create(exec, strategy);
        dcomplex_mtx->copy_from(complex_mtx);
    }

    void unsort_mtx()
    {
        gko::test::unsort_matrix(mtx, rand_engine);
        dmtx->copy_from(mtx);
    }

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<ComplexMtx> complex_mtx;
    std::unique_ptr<Mtx> square_mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec2> expected2;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec2> y2;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec2> alpha2;
    std::unique_ptr<Vec> beta;
    std::unique_ptr<Vec2> beta2;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Mtx> dmtx2;
    std::unique_ptr<ComplexMtx> dcomplex_mtx;
    std::unique_ptr<Mtx> dsquare_mtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec2> dresult2;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec2> dy2;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec2> dalpha2;
    std::unique_ptr<Vec> dbeta;
    std::unique_ptr<Vec2> dbeta2;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
    std::unique_ptr<Perm> rpermutation;
    std::unique_ptr<Perm> cpermutation;
    std::unique_ptr<ScaledPerm> srpermutation;
    std::unique_ptr<ScaledPerm> scpermutation;
};


TEST_F(Csr, StrategyAfterCopyIsEquivalentToRef)
{
    set_up_apply_data<Mtx::merge_path>();

    ASSERT_EQ(mtx->get_strategy()->get_name(),
              dmtx->get_strategy()->get_name());
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithClassical)
{
    set_up_apply_data<Mtx::classical>();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithClassicalUnsorted)
{
    set_up_apply_data<Mtx::classical>();
    unsort_mtx();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithClassical)
{
    set_up_apply_data<Mtx::classical>();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithClassical)
{
    set_up_apply_data<Mtx::classical>(3);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithClassical)
{
    set_up_apply_data<Mtx::classical>(3);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


// OpenMP doesn't have strategies
#ifndef GKO_COMPILING_OMP


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data<Mtx::load_balance>();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithLoadBalanceUnsorted)
{
    set_up_apply_data<Mtx::load_balance>();
    unsort_mtx();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data<Mtx::load_balance>();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithSparselib)
{
    set_up_apply_data<Mtx::sparselib>();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithSparselibUnsorted)
{
    set_up_apply_data<Mtx::sparselib>();
    unsort_mtx();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithSparselib)
{
    set_up_apply_data<Mtx::sparselib>();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithMergePath)
{
    set_up_apply_data<Mtx::merge_path>();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithMergePathUnsorted)
{
    set_up_apply_data<Mtx::merge_path>();
    unsort_mtx();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithMergePath)
{
    set_up_apply_data<Mtx::merge_path>();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithAutomatical)
{
    set_up_apply_data<Mtx::automatical>();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithAutomaticalUnsorted)
{
    set_up_apply_data<Mtx::automatical>();
    unsort_mtx();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data<Mtx::load_balance>(3);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data<Mtx::load_balance>(3);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithMergePath)
{
    set_up_apply_data<Mtx::merge_path>(3);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithMergePath)
{
    set_up_apply_data<Mtx::merge_path>(3);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, OneAutomaticalWorksWithDifferentMatrices)
{
    auto automatical = std::make_shared<Mtx::automatical>(exec);
#ifdef GKO_COMPILING_CUDA
    auto row_len_limit = automatical->nvidia_row_len_limit;
#elif defined(GKO_COMPILING_HIP)
    auto row_len_limit = std::max(automatical->nvidia_row_len_limit,
                                  automatical->amd_row_len_limit);
#else
    auto row_len_limit = automatical->intel_row_len_limit;
#endif
    auto load_balance_mtx =
        gen_mtx<Mtx>(1, row_len_limit + 1000, row_len_limit + 1);
    auto classical_mtx = gen_mtx<Mtx>(50, 50, 1);
    auto load_balance_mtx_d = gko::clone(exec, load_balance_mtx);
    auto classical_mtx_d = gko::clone(exec, classical_mtx);

    load_balance_mtx_d->set_strategy(automatical);
    classical_mtx_d->set_strategy(automatical);

    EXPECT_EQ("load_balance", load_balance_mtx_d->get_strategy()->get_name());
    EXPECT_EQ("classical", classical_mtx_d->get_strategy()->get_name());
    ASSERT_NE(load_balance_mtx_d->get_strategy().get(),
              classical_mtx_d->get_strategy().get());
}


#endif


TEST_F(Csr, AdvancedApplyToCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    mtx->apply(alpha, trans, beta, square_mtx);
    dmtx->apply(dalpha, d_trans, dbeta, dsquare_mtx);

    GKO_ASSERT_MTX_NEAR(dsquare_mtx, square_mtx, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(dsquare_mtx, square_mtx);
    ASSERT_TRUE(dsquare_mtx->is_sorted_by_column_index());
}


TEST_F(Csr, SimpleApplyToCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    mtx->apply(trans, square_mtx);
    dmtx->apply(d_trans, dsquare_mtx);

    GKO_ASSERT_MTX_NEAR(dsquare_mtx, square_mtx, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(dsquare_mtx, square_mtx);
    ASSERT_TRUE(dsquare_mtx->is_sorted_by_column_index());
}


TEST_F(Csr, SimpleApplyToSparseCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto mtx2 =
        gen_mtx<Mtx>(mtx->get_size()[1], square_mtx->get_size()[1], 0, 10);
    auto dmtx2 = Mtx::create(exec, mtx2->get_size());
    dmtx2->copy_from(mtx2);

    mtx->apply(mtx2, square_mtx);
    dmtx->apply(dmtx2, dsquare_mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(dsquare_mtx, square_mtx);
    GKO_ASSERT_MTX_NEAR(dsquare_mtx, square_mtx, r<value_type>::value);
    ASSERT_TRUE(dsquare_mtx->is_sorted_by_column_index());
}


TEST_F(Csr, SimpleApplySparseToSparseCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto mtx1 = gen_mtx<Mtx>(mtx->get_size()[0], mtx->get_size()[1], 0, 10);
    auto mtx2 =
        gen_mtx<Mtx>(mtx->get_size()[1], square_mtx->get_size()[1], 0, 10);
    auto dmtx1 = gko::clone(exec, mtx1);
    auto dmtx2 = gko::clone(exec, mtx2);

    mtx1->apply(mtx2, square_mtx);
    dmtx1->apply(dmtx2, dsquare_mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(dsquare_mtx, square_mtx);
    GKO_ASSERT_MTX_NEAR(dsquare_mtx, square_mtx, r<value_type>::value);
    ASSERT_TRUE(dsquare_mtx->is_sorted_by_column_index());
}


// TODO: broken in ROCm <= 4.5
#ifndef GKO_COMPILING_HIP


TEST_F(Csr, SimpleApplyToEmptyCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto mtx2 =
        gen_mtx<Mtx>(mtx->get_size()[1], square_mtx->get_size()[1], 0, 0);
    auto dmtx2 = Mtx::create(exec, mtx2->get_size());
    dmtx2->copy_from(mtx2);

    mtx->apply(mtx2, square_mtx);
    dmtx->apply(dmtx2, dsquare_mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(dsquare_mtx, square_mtx);
    GKO_ASSERT_MTX_NEAR(dsquare_mtx, square_mtx, r<value_type>::value);
    ASSERT_TRUE(dsquare_mtx->is_sorted_by_column_index());
}


#endif


TEST_F(Csr, AdvancedApplyToIdentityMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto a = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    auto b = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    auto da = gko::clone(exec, a);
    auto db = gko::clone(exec, b);
    auto id = gko::matrix::Identity<Mtx::value_type>::create(ref, mtx_size[1]);
    auto did =
        gko::matrix::Identity<Mtx::value_type>::create(exec, mtx_size[1]);

    a->apply(alpha, id, beta, b);
    da->apply(dalpha, did, dbeta, db);

    GKO_ASSERT_MTX_NEAR(b, db, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(b, db);
    ASSERT_TRUE(db->is_sorted_by_column_index());
}


TEST_F(Csr, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto complex_b = gen_mtx<ComplexVec>(this->mtx_size[1], 3, 1);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(this->mtx_size[0], 3, 1);
    auto dcomplex_x = gko::clone(exec, complex_x);

    mtx->apply(complex_b, complex_x);
    dmtx->apply(dcomplex_b, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto complex_b = gen_mtx<ComplexVec>(this->mtx_size[1], 3, 1);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(this->mtx_size[0], 3, 1);
    auto dcomplex_x = gko::clone(exec, complex_x);

    mtx->apply(alpha, complex_b, beta, complex_x);
    dmtx->apply(dalpha, dcomplex_b, dbeta, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<value_type>::value);
}


TEST_F(Csr, TransposeIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();

    auto trans = gko::as<Mtx>(mtx->transpose());
    auto d_trans = gko::as<Mtx>(dmtx->transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Csr, Transpose64IsEquivalentToRef)
{
    using Mtx64 = gko::matrix::Csr<value_type, gko::int64>;
    auto mtx = gen_mtx<Mtx64>(123, 234, 0);
    auto dmtx = gko::clone(exec, mtx);

    auto trans = gko::as<Mtx64>(mtx->transpose());
    auto d_trans = gko::as<Mtx64>(dmtx->transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Csr, ConjugateTransposeIsEquivalentToRef)
{
    set_up_apply_complex_data<ComplexMtx::classical>();

    auto trans = gko::as<ComplexMtx>(complex_mtx->conj_transpose());
    auto d_trans = gko::as<ComplexMtx>(dcomplex_mtx->conj_transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Csr, ConjugateTranspose64IsEquivalentToRef)
{
    using Mtx64 = gko::matrix::Csr<value_type, gko::int64>;
    auto mtx = gen_mtx<Mtx64>(123, 234, 0);
    auto dmtx = gko::clone(exec, mtx);

    auto trans = gko::as<Mtx64>(mtx->transpose());
    auto d_trans = gko::as<Mtx64>(dmtx->transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Csr, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto dense_mtx = gko::matrix::Dense<value_type>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<value_type>::create(exec);

    mtx->convert_to(dense_mtx);
    dmtx->convert_to(ddense_mtx);

    GKO_ASSERT_MTX_NEAR(dense_mtx, ddense_mtx, 0);
}


TEST_F(Csr, MoveToDenseIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto dense_mtx = gko::matrix::Dense<value_type>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<value_type>::create(exec);

    mtx->move_to(dense_mtx);
    dmtx->move_to(ddense_mtx);

    GKO_ASSERT_MTX_NEAR(dense_mtx, ddense_mtx, 0);
}


TEST_F(Csr, ConvertToEllIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto ell_mtx = gko::matrix::Ell<value_type>::create(ref);
    auto dell_mtx = gko::matrix::Ell<value_type>::create(exec);

    mtx->convert_to(ell_mtx);
    dmtx->convert_to(dell_mtx);

    GKO_ASSERT_MTX_NEAR(ell_mtx, dell_mtx, 0);
}


TEST_F(Csr, MoveToEllIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto ell_mtx = gko::matrix::Ell<value_type>::create(ref);
    auto dell_mtx = gko::matrix::Ell<value_type>::create(exec);

    mtx->move_to(ell_mtx);
    dmtx->move_to(dell_mtx);

    GKO_ASSERT_MTX_NEAR(ell_mtx, dell_mtx, 0);
}


TEST_F(Csr, ConvertToSparsityCsrIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto sparsity_mtx = gko::matrix::SparsityCsr<value_type>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<value_type>::create(exec);

    mtx->convert_to(sparsity_mtx);
    dmtx->convert_to(d_sparsity_mtx);

    GKO_ASSERT_MTX_NEAR(sparsity_mtx, d_sparsity_mtx, 0);
}


TEST_F(Csr, MoveToSparsityCsrIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto sparsity_mtx = gko::matrix::SparsityCsr<value_type>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<value_type>::create(exec);

    mtx->move_to(sparsity_mtx);
    dmtx->move_to(d_sparsity_mtx);

    GKO_ASSERT_MTX_NEAR(sparsity_mtx, d_sparsity_mtx, 0);
}


TEST_F(Csr, ConvertToCooIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto coo_mtx = gko::matrix::Coo<value_type>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<value_type>::create(exec);

    mtx->convert_to(coo_mtx);
    dmtx->convert_to(dcoo_mtx);

    GKO_ASSERT_MTX_NEAR(coo_mtx, dcoo_mtx, 0);
}


TEST_F(Csr, MoveToCooIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto coo_mtx = gko::matrix::Coo<value_type>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<value_type>::create(exec);

    mtx->move_to(coo_mtx);
    dmtx->move_to(dcoo_mtx);

    GKO_ASSERT_MTX_NEAR(coo_mtx, dcoo_mtx, 0);
}


TEST_F(Csr, ConvertToSellpIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto sellp_mtx = gko::matrix::Sellp<value_type>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<value_type>::create(exec);

    mtx->convert_to(sellp_mtx);
    dmtx->convert_to(dsellp_mtx);

    GKO_ASSERT_MTX_NEAR(sellp_mtx, dsellp_mtx, 0);
}


TEST_F(Csr, MoveToSellpIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    auto sellp_mtx = gko::matrix::Sellp<value_type>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<value_type>::create(exec);

    mtx->move_to(sellp_mtx);
    dmtx->move_to(dsellp_mtx);

    GKO_ASSERT_MTX_NEAR(sellp_mtx, dsellp_mtx, 0);
}


TEST_F(Csr, ConvertsEmptyToSellp)
{
    auto dempty_mtx = Mtx::create(exec);
    auto dsellp_mtx = gko::matrix::Sellp<value_type>::create(exec);

    dempty_mtx->convert_to(dsellp_mtx);

    ASSERT_EQ(exec->copy_val_to_host(dsellp_mtx->get_const_slice_sets()), 0);
    ASSERT_FALSE(dsellp_mtx->get_size());
}


TEST_F(Csr, ConvertToHybridIsEquivalentToRef)
{
    using Hybrid_type = gko::matrix::Hybrid<value_type>;
    set_up_apply_data<Mtx::classical>();
    auto hybrid_mtx = Hybrid_type::create(
        ref, std::make_shared<Hybrid_type::column_limit>(2));
    auto dhybrid_mtx = Hybrid_type::create(
        exec, std::make_shared<Hybrid_type::column_limit>(2));

    mtx->convert_to(hybrid_mtx);
    dmtx->convert_to(dhybrid_mtx);

    GKO_ASSERT_MTX_NEAR(hybrid_mtx, dhybrid_mtx, 0);
}


TEST_F(Csr, MoveToHybridIsEquivalentToRef)
{
    using Hybrid_type = gko::matrix::Hybrid<value_type>;
    set_up_apply_data<Mtx::classical>();
    auto hybrid_mtx = Hybrid_type::create(
        ref, std::make_shared<Hybrid_type::column_limit>(2));
    auto dhybrid_mtx = Hybrid_type::create(
        exec, std::make_shared<Hybrid_type::column_limit>(2));

    mtx->move_to(hybrid_mtx);
    dmtx->move_to(dhybrid_mtx);

    GKO_ASSERT_MTX_NEAR(hybrid_mtx, dhybrid_mtx, 0);
}


TEST_F(Csr, IsGenericPermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<Mtx::classical>();

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);
        auto permuted = square_mtx->permute(rpermutation, mode);
        auto dpermuted = dsquare_mtx->permute(rpermutation, mode);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
        ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
    }
}


TEST_F(Csr, IsColPermutableHypersparse)
{
    using gko::matrix::permute_mode;
    auto hypersparse_mtx = gko::initialize<Mtx>(
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 2.0}}, ref);
    auto dhypersparse_mtx = hypersparse_mtx->clone();
    auto perm3 = Perm::create(ref, gko::array<index_type>{ref, {1, 2, 0}});

    for (auto mode : {permute_mode::columns, permute_mode::inverse_columns}) {
        SCOPED_TRACE(mode);
        auto permuted = hypersparse_mtx->permute(perm3, mode);
        auto dpermuted = dhypersparse_mtx->permute(perm3, mode);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
        ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
    }
}


TEST_F(Csr, IsGenericPermutableRectangular)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<Mtx::classical>();

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::inverse_rows,
          permute_mode::inverse_columns}) {
        SCOPED_TRACE(mode);
        auto perm = (mode & permute_mode::rows) == permute_mode::rows
                        ? rpermutation.get()
                        : cpermutation.get();

        auto permuted = mtx->permute(perm, mode);
        auto dpermuted = dmtx->permute(perm, mode);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
        ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
    }
}


TEST_F(Csr, IsNonsymmPermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<Mtx::classical>();

    for (auto invert : {false, true}) {
        SCOPED_TRACE(invert);
        auto permuted = mtx->permute(rpermutation, cpermutation, invert);
        auto dpermuted = dmtx->permute(rpermutation, cpermutation, invert);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
        ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
    }
}


TEST_F(Csr, IsGenericScalePermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<Mtx::classical>();

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);
        auto permuted = square_mtx->scale_permute(srpermutation, mode);
        auto dpermuted = dsquare_mtx->scale_permute(srpermutation, mode);

        GKO_EXPECT_MTX_NEAR(permuted, dpermuted, r<value_type>::value);
        GKO_EXPECT_MTX_EQ_SPARSITY(permuted, dpermuted);
        EXPECT_TRUE(dpermuted->is_sorted_by_column_index());
    }
}


TEST_F(Csr, IsColScalePermutableHypersparse)
{
    using gko::matrix::permute_mode;
    auto hypersparse_mtx = gko::initialize<Mtx>(
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 2.0}}, ref);
    auto dhypersparse_mtx = hypersparse_mtx->clone();
    auto perm3 =
        ScaledPerm::create(ref, gko::array<value_type>{ref, {1.0, 2.0, 4.0}},
                           gko::array<index_type>{ref, {1, 2, 0}});

    for (auto mode : {permute_mode::columns, permute_mode::inverse_columns}) {
        SCOPED_TRACE(mode);
        auto permuted = hypersparse_mtx->scale_permute(perm3, mode);
        auto dpermuted = dhypersparse_mtx->scale_permute(perm3, mode);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, r<value_type>::value);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
        ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
    }
}


TEST_F(Csr, IsGenericScalePermutableRectangular)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<Mtx::classical>();

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::inverse_rows,
          permute_mode::inverse_columns}) {
        SCOPED_TRACE(mode);
        auto perm = (mode & permute_mode::rows) == permute_mode::rows
                        ? srpermutation.get()
                        : scpermutation.get();

        auto permuted = mtx->scale_permute(perm, mode);
        auto dpermuted = dmtx->scale_permute(perm, mode);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, r<value_type>::value);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
        ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
    }
}


TEST_F(Csr, IsNonsymmScalePermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<Mtx::classical>();

    for (auto invert : {false, true}) {
        SCOPED_TRACE(invert);
        auto permuted =
            mtx->scale_permute(srpermutation, scpermutation, invert);
        auto dpermuted =
            dmtx->scale_permute(srpermutation, scpermutation, invert);

        GKO_EXPECT_MTX_NEAR(permuted, dpermuted, r<value_type>::value);
        GKO_EXPECT_MTX_EQ_SPARSITY(permuted, dpermuted);
        EXPECT_TRUE(dpermuted->is_sorted_by_column_index());
    }
}


TEST_F(Csr, IsPermutable)
{
    set_up_apply_data<Mtx::classical>();

    auto permuted = gko::as<Mtx>(square_mtx->permute(rpermute_idxs.get()));
    auto dpermuted = gko::as<Mtx>(dsquare_mtx->permute(rpermute_idxs.get()));

    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Csr, IsInversePermutable)
{
    set_up_apply_data<Mtx::classical>();

    auto permuted =
        gko::as<Mtx>(square_mtx->inverse_permute(rpermute_idxs.get()));
    auto dpermuted =
        gko::as<Mtx>(dsquare_mtx->inverse_permute(rpermute_idxs.get()));

    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Csr, IsRowPermutable)
{
    set_up_apply_data<Mtx::classical>();

    auto r_permute = gko::as<Mtx>(mtx->row_permute(rpermute_idxs.get()));
    auto dr_permute = gko::as<Mtx>(dmtx->row_permute(rpermute_idxs.get()));

    GKO_ASSERT_MTX_EQ_SPARSITY(r_permute, dr_permute);
    GKO_ASSERT_MTX_NEAR(r_permute, dr_permute, 0);
}


TEST_F(Csr, IsColPermutable)
{
    set_up_apply_data<Mtx::classical>();

    auto c_permute = gko::as<Mtx>(mtx->column_permute(cpermute_idxs.get()));
    auto dc_permute = gko::as<Mtx>(dmtx->column_permute(cpermute_idxs.get()));

    ASSERT_TRUE(dc_permute->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(c_permute, dc_permute);
    GKO_ASSERT_MTX_NEAR(c_permute, dc_permute, 0);
}


TEST_F(Csr, IsInverseRowPermutable)
{
    set_up_apply_data<Mtx::classical>();

    auto inverse_r_permute =
        gko::as<Mtx>(mtx->inverse_row_permute(rpermute_idxs.get()));
    auto d_inverse_r_permute =
        gko::as<Mtx>(dmtx->inverse_row_permute(rpermute_idxs.get()));

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse_r_permute, d_inverse_r_permute);
    GKO_ASSERT_MTX_NEAR(inverse_r_permute, d_inverse_r_permute, 0);
}


TEST_F(Csr, IsInverseColPermutable)
{
    set_up_apply_data<Mtx::classical>();

    auto inverse_c_permute =
        gko::as<Mtx>(mtx->inverse_column_permute(cpermute_idxs.get()));
    auto d_inverse_c_permute =
        gko::as<Mtx>(dmtx->inverse_column_permute(cpermute_idxs.get()));

    ASSERT_TRUE(d_inverse_c_permute->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(inverse_c_permute, d_inverse_c_permute);
    GKO_ASSERT_MTX_NEAR(inverse_c_permute, d_inverse_c_permute, 0);
}


TEST_F(Csr, RecognizeSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    bool is_sorted_exec{};
    bool is_sorted_ref{};

    is_sorted_ref = mtx->is_sorted_by_column_index();
    is_sorted_exec = dmtx->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_exec);
}


TEST_F(Csr, RecognizeUnsortedMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    unsort_mtx();
    bool is_sorted_exec{};
    bool is_sorted_ref{};

    is_sorted_ref = mtx->is_sorted_by_column_index();
    is_sorted_exec = dmtx->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_exec);
}


TEST_F(Csr, SortSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    ASSERT_TRUE(dmtx->is_sorted_by_column_index());

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, SortSortedMatrixIsEquivalentToRef64)
{
    using Mtx64 = gko::matrix::Csr<value_type, gko::int64>;
    auto mtx = gen_mtx<Mtx64>(123, 234, 0);
    auto dmtx = gko::clone(exec, mtx);
    ASSERT_TRUE(dmtx->is_sorted_by_column_index());

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, SortUnsortedMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();
    unsort_mtx();
    ASSERT_FALSE(dmtx->is_sorted_by_column_index());

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, SortUnsortedMatrixIsEquivalentToRef64)
{
    using Mtx64 = gko::matrix::Csr<value_type, gko::int64>;
    auto mtx = gen_mtx<Mtx64>(123, 234, 0);
    gko::test::unsort_matrix(mtx, rand_engine);
    auto dmtx = gko::clone(exec, mtx);
    ASSERT_FALSE(dmtx->is_sorted_by_column_index());

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, SortSortedComplexMatrixIsEquivalentToRef)
{
    using MtxComplex = gko::matrix::Csr<std::complex<value_type>, gko::int32>;
    auto mtx = gen_mtx<MtxComplex>(123, 234, 0);
    auto dmtx = gko::clone(exec, mtx);
    ASSERT_TRUE(dmtx->is_sorted_by_column_index());

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, SortSortedComplexMatrixIsEquivalentToRef64)
{
    using MtxComplex64 = gko::matrix::Csr<std::complex<value_type>, gko::int64>;
    auto mtx = gen_mtx<MtxComplex64>(123, 234, 0);
    auto dmtx = gko::clone(exec, mtx);
    ASSERT_TRUE(dmtx->is_sorted_by_column_index());

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, SortUnsortedComplexMatrixIsEquivalentToRef)
{
    using MtxComplex = gko::matrix::Csr<std::complex<value_type>, gko::int32>;
    auto mtx = gen_mtx<MtxComplex>(123, 234, 0);
    gko::test::unsort_matrix(mtx, rand_engine);
    auto dmtx = gko::clone(exec, mtx);
    ASSERT_FALSE(dmtx->is_sorted_by_column_index());

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, SortUnsortedComplexMatrixIsEquivalentToRef64)
{
    using MtxComplex64 = gko::matrix::Csr<std::complex<value_type>, gko::int64>;
    auto mtx = gen_mtx<MtxComplex64>(123, 234, 0);
    gko::test::unsort_matrix(mtx, rand_engine);
    auto dmtx = gko::clone(exec, mtx);
    ASSERT_FALSE(dmtx->is_sorted_by_column_index());

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();

    auto diag = mtx->extract_diagonal();
    auto ddiag = dmtx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 0);
}


TEST_F(Csr, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();

    mtx->compute_absolute_inplace();
    dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<value_type>::value);
}


TEST_F(Csr, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data<Mtx::classical>();

    auto abs_mtx = mtx->compute_absolute();
    auto dabs_mtx = dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, r<value_type>::value);
}


TEST_F(Csr, InplaceAbsoluteComplexMatrixIsEquivalentToRef)
{
    set_up_apply_complex_data<ComplexMtx::classical>();

    complex_mtx->compute_absolute_inplace();
    dcomplex_mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(complex_mtx, dcomplex_mtx, r<value_type>::value);
}


TEST_F(Csr, OutplaceAbsoluteComplexMatrixIsEquivalentToRef)
{
    set_up_apply_complex_data<ComplexMtx::classical>();

    auto abs_mtx = complex_mtx->compute_absolute();
    auto dabs_mtx = dcomplex_mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, r<value_type>::value);
}


TEST_F(Csr, CalculateNnzPerRowInSpanIsEquivalentToRef)
{
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    set_up_mat_data();
    gko::span rspan{7, 51};
    gko::span cspan{22, 88};
    auto size = this->mtx2->get_size();
    auto row_nnz = gko::array<int>(this->ref, rspan.length() + 1);
    auto drow_nnz = gko::array<int>(this->exec, row_nnz);

    gko::kernels::reference::csr::calculate_nonzeros_per_row_in_span(
        this->ref, this->mtx2.get(), rspan, cspan, &row_nnz);
    gko::kernels::EXEC_NAMESPACE::csr::calculate_nonzeros_per_row_in_span(
        this->exec, this->dmtx2.get(), rspan, cspan, &drow_nnz);

    GKO_ASSERT_ARRAY_EQ(row_nnz, drow_nnz);
}


TEST_F(Csr, ComputeSubmatrixIsEquivalentToRef)
{
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    set_up_mat_data();
    gko::span rspan{7, 51};
    gko::span cspan{22, 88};
    auto size = this->mtx2->get_size();
    auto row_nnz = gko::array<int>(this->ref, rspan.length() + 1);
    row_nnz.fill(gko::zero<int>());
    gko::kernels::reference::csr::calculate_nonzeros_per_row_in_span(
        this->ref, this->mtx2.get(), rspan, cspan, &row_nnz);
    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->ref, row_nnz.get_data(), row_nnz.get_size());
    auto num_nnz = row_nnz.get_data()[rspan.length()];
    auto drow_nnz = gko::array<int>(this->exec, row_nnz);
    auto smat1 =
        Mtx::create(this->ref, gko::dim<2>(rspan.length(), cspan.length()),
                    std::move(gko::array<value_type>(this->ref, num_nnz)),
                    std::move(gko::array<index_type>(this->ref, num_nnz)),
                    std::move(row_nnz));
    auto sdmat1 =
        Mtx::create(this->exec, gko::dim<2>(rspan.length(), cspan.length()),
                    std::move(gko::array<value_type>(this->exec, num_nnz)),
                    std::move(gko::array<index_type>(this->exec, num_nnz)),
                    std::move(drow_nnz));


    gko::kernels::reference::csr::compute_submatrix(this->ref, this->mtx2.get(),
                                                    rspan, cspan, smat1.get());
    gko::kernels::EXEC_NAMESPACE::csr::compute_submatrix(
        this->exec, this->dmtx2.get(), rspan, cspan, sdmat1.get());

    GKO_ASSERT_MTX_NEAR(sdmat1, smat1, 0.0);
}


#ifdef GKO_COMPILING_OMP


TEST_F(Csr, CalculateNnzPerRowInIndexSetIsEquivalentToRef)
{
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    set_up_mat_data();
    gko::index_set<index_type> rset{
        this->ref, {42, 7, 8, 9, 10, 22, 25, 26, 34, 35, 36, 51}};
    gko::index_set<index_type> cset{this->ref,
                                    {42, 22, 24, 26, 28, 30, 81, 82, 83, 88}};
    gko::index_set<index_type> drset(this->exec, rset);
    gko::index_set<index_type> dcset(this->exec, cset);
    auto row_nnz = gko::array<int>(this->ref, rset.get_size() + 1);
    row_nnz.fill(gko::zero<int>());
    auto drow_nnz = gko::array<int>(this->exec, row_nnz);

    gko::kernels::reference::csr::calculate_nonzeros_per_row_in_index_set(
        this->ref, this->mtx2.get(), rset, cset, row_nnz.get_data());
    gko::kernels::EXEC_NAMESPACE::csr::calculate_nonzeros_per_row_in_index_set(
        this->exec, this->dmtx2.get(), drset, dcset, drow_nnz.get_data());

    GKO_ASSERT_ARRAY_EQ(row_nnz, drow_nnz);
}


TEST_F(Csr, ComputeSubmatrixFromIndexSetIsEquivalentToRef)
{
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    set_up_mat_data();
    gko::index_set<index_type> rset{
        this->ref, {42, 7, 8, 9, 10, 22, 25, 26, 34, 35, 36, 51}};
    gko::index_set<index_type> cset{this->ref,
                                    {42, 22, 24, 26, 28, 30, 81, 82, 83, 88}};
    gko::index_set<index_type> drset(this->exec, rset);
    gko::index_set<index_type> dcset(this->exec, cset);
    auto row_nnz = gko::array<int>(this->ref, rset.get_size() + 1);
    row_nnz.fill(gko::zero<int>());
    gko::kernels::reference::csr::calculate_nonzeros_per_row_in_index_set(
        this->ref, this->mtx2.get(), rset, cset, row_nnz.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->ref, row_nnz.get_data(), row_nnz.get_size());
    auto num_nnz = row_nnz.get_data()[rset.get_size()];
    auto drow_nnz = gko::array<int>(this->exec, row_nnz);
    auto smat1 =
        Mtx::create(this->ref, gko::dim<2>(rset.get_size(), cset.get_size()),
                    std::move(gko::array<value_type>(this->ref, num_nnz)),
                    std::move(gko::array<index_type>(this->ref, num_nnz)),
                    std::move(row_nnz));
    auto sdmat1 =
        Mtx::create(this->exec, gko::dim<2>(rset.get_size(), cset.get_size()),
                    std::move(gko::array<value_type>(this->exec, num_nnz)),
                    std::move(gko::array<index_type>(this->exec, num_nnz)),
                    std::move(drow_nnz));

    gko::kernels::reference::csr::compute_submatrix_from_index_set(
        this->ref, this->mtx2.get(), rset, cset, smat1.get());
    gko::kernels::EXEC_NAMESPACE::csr::compute_submatrix_from_index_set(
        this->exec, this->dmtx2.get(), drset, dcset, sdmat1.get());

    GKO_ASSERT_MTX_NEAR(sdmat1, smat1, 0.0);
}


TEST_F(Csr, CreateSubMatrixFromIndexSetIsEquivalentToRef)
{
    set_up_mat_data();

    gko::index_set<index_type> rset{
        this->ref, {42, 7, 8, 9, 10, 22, 25, 26, 34, 35, 36, 51}};
    gko::index_set<index_type> cset{this->ref,
                                    {42, 22, 24, 26, 28, 30, 81, 82, 83, 88}};
    gko::index_set<index_type> drset(this->exec, rset);
    gko::index_set<index_type> dcset(this->exec, cset);
    auto smat1 = this->mtx2->create_submatrix(rset, cset);
    auto sdmat1 = this->dmtx2->create_submatrix(drset, dcset);

    GKO_ASSERT_MTX_NEAR(sdmat1, smat1, 0.0);
}


#endif  // GKO_COMPILING_OMP


TEST_F(Csr, CreateSubMatrixIsEquivalentToRef)
{
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    set_up_mat_data();
    gko::span rspan{47, 81};
    gko::span cspan{2, 31};

    auto smat1 = this->mtx2->create_submatrix(rspan, cspan);
    auto sdmat1 = this->dmtx2->create_submatrix(rspan, cspan);

    GKO_ASSERT_MTX_NEAR(sdmat1, smat1, 0.0);
}


TEST_F(Csr, CanDetectMissingDiagonalEntry)
{
    using T = double;
    using Csr = Mtx;
    auto ref_mtx = gen_mtx<Csr>(103, 104, 10);
    const auto rowptrs = ref_mtx->get_row_ptrs();
    const auto colidxs = ref_mtx->get_col_idxs();
    gko::utils::ensure_all_diagonal_entries(ref_mtx.get());
    // Choose the last row to ensure that kernel assign enough work
    const int testrow = 102;
    gko::utils::remove_diagonal_entry_from_row(ref_mtx.get(), testrow);
    auto mtx = gko::clone(exec, ref_mtx);
    bool has_diags = true;

    gko::kernels::EXEC_NAMESPACE::csr::check_diagonal_entries_exist(
        exec, mtx.get(), has_diags);

    ASSERT_FALSE(has_diags);
}


TEST_F(Csr, CanDetectWhenAllDiagonalEntriesArePresent)
{
    using Csr = Mtx;
    auto ref_mtx = gen_mtx<Csr>(103, 98, 10);
    gko::utils::ensure_all_diagonal_entries(ref_mtx.get());
    auto mtx = gko::clone(exec, ref_mtx);
    bool has_diags = true;

    gko::kernels::EXEC_NAMESPACE::csr::check_diagonal_entries_exist(
        exec, mtx.get(), has_diags);

    ASSERT_TRUE(has_diags);
}


TEST_F(Csr, AddScaledIdentityToNonSquare)
{
    set_up_apply_data<Mtx::classical>();
    gko::utils::ensure_all_diagonal_entries(mtx.get());
    dmtx->copy_from(mtx);

    mtx->add_scaled_identity(alpha, beta);
    dmtx->add_scaled_identity(dalpha, dbeta);

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<value_type>::value);
}
