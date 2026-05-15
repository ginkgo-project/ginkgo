// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <random>
#include <stdexcept>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
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
#include "core/matrix/csr_strategy.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/common_fixture.hpp"


class Csr : public CommonTestFixture {
protected:
    using Arr = gko::array<index_type>;
    using Vec = gko::matrix::Dense<value_type>;
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

    template <gko::matrix::csr::spmv_strategy strategy>
    void set_up_apply_data(int num_vectors = 1)
    {
        mtx = Mtx::create(ref, strategy);
        mtx->move_from(gen_mtx<Vec>(mtx_size[0], mtx_size[1], 1));
        square_mtx = Mtx::create(ref, strategy);
        square_mtx->move_from(gen_mtx<Vec>(mtx_size[0], mtx_size[0], 1));
        expected = gen_mtx<Vec>(mtx_size[0], num_vectors, 1);
        y = gen_mtx<Vec>(mtx_size[1], num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(exec, strategy);
        dmtx->copy_from(mtx);
        dsquare_mtx = Mtx::create(exec, strategy);
        dsquare_mtx->copy_from(square_mtx);
        dresult = gko::clone(exec, expected);
        dy = gko::clone(exec, y);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);

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

    template <gko::matrix::csr::spmv_strategy strategy>
    void set_up_apply_complex_data()
    {
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
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Mtx> dmtx2;
    std::unique_ptr<ComplexMtx> dcomplex_mtx;
    std::unique_ptr<Mtx> dsquare_mtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
    std::unique_ptr<Perm> rpermutation;
    std::unique_ptr<Perm> cpermutation;
    std::unique_ptr<ScaledPerm> srpermutation;
    std::unique_ptr<ScaledPerm> scpermutation;
};


TEST_F(Csr, StrategyAfterCopyIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::merge_path>();

    ASSERT_EQ(mtx->get_strategy(), dmtx->get_strategy());
}


TEST_F(Csr, SrowIsCorrectFromLoadBalance)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::load_balance>();

    if (std::dynamic_pointer_cast<const gko::OmpExecutor>(exec)) {
        GTEST_SKIP() << "Csr does not have load balance on OmpExecutor";
    }
    int warp_size = 0;
    if (auto dexec = std::dynamic_pointer_cast<const gko::CudaExecutor>(exec)) {
        warp_size = dexec->get_warp_size();
    } else if (auto dexec =
                   std::dynamic_pointer_cast<const gko::HipExecutor>(exec)) {
        warp_size = dexec->get_warp_size();
    } else if (auto dexec =
                   std::dynamic_pointer_cast<const gko::DpcppExecutor>(exec)) {
        warp_size = 32;
    }
    const auto srow_size = dmtx->get_num_srow_elements();
    // group `warp_size` as a unit, num_lines means how many units we need to
    // handle
    const auto num_lines =
        gko::ceildiv(dmtx->get_num_stored_elements(), warp_size);
    ASSERT_GT(srow_size, 0);
    ASSERT_EQ(exec->copy_val_to_host(dmtx->get_const_srow()), 0);
    for (int i = 1; i < srow_size; i++) {
        auto start = (i * num_lines / srow_size) * warp_size;
        auto srow_val = exec->copy_val_to_host(dmtx->get_const_srow() + i);
        if (srow_val > 0) {
            // the number of elements before this row should be less than the
            // assigned number
            ASSERT_LE(exec->copy_val_to_host(dmtx->get_const_row_ptrs() +
                                             srow_val - 1),
                      start);
        }
        // the starting point should be in this row not the next row.
        ASSERT_GE(start, exec->copy_val_to_host(dmtx->get_const_row_ptrs() +
                                                srow_val));
        ASSERT_LT(start, exec->copy_val_to_host(dmtx->get_const_row_ptrs() +
                                                srow_val + 1));
    }
}

TEST_F(Csr, SimpleApplyIsEquivalentToRefWithClassical)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithClassicalUnsorted)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    unsort_mtx();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithClassical)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithClassical)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>(3);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithClassical)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>(3);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


// OpenMP doesn't have strategies
#ifndef GKO_COMPILING_OMP


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::load_balance>();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithLoadBalanceUnsorted)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::load_balance>();
    unsort_mtx();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::load_balance>();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithSparselib)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::sparselib>();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithSparselibUnsorted)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::sparselib>();
    unsort_mtx();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithSparselib)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::sparselib>();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithMergePath)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::merge_path>();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithMergePathUnsorted)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::merge_path>();
    unsort_mtx();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithMergePath)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::merge_path>();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithAutomatic)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::automatic>();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithAutomaticUnsorted)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::automatic>();
    unsort_mtx();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::load_balance>(3);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::load_balance>(3);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithMergePath)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::merge_path>(3);

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithMergePath)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::merge_path>(3);

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, OneAutomaticWorksWithDifferentMatrices)
{
    if (std::dynamic_pointer_cast<const gko::OmpExecutor>(exec)) {
        GTEST_SKIP() << "Csr does not have load balance under automatic on "
                        "OmpExecutor";
    }
    auto automatic = gko::matrix::csr::spmv_strategy::automatic;
#ifdef GKO_COMPILING_CUDA
    int64_t nnz_limit = 1e6;
    int64_t row_len_limit = 1024;
#elif defined(GKO_COMPILING_HIP)
    int64_t nnz_limit = 1e8;
    int64_t row_len_limit = 768;
#else  // INTEL
    int64_t nnz_limit = 3e8;
    int64_t row_len_limit = 25600;
#endif
    auto load_balance_mtx =
        gen_mtx<Mtx>(1, row_len_limit + 1000, row_len_limit + 1);
    auto classical_mtx = gen_mtx<Mtx>(50, 50, 1);
    auto get_max_nnz_per_row = [](gko::size_type num_rows, auto row_ptrs) {
        int64_t max_row_nnz = 0;
        for (gko::size_type i = 0; i < num_rows; i++) {
            max_row_nnz =
                std::max(max_row_nnz,
                         static_cast<int64_t>(row_ptrs[i + 1] - row_ptrs[i]));
        }
        return max_row_nnz;
    };
    auto load_balance_max_row_nnz =
        get_max_nnz_per_row(load_balance_mtx->get_size()[0],
                            load_balance_mtx->get_const_row_ptrs());
    auto classical_max_row_nnz = get_max_nnz_per_row(
        classical_mtx->get_size()[0], classical_mtx->get_const_row_ptrs());
    auto load_balance_mtx_d = gko::clone(exec, load_balance_mtx);
    auto classical_mtx_d = gko::clone(exec, classical_mtx);

    load_balance_mtx_d->set_strategy(automatic);
    classical_mtx_d->set_strategy(automatic);

    EXPECT_EQ(gko::matrix::csr::detail::get_actual_strategy(
                  exec, load_balance_mtx_d->get_strategy(),
                  load_balance_mtx_d->get_num_stored_elements(),
                  static_cast<gko::size_type>(load_balance_max_row_nnz)),
              gko::matrix::csr::spmv_strategy::load_balance);
    EXPECT_EQ(gko::matrix::csr::detail::get_actual_strategy(
                  exec, classical_mtx_d->get_strategy(),
                  classical_mtx_d->get_num_stored_elements(),
                  static_cast<gko::size_type>(classical_max_row_nnz)),
              gko::matrix::csr::spmv_strategy::classical);
}


#endif


TEST_F(Csr, AdvancedApplyToCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto trans = mtx->transpose();
    auto dtrans = dmtx->transpose();

    mtx->apply(alpha, trans, beta, square_mtx);
    dmtx->apply(dalpha, dtrans, dbeta, dsquare_mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(dsquare_mtx, square_mtx);
    GKO_ASSERT_MTX_NEAR(dsquare_mtx, square_mtx, r<value_type>::value);
    ASSERT_TRUE(dsquare_mtx->is_sorted_by_column_index());
}


TEST_F(Csr, MultiplyAddIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto trans = gko::as<Mtx>(mtx->transpose());
    auto dtrans = gko::as<Mtx>(dmtx->transpose());

    auto result = mtx->multiply_add(alpha, trans, beta, square_mtx);
    auto dresult = dmtx->multiply_add(dalpha, dtrans, dbeta, dsquare_mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, dresult);
    GKO_ASSERT_MTX_NEAR(result, dresult, r<value_type>::value);
    ASSERT_TRUE(dresult->is_sorted_by_column_index());
}


TEST_F(Csr, MultiplyAddIsEquivalentToRefCrossExecutor)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto trans = gko::as<Mtx>(mtx->transpose());

    auto result = mtx->multiply_add(alpha, trans, beta, square_mtx);
    auto dresult = dmtx->multiply_add(alpha, trans, beta, square_mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, dresult);
    GKO_ASSERT_MTX_NEAR(result, dresult, r<value_type>::value);
    ASSERT_TRUE(dresult->is_sorted_by_column_index());
    ASSERT_EQ(dresult->get_executor(), exec);
}


TEST_F(Csr, MultiplyAddReuseCrossExecutor)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto trans = gko::as<Mtx>(mtx->transpose());

    auto [dresult, _dreuse] =
        dmtx->multiply_add_reuse(alpha, trans, beta, square_mtx);
    auto result = mtx->multiply_add(alpha, trans, beta, square_mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(dresult, result);
    GKO_ASSERT_MTX_NEAR(dresult, result, r<value_type>::value);
    ASSERT_TRUE(dresult->is_sorted_by_column_index());
    ASSERT_EQ(dresult->get_executor(), exec);
}


TEST_F(Csr, MultiplyAddReuseUpdateCrossExecutor)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto trans = gko::as<Mtx>(mtx->transpose());
    auto [dresult, dreuse] =
        dmtx->multiply_add_reuse(alpha, trans, beta, square_mtx);
    auto result = mtx->multiply_add(alpha, trans, beta, square_mtx);
    // modify all involved matrices and scalars
    trans->scale(alpha);
    dmtx->scale(beta);
    square_mtx->scale(alpha);
    alpha->scale(alpha);
    beta->scale(beta);

    auto expected = dmtx->multiply_add(alpha, trans, beta, square_mtx);
    mtx = gko::clone(ref, dmtx);
    dreuse.update_values(mtx, alpha, trans, beta, square_mtx, result);

    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
}


TEST_F(Csr, SimpleApplyToCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto trans = mtx->transpose();
    auto dtrans = dmtx->transpose();

    mtx->apply(trans, square_mtx);
    dmtx->apply(dtrans, dsquare_mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(dsquare_mtx, square_mtx);
    GKO_ASSERT_MTX_NEAR(dsquare_mtx, square_mtx, r<value_type>::value);
    ASSERT_TRUE(dsquare_mtx->is_sorted_by_column_index());
}


TEST_F(Csr, MultiplyIsEquivalentToRefCrossExecutor)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto trans = gko::as<Mtx>(mtx->transpose());

    auto result = mtx->multiply(trans);
    auto dresult = dmtx->multiply(trans);

    GKO_ASSERT_MTX_EQ_SPARSITY(dresult, result);
    GKO_ASSERT_MTX_NEAR(dresult, result, r<value_type>::value);
    ASSERT_TRUE(dresult->is_sorted_by_column_index());
    ASSERT_EQ(dresult->get_executor(), exec);
}


TEST_F(Csr, MultiplyWithSparseIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto mtx2 =
        gen_mtx<Mtx>(mtx->get_size()[1], square_mtx->get_size()[1], 0, 10);
    auto dmtx2 = gko::clone(exec, mtx2);

    auto result = mtx->multiply(mtx2);
    auto dresult = dmtx->multiply(dmtx2);

    GKO_ASSERT_MTX_EQ_SPARSITY(dsquare_mtx, square_mtx);
    GKO_ASSERT_MTX_NEAR(dsquare_mtx, square_mtx, r<value_type>::value);
    ASSERT_TRUE(dsquare_mtx->is_sorted_by_column_index());
}


TEST_F(Csr, MultiplySparseWithSparseIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto mtx1 = gen_mtx<Mtx>(mtx->get_size()[0], mtx->get_size()[1], 0, 10);
    auto mtx2 =
        gen_mtx<Mtx>(mtx->get_size()[1], square_mtx->get_size()[1], 0, 10);
    auto dmtx1 = gko::clone(exec, mtx1);
    auto dmtx2 = gko::clone(exec, mtx2);

    auto result = mtx1->multiply(mtx2);
    auto dresult = dmtx1->multiply(dmtx2);

    GKO_ASSERT_MTX_EQ_SPARSITY(dresult, result);
    GKO_ASSERT_MTX_NEAR(dresult, result, r<value_type>::value);
    ASSERT_TRUE(dsquare_mtx->is_sorted_by_column_index());
}


TEST_F(Csr, MultiplyWithEmptyIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto mtx2 =
        gen_mtx<Mtx>(mtx->get_size()[1], square_mtx->get_size()[1], 0, 0);
    auto dmtx2 = gko::clone(exec, mtx2);

    auto result = mtx->multiply(mtx2);
    auto dresult = dmtx->multiply(dmtx2);

    GKO_ASSERT_MTX_EQ_SPARSITY(dresult, result);
    GKO_ASSERT_MTX_NEAR(dresult, result, 0);
    ASSERT_TRUE(dresult->is_sorted_by_column_index());
}


TEST_F(Csr, MultiplyReuseCrossExecutor)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto trans = gko::as<Mtx>(mtx->transpose());

    auto [dresult, _dreuse] = dmtx->multiply_reuse(trans);
    auto expected = mtx->multiply(trans);

    GKO_ASSERT_MTX_EQ_SPARSITY(dresult, expected);
    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
    ASSERT_TRUE(dresult->is_sorted_by_column_index());
    ASSERT_EQ(dresult->get_executor(), exec);
}


TEST_F(Csr, MultiplyReuseUpdateCrossExecutor)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto trans = gko::as<Mtx>(mtx->transpose());
    auto [dresult, dreuse] = dmtx->multiply_reuse(trans);
    auto expected = mtx->multiply(trans);
    auto result = expected->clone();
    // modify all involved matrices and scalars
    mtx->scale(alpha);
    trans->scale(beta);

    dreuse.update_values(mtx, trans, result);
    expected = mtx->multiply(trans);

    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
}


TEST_F(Csr, AdvancedApplyToIdentityMatrixIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
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


TEST_F(Csr, ScaleAddZeroIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto a = Mtx::create(ref);
    auto b = Mtx::create(ref);
    auto da = gko::clone(exec, a);
    auto db = gko::clone(exec, b);

    auto result = a->scale_add(alpha, beta, b);
    auto dresult = da->scale_add(dalpha, dbeta, db);

    GKO_ASSERT_MTX_NEAR(result, dresult, 0);
}


TEST_F(Csr, ScaleAddIsEquivalentToRefCrossExecutor)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto a = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    auto b = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    auto da = gko::clone(exec, a);

    auto result = a->scale_add(alpha, beta, b);
    auto dresult = da->scale_add(alpha, beta, b);

    GKO_ASSERT_MTX_NEAR(result, dresult, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(result, dresult);
    ASSERT_TRUE(dresult->is_sorted_by_column_index());
    ASSERT_EQ(dresult->get_executor(), exec);
}


TEST_F(Csr, ScaleAddReuseCrossExecutor)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    mtx = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    mtx2 = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    dmtx = gko::clone(exec, mtx);

    auto [dresult, _dreuse] = dmtx->add_scale_reuse(alpha, beta, mtx2);
    auto expected = dmtx->scale_add(alpha, beta, mtx2);
    auto result = expected->clone();

    GKO_ASSERT_MTX_EQ_SPARSITY(dresult, expected);
    GKO_ASSERT_MTX_NEAR(dresult, expected, r<value_type>::value);
    ASSERT_TRUE(dresult->is_sorted_by_column_index());
    ASSERT_EQ(dresult->get_executor(), exec);
}


TEST_F(Csr, ScaleAddReuseUpdateCrossExecutor)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    mtx = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    mtx2 = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    dmtx = gko::clone(exec, mtx);
    auto [dresult, dreuse] = dmtx->add_scale_reuse(alpha, beta, mtx2);
    auto expected = dmtx->scale_add(alpha, beta, mtx2);
    auto result = expected->clone();
    // modify all involved matrices and scalars
    dmtx->scale(beta);
    mtx2->scale(alpha);
    alpha->scale(alpha);
    beta->scale(beta);

    expected = dmtx->scale_add(alpha, beta, mtx2);
    mtx = gko::clone(ref, dmtx);
    dreuse.update_values(alpha, mtx, beta, mtx2, result);

    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
}


TEST_F(Csr, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    auto trans = gko::as<Mtx>(mtx->transpose());
    auto dtrans = gko::as<Mtx>(dmtx->transpose());

    GKO_ASSERT_MTX_NEAR(dtrans, trans, 0.0);
    ASSERT_TRUE(dtrans->is_sorted_by_column_index());
}


TEST_F(Csr, Transpose64IsEquivalentToRef)
{
    using Mtx64 = gko::matrix::Csr<value_type, gko::int64>;
    auto mtx = gen_mtx<Mtx64>(123, 234, 0);
    auto dmtx = gko::clone(exec, mtx);

    auto trans = gko::as<Mtx64>(mtx->transpose());
    auto dtrans = gko::as<Mtx64>(dmtx->transpose());

    GKO_ASSERT_MTX_NEAR(dtrans, trans, 0.0);
    ASSERT_TRUE(dtrans->is_sorted_by_column_index());
}


TEST_F(Csr, TransposeReuseIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    auto [trans, reuse] = mtx->transpose_reuse();
    auto [dtrans, dreuse] = dmtx->transpose_reuse();

    GKO_ASSERT_MTX_NEAR(dtrans, trans, 0);
    ASSERT_TRUE(dtrans->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(dreuse.value_permutation,
                               reuse.value_permutation);
}


TEST_F(Csr, TransposeReuseUpdateIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto [trans, reuse] = mtx->transpose_reuse();
    auto [dtrans, dreuse] = dmtx->transpose_reuse();
    // test that the value permutation works: modify input values
    mtx->create_value_view()->scale(alpha);
    dmtx->create_value_view()->scale(dalpha);

    reuse.update_values(mtx, trans);
    dreuse.update_values(dmtx, dtrans);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(mtx->transpose()), trans, 0);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(dmtx->transpose()), dtrans, 0);
}


TEST_F(Csr, TransposeReuse64IsEquivalentToRef)
{
    SKIP_IF_SINGLE_MODE;
    using Mtx64 = gko::matrix::Csr<value_type, gko::int64>;
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto mtx = gen_mtx<Mtx64>(123, 234, 0);
    auto dmtx = gko::clone(exec, mtx);

    auto [trans, reuse] = mtx->transpose_reuse();
    auto [dtrans, dreuse] = dmtx->transpose_reuse();

    GKO_ASSERT_MTX_NEAR(dtrans, trans, 0);
    ASSERT_TRUE(dtrans->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(dreuse.value_permutation,
                               reuse.value_permutation);
}


TEST_F(Csr, TransposeReuse64UpdateIsEquivalentToRef)
{
    SKIP_IF_SINGLE_MODE;
    using Mtx64 = gko::matrix::Csr<value_type, gko::int64>;
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto mtx = gen_mtx<Mtx64>(123, 234, 0);
    auto dmtx = gko::clone(exec, mtx);
    auto [trans, reuse] = mtx->transpose_reuse();
    auto [dtrans, dreuse] = dmtx->transpose_reuse();
    // test that the value permutation works: modify input values
    mtx->create_value_view()->scale(alpha);
    dmtx->create_value_view()->scale(dalpha);

    reuse.update_values(mtx, trans);
    dreuse.update_values(dmtx, dtrans);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx64>(mtx->transpose()), trans, 0);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx64>(dmtx->transpose()), dtrans, 0);
}


TEST_F(Csr, ConjugateTransposeIsEquivalentToRef)
{
    set_up_apply_complex_data<gko::matrix::csr::spmv_strategy::classical>();

    auto trans = gko::as<ComplexMtx>(complex_mtx->conj_transpose());
    auto dtrans = gko::as<ComplexMtx>(dcomplex_mtx->conj_transpose());

    GKO_ASSERT_MTX_NEAR(dtrans, trans, 0.0);
    ASSERT_TRUE(dtrans->is_sorted_by_column_index());
}


TEST_F(Csr, ConjugateTranspose64IsEquivalentToRef)
{
    using Mtx64 = gko::matrix::Csr<value_type, gko::int64>;
    auto mtx = gen_mtx<Mtx64>(123, 234, 0);
    auto dmtx = gko::clone(exec, mtx);

    auto trans = gko::as<Mtx64>(mtx->transpose());
    auto dtrans = gko::as<Mtx64>(dmtx->transpose());

    GKO_ASSERT_MTX_NEAR(dtrans, trans, 0.0);
    ASSERT_TRUE(dtrans->is_sorted_by_column_index());
}


TEST_F(Csr, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto dense_mtx = gko::matrix::Dense<value_type>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<value_type>::create(exec);

    mtx->convert_to(dense_mtx);
    dmtx->convert_to(ddense_mtx);

    GKO_ASSERT_MTX_NEAR(dense_mtx, ddense_mtx, 0);
}


TEST_F(Csr, MoveToDenseIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto dense_mtx = gko::matrix::Dense<value_type>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<value_type>::create(exec);

    mtx->move_to(dense_mtx);
    dmtx->move_to(ddense_mtx);

    GKO_ASSERT_MTX_NEAR(dense_mtx, ddense_mtx, 0);
}


TEST_F(Csr, ConvertToEllIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto ell_mtx = gko::matrix::Ell<value_type>::create(ref);
    auto dell_mtx = gko::matrix::Ell<value_type>::create(exec);

    mtx->convert_to(ell_mtx);
    dmtx->convert_to(dell_mtx);

    GKO_ASSERT_MTX_NEAR(ell_mtx, dell_mtx, 0);
}


TEST_F(Csr, MoveToEllIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto ell_mtx = gko::matrix::Ell<value_type>::create(ref);
    auto dell_mtx = gko::matrix::Ell<value_type>::create(exec);

    mtx->move_to(ell_mtx);
    dmtx->move_to(dell_mtx);

    GKO_ASSERT_MTX_NEAR(ell_mtx, dell_mtx, 0);
}


TEST_F(Csr, ConvertToSparsityCsrIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto sparsity_mtx = gko::matrix::SparsityCsr<value_type>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<value_type>::create(exec);

    mtx->convert_to(sparsity_mtx);
    dmtx->convert_to(d_sparsity_mtx);

    GKO_ASSERT_MTX_NEAR(sparsity_mtx, d_sparsity_mtx, 0);
}


TEST_F(Csr, MoveToSparsityCsrIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto sparsity_mtx = gko::matrix::SparsityCsr<value_type>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<value_type>::create(exec);

    mtx->move_to(sparsity_mtx);
    dmtx->move_to(d_sparsity_mtx);

    GKO_ASSERT_MTX_NEAR(sparsity_mtx, d_sparsity_mtx, 0);
}


TEST_F(Csr, ConvertToCooIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto coo_mtx = gko::matrix::Coo<value_type>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<value_type>::create(exec);

    mtx->convert_to(coo_mtx);
    dmtx->convert_to(dcoo_mtx);

    GKO_ASSERT_MTX_NEAR(coo_mtx, dcoo_mtx, 0);
}


TEST_F(Csr, MoveToCooIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto coo_mtx = gko::matrix::Coo<value_type>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<value_type>::create(exec);

    mtx->move_to(coo_mtx);
    dmtx->move_to(dcoo_mtx);

    GKO_ASSERT_MTX_NEAR(coo_mtx, dcoo_mtx, 0);
}


TEST_F(Csr, ConvertToSellpIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    auto sellp_mtx = gko::matrix::Sellp<value_type>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<value_type>::create(exec);

    mtx->convert_to(sellp_mtx);
    dmtx->convert_to(dsellp_mtx);

    GKO_ASSERT_MTX_NEAR(sellp_mtx, dsellp_mtx, 0);
}


TEST_F(Csr, MoveToSellpIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

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


TEST_F(Csr, IsGenericReusePermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);
        auto [permuted, reuse] = square_mtx->permute_reuse(rpermutation, mode);
        auto [dpermuted, dreuse] =
            dsquare_mtx->permute_reuse(rpermutation, mode);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
        ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
        GKO_ASSERT_MTX_EQ_SPARSITY(reuse.value_permutation,
                                   dreuse.value_permutation);
    }
}


TEST_F(Csr, IsGenericReusePermuteUpdatable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);
        auto [permuted, reuse] = square_mtx->permute_reuse(rpermutation, mode);
        auto [dpermuted, dreuse] =
            dsquare_mtx->permute_reuse(rpermutation, mode);
        // test that the value permutation works: modify input values
        square_mtx->create_value_view()->scale(alpha);
        dsquare_mtx->create_value_view()->scale(dalpha);

        reuse.update_values(square_mtx, permuted);
        dreuse.update_values(dsquare_mtx, dpermuted);

        GKO_ASSERT_MTX_NEAR(square_mtx->permute(rpermutation, mode), permuted,
                            0);
        GKO_ASSERT_MTX_NEAR(dsquare_mtx->permute(rpermutation, mode), dpermuted,
                            0);
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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    for (auto invert : {false, true}) {
        SCOPED_TRACE(invert);
        auto permuted = mtx->permute(rpermutation, cpermutation, invert);
        auto dpermuted = dmtx->permute(rpermutation, cpermutation, invert);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
        ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
    }
}


TEST_F(Csr, IsNonsymmReusePermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    for (auto invert : {false, true}) {
        SCOPED_TRACE(invert);
        auto [permuted, reuse] =
            mtx->permute_reuse(rpermutation, cpermutation, invert);
        auto [dpermuted, dreuse] =
            dmtx->permute_reuse(rpermutation, cpermutation, invert);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
        ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
        GKO_ASSERT_MTX_EQ_SPARSITY(reuse.value_permutation,
                                   dreuse.value_permutation);
    }
}


TEST_F(Csr, IsNonsymmReusePermuteUpdatable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    for (auto invert : {false, true}) {
        SCOPED_TRACE(invert);
        auto [permuted, reuse] =
            mtx->permute_reuse(rpermutation, cpermutation, invert);
        auto [dpermuted, dreuse] =
            dmtx->permute_reuse(rpermutation, cpermutation, invert);
        // test that the value permutation works: modify input values
        mtx->create_value_view()->scale(alpha);
        dmtx->create_value_view()->scale(dalpha);

        reuse.update_values(mtx, permuted);
        dreuse.update_values(dmtx, dpermuted);

        GKO_ASSERT_MTX_NEAR(mtx->permute(rpermutation, cpermutation, invert),
                            permuted, 0);
        GKO_ASSERT_MTX_NEAR(dmtx->permute(rpermutation, cpermutation, invert),
                            dpermuted, 0);
    }
}


TEST_F(Csr, IsGenericScalePermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    auto permuted = gko::as<Mtx>(square_mtx->permute(rpermute_idxs.get()));
    auto dpermuted = gko::as<Mtx>(dsquare_mtx->permute(rpermute_idxs.get()));

    ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Csr, IsInversePermutable)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    auto permuted =
        gko::as<Mtx>(square_mtx->inverse_permute(rpermute_idxs.get()));
    auto dpermuted =
        gko::as<Mtx>(dsquare_mtx->inverse_permute(rpermute_idxs.get()));

    ASSERT_TRUE(dpermuted->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Csr, IsRowPermutable)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    auto r_permute = gko::as<Mtx>(mtx->row_permute(rpermute_idxs.get()));
    auto dr_permute = gko::as<Mtx>(dmtx->row_permute(rpermute_idxs.get()));

    ASSERT_TRUE(dr_permute->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(r_permute, dr_permute);
    GKO_ASSERT_MTX_NEAR(r_permute, dr_permute, 0);
}


TEST_F(Csr, IsColPermutable)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    auto c_permute = gko::as<Mtx>(mtx->column_permute(cpermute_idxs.get()));
    auto dc_permute = gko::as<Mtx>(dmtx->column_permute(cpermute_idxs.get()));

    ASSERT_TRUE(dc_permute->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(c_permute, dc_permute);
    GKO_ASSERT_MTX_NEAR(c_permute, dc_permute, 0);
}


TEST_F(Csr, IsInverseRowPermutable)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    auto inverse_r_permute =
        gko::as<Mtx>(mtx->inverse_row_permute(rpermute_idxs.get()));
    auto d_inverse_r_permute =
        gko::as<Mtx>(dmtx->inverse_row_permute(rpermute_idxs.get()));

    ASSERT_TRUE(d_inverse_r_permute->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(inverse_r_permute, d_inverse_r_permute);
    GKO_ASSERT_MTX_NEAR(inverse_r_permute, d_inverse_r_permute, 0);
}


TEST_F(Csr, IsInverseColPermutable)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    bool is_sorted_exec{};
    bool is_sorted_ref{};

    is_sorted_ref = mtx->is_sorted_by_column_index();
    is_sorted_exec = dmtx->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_exec);
}


TEST_F(Csr, RecognizeUnsortedMatrixIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    unsort_mtx();
    bool is_sorted_exec{};
    bool is_sorted_ref{};

    is_sorted_ref = mtx->is_sorted_by_column_index();
    is_sorted_exec = dmtx->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_exec);
}


TEST_F(Csr, SortSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
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
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    auto diag = mtx->extract_diagonal();
    auto ddiag = dmtx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 0);
}


TEST_F(Csr, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    mtx->compute_absolute_inplace();
    dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<value_type>::value);
}


TEST_F(Csr, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();

    auto abs_mtx = mtx->compute_absolute();
    auto dabs_mtx = dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, r<value_type>::value);
}


TEST_F(Csr, InplaceAbsoluteComplexMatrixIsEquivalentToRef)
{
    set_up_apply_complex_data<gko::matrix::csr::spmv_strategy::classical>();

    complex_mtx->compute_absolute_inplace();
    dcomplex_mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(complex_mtx, dcomplex_mtx, r<value_type>::value);
}


TEST_F(Csr, OutplaceAbsoluteComplexMatrixIsEquivalentToRef)
{
    set_up_apply_complex_data<gko::matrix::csr::spmv_strategy::classical>();

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
        this->ref, this->mtx2->get_const_device_view(), rspan, cspan, row_nnz);
    gko::kernels::GKO_DEVICE_NAMESPACE::csr::calculate_nonzeros_per_row_in_span(
        this->exec, this->dmtx2->get_const_device_view(), rspan, cspan,
        drow_nnz);

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
        this->ref, this->mtx2->get_const_device_view(), rspan, cspan, row_nnz);
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


    gko::kernels::reference::csr::compute_submatrix(
        this->ref, this->mtx2->get_const_device_view(), rspan, cspan,
        smat1->get_device_view());
    gko::kernels::GKO_DEVICE_NAMESPACE::csr::compute_submatrix(
        this->exec, this->dmtx2->get_const_device_view(), rspan, cspan,
        sdmat1->get_device_view());

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
        this->ref, this->mtx2->get_const_device_view(), rset, cset,
        row_nnz.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::csr::
        calculate_nonzeros_per_row_in_index_set(
            this->exec, this->dmtx2->get_const_device_view(), drset, dcset,
            drow_nnz.get_data());

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
        this->ref, this->mtx2->get_const_device_view(), rset, cset,
        row_nnz.get_data());
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
        this->ref, this->mtx2->get_const_device_view(), rset, cset,
        smat1->get_device_view());
    gko::kernels::GKO_DEVICE_NAMESPACE::csr::compute_submatrix_from_index_set(
        this->exec, this->dmtx2->get_const_device_view(), drset, dcset,
        sdmat1->get_device_view());

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

    gko::kernels::GKO_DEVICE_NAMESPACE::csr::check_diagonal_entries_exist(
        exec, mtx->get_const_device_view(), has_diags);

    ASSERT_FALSE(has_diags);
}


TEST_F(Csr, CanDetectWhenAllDiagonalEntriesArePresent)
{
    using Csr = Mtx;
    auto ref_mtx = gen_mtx<Csr>(103, 98, 10);
    gko::utils::ensure_all_diagonal_entries(ref_mtx.get());
    auto mtx = gko::clone(exec, ref_mtx);
    bool has_diags = true;

    gko::kernels::GKO_DEVICE_NAMESPACE::csr::check_diagonal_entries_exist(
        exec, mtx->get_const_device_view(), has_diags);

    ASSERT_TRUE(has_diags);
}


TEST_F(Csr, AddScaledIdentityToNonSquare)
{
    set_up_apply_data<gko::matrix::csr::spmv_strategy::classical>();
    gko::utils::ensure_all_diagonal_entries(mtx.get());
    dmtx->copy_from(mtx);

    mtx->add_scaled_identity(alpha, beta);
    dmtx->add_scaled_identity(dalpha, dbeta);

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<value_type>::value);
}
