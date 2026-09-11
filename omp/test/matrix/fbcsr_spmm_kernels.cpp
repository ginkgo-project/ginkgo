// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"


namespace {


class FbcsrSpmm : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    static constexpr index_type num_brows = 50;
    static constexpr index_type num_bcols = 38;
    static constexpr int blk_sz = 2;
    static constexpr gko::size_type num_rows = num_brows * blk_sz;
    static constexpr gko::size_type num_cols = num_bcols * blk_sz;
    static constexpr gko::size_type num_rhs = 16;
    static constexpr value_type tolerance = 1e-12;

    FbcsrSpmm() : rand_engine(42) {}

    void SetUp() override
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown() override
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    template <typename VecType>
    std::unique_ptr<VecType> gen_dense(gko::size_type rows, gko::size_type cols,
                                       int min_nnz_row)
    {
        return gko::test::generate_random_matrix<VecType>(
            rows, cols,
            std::uniform_int_distribution<>(min_nnz_row,
                                            static_cast<int>(cols)),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data()
    {
        rand_engine.seed(42);
        mtx = gko::test::generate_random_fbcsr<value_type>(
            ref, num_brows, num_bcols, blk_sz, false, false, rand_engine);
        y = gen_dense<Vec>(num_cols, num_rhs, 1);
        expected = Vec::create(ref, gko::dim<2>{num_rows, num_rhs});
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.5}, ref);
        dmtx = Mtx::create(omp);
        dmtx->copy_from(mtx);
        dy = Vec::create(omp);
        dy->copy_from(y);
        dresult = Vec::create(omp, gko::dim<2>{num_rows, num_rhs});
        dalpha = Vec::create(omp);
        dalpha->copy_from(alpha);
        dbeta = Vec::create(omp);
        dbeta->copy_from(beta);
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;
    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;
    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(FbcsrSpmm, ApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, tolerance);
}


TEST_F(FbcsrSpmm, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, tolerance);
}


}  // namespace
