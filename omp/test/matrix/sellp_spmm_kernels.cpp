// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sellp.hpp>

#include "core/test/utils.hpp"


namespace {


class SellpSpmm : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using Mtx = gko::matrix::Sellp<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    static constexpr gko::size_type num_rows = 200;
    static constexpr gko::size_type num_cols = 150;
    static constexpr gko::size_type num_rhs = 16;
    static constexpr value_type tolerance = 1e-12;

    SellpSpmm() : rand_engine(42) {}

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

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(gko::size_type rows, gko::size_type cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            rows, cols,
            std::uniform_int_distribution<>(min_nnz_row,
                                            static_cast<int>(cols)),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data()
    {
        rand_engine.seed(42);
        mtx = gen_mtx<Mtx>(num_rows, num_cols, 5);
        y = gen_mtx<Vec>(num_cols, num_rhs, 1);
        expected = gen_mtx<Vec>(num_rows, num_rhs, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.5}, ref);
        dmtx = Mtx::create(omp);
        dmtx->copy_from(mtx);
        dy = Vec::create(omp);
        dy->copy_from(y);
        dresult = Vec::create(omp);
        dresult->copy_from(expected);
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


TEST_F(SellpSpmm, ApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y, expected);
    dmtx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, tolerance);
}


TEST_F(SellpSpmm, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha, y, beta, expected);
    dmtx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, expected, tolerance);
}


}  // namespace
