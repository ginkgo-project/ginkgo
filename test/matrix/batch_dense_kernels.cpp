// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_dense_kernels.hpp"


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/array_generator.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/batch_helpers.hpp"
#include "test/utils/executor.hpp"


class Dense : public CommonTestFixture {
protected:
    using BMtx = gko::batch::matrix::Dense<value_type>;
    using BMVec = gko::batch::MultiVector<value_type>;

    Dense() : rand_engine(15) {}

    template <typename BMtxType>
    std::unique_ptr<BMtxType> gen_mtx(const gko::size_type num_batch_items,
                                      gko::size_type num_rows,
                                      gko::size_type num_cols)
    {
        return gko::test::generate_random_batch_matrix<BMtxType>(
            num_batch_items, num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(gko::size_type num_rows,
                           gko::size_type num_cols = 32,
                           gko::size_type num_vecs = 1)
    {
        mat = gen_mtx<BMtx>(batch_size, num_rows, num_cols);
        mat2 = gen_mtx<BMtx>(batch_size, num_rows, num_cols);
        y = gen_mtx<BMVec>(batch_size, num_cols, num_vecs);
        alpha = gen_mtx<BMVec>(batch_size, 1, 1);
        beta = gen_mtx<BMVec>(batch_size, 1, 1);
        dmat = gko::clone(exec, mat);
        dmat2 = gko::clone(exec, mat2);
        dy = gko::clone(exec, y);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
        row_scale = gko::test::generate_random_array<value_type>(
            num_rows * batch_size, std::normal_distribution<>(2.0, 0.5),
            rand_engine, ref);
        col_scale = gko::test::generate_random_array<value_type>(
            num_cols * batch_size, std::normal_distribution<>(4.0, 0.5),
            rand_engine, ref);
        drow_scale = gko::array<value_type>(exec, row_scale);
        dcol_scale = gko::array<value_type>(exec, col_scale);
        expected = BMVec::create(
            ref,
            gko::batch_dim<2>(batch_size, gko::dim<2>{num_rows, num_vecs}));
        expected->fill(gko::one<value_type>());
        dresult = gko::clone(exec, expected);
    }

    std::default_random_engine rand_engine;

    const gko::size_type batch_size = 11;
    std::unique_ptr<BMtx> mat;
    std::unique_ptr<BMtx> mat2;
    std::unique_ptr<BMVec> y;
    std::unique_ptr<BMVec> alpha;
    std::unique_ptr<BMVec> beta;
    std::unique_ptr<BMVec> expected;
    std::unique_ptr<BMVec> dresult;
    std::unique_ptr<BMtx> dmat;
    std::unique_ptr<BMtx> dmat2;
    std::unique_ptr<BMVec> dy;
    std::unique_ptr<BMVec> dalpha;
    std::unique_ptr<BMVec> dbeta;
    gko::array<value_type> row_scale;
    gko::array<value_type> col_scale;
    gko::array<value_type> drow_scale;
    gko::array<value_type> dcol_scale;
};


TEST_F(Dense, SingleVectorApplyIsEquivalentToRefForSmallMatrices)
{
    set_up_apply_data(10);

    mat->apply(y.get(), expected.get());
    dmat->apply(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Dense, SingleVectorApplyIsEquivalentToRef)
{
    set_up_apply_data(257);

    mat->apply(y.get(), expected.get());
    dmat->apply(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Dense, SingleVectorAdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data(257);

    mat->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmat->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Dense, TwoSidedScaleIsEquivalentToRef)
{
    set_up_apply_data(257);

    mat->scale(row_scale, col_scale);
    dmat->scale(drow_scale, dcol_scale);

    GKO_ASSERT_BATCH_MTX_NEAR(dmat, mat, r<value_type>::value);
}


TEST_F(Dense, ScaleAddIsEquivalentToRef)
{
    set_up_apply_data(42, 42, 15);

    mat->scale_add(alpha, mat2);
    dmat->scale_add(dalpha, dmat2);

    GKO_ASSERT_BATCH_MTX_NEAR(dmat, mat, r<value_type>::value);
}


TEST_F(Dense, AddScaledIdentityIsEquivalentToRef)
{
    set_up_apply_data(42, 42, 15);

    mat->add_scaled_identity(alpha, beta);
    dmat->add_scaled_identity(dalpha, dbeta);

    GKO_ASSERT_BATCH_MTX_NEAR(dmat, mat, r<value_type>::value);
}


TEST_F(Dense, AddScaledIdentityRectMatIsEquivalentToRef)
{
    set_up_apply_data(42, 40, 15);

    mat->add_scaled_identity(alpha, beta);
    dmat->add_scaled_identity(dalpha, dbeta);

    GKO_ASSERT_BATCH_MTX_NEAR(dmat, mat, r<value_type>::value);
}
