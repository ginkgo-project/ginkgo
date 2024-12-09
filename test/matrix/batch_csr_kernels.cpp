// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_csr_kernels.hpp"

#include <memory>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>

#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/array_generator.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/batch_helpers.hpp"
#include "test/utils/common_fixture.hpp"


class Csr : public CommonTestFixture {
protected:
    using BMtx = gko::batch::matrix::Csr<value_type, gko::int32>;
    using BMVec = gko::batch::MultiVector<value_type>;

    Csr() : rand_engine(15) {}

    template <typename BMtxType>
    std::unique_ptr<BMtxType> gen_mtx(const gko::size_type num_batch_items,
                                      gko::size_type num_rows,
                                      gko::size_type num_cols,
                                      int num_elems_per_row,
                                      bool diag_dominant = false)
    {
        if (diag_dominant) {
            return gko::test::generate_diag_dominant_batch_matrix<BMtxType>(
                ref, num_batch_items, num_rows, false, 4 * num_rows - 3);

        } else {
            return gko::test::generate_random_batch_matrix<BMtxType>(
                num_batch_items, num_rows, num_cols,
                std::uniform_int_distribution<>(num_elems_per_row,
                                                num_elems_per_row),
                std::normal_distribution<>(-1.0, 1.0), rand_engine, ref,
                num_elems_per_row * num_rows);
        }
    }

    std::unique_ptr<BMVec> gen_mvec(const gko::size_type num_batch_items,
                                    gko::size_type num_rows,
                                    gko::size_type num_cols)
    {
        return gko::test::generate_random_batch_matrix<BMVec>(
            num_batch_items, num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(gko::size_type num_vecs = 1,
                           int num_elems_per_row = 5,
                           gko::size_type num_rows = 252,
                           gko::size_type num_cols = 32,
                           bool diag_dominant = false)
    {
        GKO_ASSERT(num_elems_per_row <= num_cols);

        mat = gen_mtx<BMtx>(batch_size, num_rows, num_cols, num_elems_per_row,
                            diag_dominant);
        y = gen_mvec(batch_size, num_cols, num_vecs);
        alpha = gen_mvec(batch_size, 1, 1);
        beta = gen_mvec(batch_size, 1, 1);
        dmat = gko::clone(exec, mat);
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
    std::unique_ptr<BMVec> y;
    std::unique_ptr<BMVec> alpha;
    std::unique_ptr<BMVec> beta;
    std::unique_ptr<BMVec> expected;
    std::unique_ptr<BMVec> dresult;
    std::unique_ptr<BMtx> dmat;
    std::unique_ptr<BMVec> dy;
    std::unique_ptr<BMVec> dalpha;
    std::unique_ptr<BMVec> dbeta;
    gko::array<value_type> row_scale;
    gko::array<value_type> col_scale;
    gko::array<value_type> drow_scale;
    gko::array<value_type> dcol_scale;
};


TEST_F(Csr, SingleVectorApplyIsEquivalentToRef)
{
    set_up_apply_data(1);

    mat->apply(y.get(), expected.get());
    dmat->apply(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, SingleVectorAdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data(1);

    mat->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmat->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Csr, TwoSidedScaleIsEquivalentToRef)
{
    set_up_apply_data(257);

    mat->scale(row_scale, col_scale);
    dmat->scale(drow_scale, dcol_scale);

    GKO_ASSERT_BATCH_MTX_NEAR(dmat, mat, r<value_type>::value);
}


TEST_F(Csr, AddScaledIdentityIsEquivalentToRef)
{
    set_up_apply_data(2, 5, 151, 151, true);

    mat->add_scaled_identity(alpha, beta);
    dmat->add_scaled_identity(dalpha, dbeta);

    GKO_ASSERT_BATCH_MTX_NEAR(dmat, mat, r<value_type>::value);
}


TEST_F(Csr, AddScaledIdentityWithRecMatIsEquivalentToRef)
{
    set_up_apply_data(2, 5, 151, 148, true);

    mat->add_scaled_identity(alpha, beta);
    dmat->add_scaled_identity(dalpha, dbeta);

    GKO_ASSERT_BATCH_MTX_NEAR(dmat, mat, r<value_type>::value);
}
