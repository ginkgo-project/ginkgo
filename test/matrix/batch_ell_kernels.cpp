// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_ell_kernels.hpp"


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/batch_helpers.hpp"
#include "test/utils/executor.hpp"


class Ell : public CommonTestFixture {
protected:
    using BMtx = gko::batch::matrix::Ell<value_type, gko::int32>;
    using BMVec = gko::batch::MultiVector<value_type>;

    Ell() : rand_engine(15) {}

    template <typename BMtxType>
    std::unique_ptr<BMtxType> gen_mtx(const gko::size_type num_batch_items,
                                      gko::size_type num_rows,
                                      gko::size_type num_cols,
                                      int num_elems_per_row)
    {
        return gko::test::generate_random_batch_matrix<BMtxType>(
            num_batch_items, num_rows, num_cols,
            std::uniform_int_distribution<>(num_elems_per_row,
                                            num_elems_per_row),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref,
            num_elems_per_row);
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
                           int num_elems_per_row = 5)
    {
        const gko::size_type num_rows = 252;
        const gko::size_type num_cols = 32;
        GKO_ASSERT(num_elems_per_row <= num_cols);
        mat = gen_mtx<BMtx>(batch_size, num_rows, num_cols, num_elems_per_row);
        y = gen_mvec(batch_size, num_cols, num_vecs);
        alpha = gen_mvec(batch_size, 1, 1);
        beta = gen_mvec(batch_size, 1, 1);
        dmat = gko::clone(exec, mat);
        dy = gko::clone(exec, y);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
        expected = BMVec::create(
            ref,
            gko::batch_dim<2>(batch_size, gko::dim<2>{num_rows, num_vecs}));
        expected->fill(gko::one<value_type>());
        dresult = gko::clone(exec, expected);
    }

    std::ranlux48 rand_engine;

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
};


TEST_F(Ell, SingleVectorApplyIsEquivalentToRef)
{
    set_up_apply_data(1);

    mat->apply(y.get(), expected.get());
    dmat->apply(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Ell, SingleVectorAdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data(1);

    mat->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmat->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}
