// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <numeric>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class Permutation : public CommonTestFixture {
protected:
    using Perm = gko::matrix::Permutation<index_type>;
    using Mtx = gko::matrix::Dense<value_type>;

    Permutation() : rand_engine(42)
    {
        std::vector<int> tmp(1000, 0);
        std::iota(tmp.begin(), tmp.end(), 0);
        auto tmp2 = tmp;
        std::shuffle(tmp.begin(), tmp.end(), rand_engine);
        std::shuffle(tmp2.begin(), tmp2.end(), rand_engine);
        permutation = Perm::create(
            ref, gko::array<index_type>(ref, tmp.begin(), tmp.end()));
        permutation2 = Perm::create(
            ref, gko::array<index_type>(ref, tmp2.begin(), tmp2.end()));
        dpermutation = permutation->clone(exec);

        mtx = gko::test::generate_random_matrix<Mtx>(
            tmp.size(), 4, std::uniform_int_distribution<>(4, 4),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
        mtx2 = gko::test::generate_random_matrix<Mtx>(
            tmp.size(), 4, std::uniform_int_distribution<>(4, 4),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-3.0}, ref);
        dmtx = mtx->clone();
    }

    std::default_random_engine rand_engine;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Perm> permutation;
    std::unique_ptr<Perm> permutation2;
    std::unique_ptr<Perm> dpermutation;
};


TEST_F(Permutation, InvertIsEquivalentToRef)
{
    auto inv = permutation->compute_inverse();
    auto dinv = dpermutation->compute_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(inv, dinv);
}


TEST_F(Permutation, ApplyIsEquivalentToRef)
{
    auto out = mtx->clone();
    auto dout = dmtx->clone();

    permutation->apply(mtx, out);
    dpermutation->apply(dmtx, dout);

    GKO_ASSERT_MTX_NEAR(out, dout, 0.0);
}


TEST_F(Permutation, AdvancedApplyIsEquivalentToRef)
{
    auto out = mtx->clone();
    auto dout = dmtx->clone();

    permutation->apply(alpha, mtx, beta, out);
    dpermutation->apply(alpha, dmtx, beta, dout);

    GKO_ASSERT_MTX_NEAR(out, dout, r<value_type>::value);
}


TEST_F(Permutation, CombineIsEquivalentToRef)
{
    auto combined = permutation->compose(permutation2);
    auto dcombined = dpermutation->compose(permutation2);

    GKO_ASSERT_MTX_EQ_SPARSITY(combined, dcombined);
}
