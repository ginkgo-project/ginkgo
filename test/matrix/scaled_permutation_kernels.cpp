// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <numeric>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/scaled_permutation.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class ScaledPermutation : public CommonTestFixture {
protected:
    using ScaledPerm = gko::matrix::ScaledPermutation<value_type, index_type>;
    using Mtx = gko::matrix::Dense<value_type>;

    ScaledPermutation() : rand_engine(42)
    {
        std::vector<int> tmp(1000, 0);
        std::iota(tmp.begin(), tmp.end(), 0);
        auto tmp2 = tmp;
        std::shuffle(tmp.begin(), tmp.end(), rand_engine);
        std::shuffle(tmp2.begin(), tmp2.end(), rand_engine);
        std::vector<value_type> scale(tmp.size());
        std::vector<value_type> scale2(tmp2.size());
        std::uniform_real_distribution<value_type> dist(1, 2);
        auto gen = [&] { return dist(rand_engine); };
        std::generate(scale.begin(), scale.end(), gen);
        std::generate(scale2.begin(), scale2.end(), gen);
        permutation = ScaledPerm::create(
            ref, gko::array<value_type>(ref, scale.begin(), scale.end()),
            gko::array<index_type>(ref, tmp.begin(), tmp.end()));
        permutation2 = ScaledPerm::create(
            ref, gko::array<value_type>(ref, scale2.begin(), scale2.end()),
            gko::array<index_type>(ref, tmp2.begin(), tmp2.end()));
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
    std::unique_ptr<ScaledPerm> permutation;
    std::unique_ptr<ScaledPerm> permutation2;
    std::unique_ptr<ScaledPerm> dpermutation;
};


TEST_F(ScaledPermutation, InvertIsEquivalentToRef)
{
    auto inv = permutation->compute_inverse();
    auto dinv = dpermutation->compute_inverse();

    GKO_ASSERT_MTX_NEAR(inv, dinv, r<value_type>::value);
}


TEST_F(ScaledPermutation, ApplyIsEquivalentToRef)
{
    auto out = mtx->clone();
    auto dout = dmtx->clone();

    permutation->apply(mtx, out);
    dpermutation->apply(dmtx, dout);

    GKO_ASSERT_MTX_NEAR(out, dout, r<value_type>::value);
}


TEST_F(ScaledPermutation, AdvancedApplyIsEquivalentToRef)
{
    auto out = mtx->clone();
    auto dout = dmtx->clone();

    permutation->apply(alpha, mtx, beta, out);
    dpermutation->apply(alpha, dmtx, beta, dout);

    GKO_ASSERT_MTX_NEAR(out, dout, r<value_type>::value);
}


TEST_F(ScaledPermutation, CombineIsEquivalentToRef)
{
    auto combined = permutation->compose(permutation2);
    auto dcombined = dpermutation->compose(permutation2);

    GKO_ASSERT_MTX_NEAR(combined, dcombined, r<value_type>::value);
}
