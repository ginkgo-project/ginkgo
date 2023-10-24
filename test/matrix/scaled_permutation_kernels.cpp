/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
    auto inv = permutation->invert();
    auto dinv = dpermutation->invert();

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
    auto combined = permutation->combine(permutation2);
    auto dcombined = dpermutation->combine(permutation2);

    GKO_ASSERT_MTX_NEAR(combined, dcombined, r<value_type>::value);
}
