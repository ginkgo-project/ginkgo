/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/matrix/permutation.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class Permutation : public ::testing::Test {
protected:
    using i_type = int;
    using v_type = double;
    using Vec = gko::matrix::Dense<v_type>;
    using Csr = gko::matrix::Csr<v_type, i_type>;
    Permutation() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};


TEST_F(Permutation, AppliesRowPermutationToDense)
{
    // clang-format off
    auto x = gko::initialize<Vec>(
                                  {{2.0, 3.0},
                                   {4.0, 2.5}}, exec);
    // clang-format on
    auto y = Vec::create(exec, gko::dim<2>{2});
    i_type rdata[] = {1, 0};

    auto perm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{2}, gko::Array<i_type>::view(exec, 2, rdata));

    perm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                        l({{4.0, 2.5},
                           {2.0, 3.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesColPermutationToDense)
{
    // clang-format off
    auto x = gko::initialize<Vec>(
                                  {{2.0, 3.0},
                                   {4.0, 2.5}}, exec);
    // clang-format on
    auto y = Vec::create(exec, gko::dim<2>{2});
    i_type rdata[] = {1, 0};

    auto perm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{2}, gko::Array<i_type>::view(exec, 2, rdata), false,
        false, true);

    perm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                        l({{3.0, 2.0},
                           {2.5, 4.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesRowAndColPermutationToDense)
{
    // clang-format off
    auto x = gko::initialize<Vec>(
                                  {{2.0, 3.0},
                                   {4.0, 2.5}}, exec);
    // clang-format on
    auto y1 = Vec::create(exec, gko::dim<2>{2});
    auto y2 = Vec::create(exec, gko::dim<2>{2});
    i_type cdata[] = {1, 0};
    i_type rdata[] = {1, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{2}, gko::Array<i_type>::view(exec, 2, rdata));
    auto cperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{2}, gko::Array<i_type>::view(exec, 2, cdata), false,
        false, true);

    rperm->apply(x.get(), y1.get());
    cperm->apply(y1.get(), y2.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y2.get(),
                        l({{2.5, 4.0},
                           {3.0, 2.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesInverseRowAndColPermutationToDense)
{
    // clang-format off
  auto x = gko::initialize<Vec>({{2.0, 3.0, 0.0},
                                {0.0, 1.0, 0.0},
                                {0.0, 4.0, 2.5}},
                               exec);
    // clang-format on
    auto y1 = Vec::create(exec, gko::dim<2>{3});
    auto y2 = Vec::create(exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};
    i_type rdata[] = {1, 2, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, rdata), true);
    auto cperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, cdata), true,
        false, true);

    rperm->apply(x.get(), y1.get());
    cperm->apply(y1.get(), y2.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y2.get(),
                        l({{2.5, 0.0, 4.0},
                           {0.0, 2.0, 3.0},
                           {0.0, 0.0, 1.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesInverseRowPermutationToDense)
{
    // clang-format off
    auto x = gko::initialize<Vec>({{2.0, 3.0, 0.0},
                                 {0.0, 1.0, 0.0},
                                 {0.0, 4.0, 2.5}},
                                exec);
    // clang-format on
    auto y = Vec::create(exec, gko::dim<2>{3});
    i_type rdata[] = {1, 2, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, rdata), true);

    rperm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                        l({{0.0, 4.0, 2.5},
                           {2.0, 3.0, 0.0},
                           {0.0, 1.0, 0.0}}),
                          0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesInverseColPermutationToDense)
{
    // clang-format off
    auto x = gko::initialize<Vec>({{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  exec);
    // clang-format on
    auto y = Vec::create(exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};

    auto cperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, cdata), true,
        false, true);

    cperm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                      l({{0.0, 2.0, 3.0},
                         {0.0, 0.0, 1.0},
                         {2.5, 0.0, 4.0}}),
                      0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesRowPermutationToCsr)
{
    // clang-format off
    auto x = gko::initialize<Csr>(
                                  {{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  exec);
    // clang-format on
    auto y = Csr::create(exec, gko::dim<2>{3});
    i_type rdata[] = {1, 2, 0};

    auto perm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, rdata));

    perm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                        l({{0.0, 1.0, 0.0},
                           {0.0, 4.0, 2.5},
                           {2.0, 3.0, 0.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesColPermutationToCsr)
{
    // clang-format off
    auto x = gko::initialize<Csr>(
                                  {{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  exec);
    // clang-format on
    auto y = Csr::create(exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};

    auto perm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, cdata), false,
        false, true);

    perm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                      l({{3.0, 0.0, 2.0},
                         {1.0, 0.0, 0.0},
                         {4.0, 2.5, 0.0}}),
                      0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesRowAndColPermutationToCsr)
{
    // clang-format off
    auto x = gko::initialize<Csr>(
                                  {{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  exec);
    // clang-format on
    auto y1 = Csr::create(exec, gko::dim<2>{3});
    auto y2 = Csr::create(exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};
    i_type rdata[] = {1, 2, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, rdata));
    auto cperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, cdata), false,
        false, true);

    rperm->apply(x.get(), y1.get());
    cperm->apply(y1.get(), y2.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y2.get(),
                      l({{1.0, 0.0, 0.0},
                         {4.0, 2.5, 0.0},
                         {3.0, 0.0, 2.0}}),
                      0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesInverseRowPermutationToCsr)
{
    // clang-format off
    auto x = gko::initialize<Csr>({{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  exec);
    // clang-format on
    auto y = Csr::create(exec, gko::dim<2>{3});
    i_type rdata[] = {1, 2, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, rdata), true);

    rperm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                        l({{0.0, 4.0, 2.5},
                           {2.0, 3.0, 0.0},
                           {0.0, 1.0, 0.0}}),
                          0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesInverseColPermutationToCsr)
{
    // clang-format off
    auto x = gko::initialize<Csr>({{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  exec);
    // clang-format on
    auto y = Csr::create(exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};

    auto cperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, cdata), true,
        false, true);

    cperm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                      l({{0.0, 2.0, 3.0},
                         {0.0, 0.0, 1.0},
                         {2.5, 0.0, 4.0}}),
                      0.0);
    // clang-format on
}


TEST_F(Permutation, AppliesInverseRowAndColPermutationToCsr)
{
    // clang-format off
    auto x = gko::initialize<Csr>({{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  exec);
    // clang-format on
    auto y1 = Csr::create(exec, gko::dim<2>{3});
    auto y2 = Csr::create(exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};
    i_type rdata[] = {1, 2, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, rdata), true);
    auto cperm = gko::matrix::Permutation<i_type>::create(
        exec, gko::dim<2>{3}, gko::Array<i_type>::view(exec, 3, cdata), true,
        false, true);

    rperm->apply(x.get(), y1.get());
    cperm->apply(y1.get(), y2.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y2.get(),
                        l({{2.5, 0.0, 4.0},
                           {0.0, 2.0, 3.0},
                           {0.0, 0.0, 1.0}}),
                        0.0);
    // clang-format on
}


}  // namespace
