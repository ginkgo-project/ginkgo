/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Permutation : public ::testing::Test {
protected:
    using v_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using i_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Vec = gko::matrix::Dense<v_type>;
    using Csr = gko::matrix::Csr<v_type, i_type>;

    Permutation() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_CASE(Permutation, gko::test::ValueIndexTypes);


TYPED_TEST(Permutation, AppliesRowPermutationToDense)
{
    using i_type = typename TestFixture::i_type;
    using T = typename TestFixture::v_type;
    using Vec = typename TestFixture::Vec;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{2});
    i_type rdata[] = {1, 0};

    auto perm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{2},
        gko::Array<i_type>::view(this->exec, 2, rdata));

    perm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                        l({{4.0, 2.5},
                           {2.0, 3.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesColPermutationToDense)
{
    using i_type = typename TestFixture::i_type;
    using T = typename TestFixture::v_type;
    using Vec = typename TestFixture::Vec;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{2});
    i_type rdata[] = {1, 0};

    auto perm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{2},
        gko::Array<i_type>::view(this->exec, 2, rdata),
        gko::matrix::column_permute);

    perm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                        l({{3.0, 2.0},
                           {2.5, 4.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesRowAndColPermutationToDense)
{
    using i_type = typename TestFixture::i_type;
    using T = typename TestFixture::v_type;
    using Vec = typename TestFixture::Vec;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y1 = Vec::create(this->exec, gko::dim<2>{2});
    auto y2 = Vec::create(this->exec, gko::dim<2>{2});
    i_type cdata[] = {1, 0};
    i_type rdata[] = {1, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{2},
        gko::Array<i_type>::view(this->exec, 2, rdata));
    auto cperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{2},
        gko::Array<i_type>::view(this->exec, 2, cdata),
        gko::matrix::column_permute);

    rperm->apply(x.get(), y1.get());
    cperm->apply(y1.get(), y2.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y2.get(),
                        l({{2.5, 4.0},
                           {3.0, 2.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesRowAndColPermutationToDenseWithOneArray)
{
    using i_type = typename TestFixture::i_type;
    using T = typename TestFixture::v_type;
    using Vec = typename TestFixture::Vec;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y1 = Vec::create(this->exec, gko::dim<2>{2});
    i_type data[] = {1, 0};

    auto perm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{2},
        gko::Array<i_type>::view(this->exec, 2, data),
        gko::matrix::row_permute | gko::matrix::column_permute);

    perm->apply(x.get(), y1.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y1.get(),
                        l({{2.5, 4.0},
                           {3.0, 2.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesInverseRowAndColPermutationToDense)
{
    using i_type = typename TestFixture::i_type;
    using Vec = typename TestFixture::Vec;
    // clang-format off
    auto x = gko::initialize<Vec>({{2.0, 3.0, 0.0},
                                  {0.0, 1.0, 0.0},
                                  {0.0, 4.0, 2.5}},
                                  this->exec);
    // clang-format on
    auto y1 = Vec::create(this->exec, gko::dim<2>{3});
    auto y2 = Vec::create(this->exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};
    i_type rdata[] = {1, 2, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, rdata),
        gko::matrix::row_permute | gko::matrix::inverse_permute);
    auto cperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, cdata),
        gko::matrix::inverse_permute | gko::matrix::column_permute);

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


TYPED_TEST(Permutation, AppliesInverseRowAndColPermutationToDenseWithOneArray)
{
    using i_type = typename TestFixture::i_type;
    using Vec = typename TestFixture::Vec;
    // clang-format off
    auto x = gko::initialize<Vec>({{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                 this->exec);
    // clang-format on
    auto y1 = Vec::create(this->exec, gko::dim<2>{3});
    i_type data[] = {1, 2, 0};

    auto perm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, data),
        gko::matrix::column_permute | gko::matrix::row_permute |
            gko::matrix::inverse_permute);

    perm->apply(x.get(), y1.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y1.get(),
                        l({{2.5, 0.0, 4.0},
                           {0.0, 2.0, 3.0},
                           {0.0, 0.0, 1.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesInverseRowPermutationToDense)
{
    using i_type = typename TestFixture::i_type;
    using Vec = typename TestFixture::Vec;
    // clang-format off
    auto x = gko::initialize<Vec>({{2.0, 3.0, 0.0},
                                 {0.0, 1.0, 0.0},
                                 {0.0, 4.0, 2.5}},
                                this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{3});
    i_type rdata[] = {1, 2, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, rdata),
        gko::matrix::row_permute | gko::matrix::inverse_permute);

    rperm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                        l({{0.0, 4.0, 2.5},
                           {2.0, 3.0, 0.0},
                           {0.0, 1.0, 0.0}}),
                          0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesInverseColPermutationToDense)
{
    using i_type = typename TestFixture::i_type;
    using Vec = typename TestFixture::Vec;
    // clang-format off
    auto x = gko::initialize<Vec>({{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};

    auto cperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, cdata),
        gko::matrix::inverse_permute | gko::matrix::column_permute);

    cperm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                      l({{0.0, 2.0, 3.0},
                         {0.0, 0.0, 1.0},
                         {2.5, 0.0, 4.0}}),
                      0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesRowPermutationToCsr)
{
    using i_type = typename TestFixture::i_type;
    using Csr = typename TestFixture::Csr;
    // clang-format off
    auto x = gko::initialize<Csr>(
                                  {{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  this->exec);
    // clang-format on
    auto y = Csr::create(this->exec, gko::dim<2>{3});
    i_type rdata[] = {1, 2, 0};

    auto perm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, rdata));

    perm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                        l({{0.0, 1.0, 0.0},
                           {0.0, 4.0, 2.5},
                           {2.0, 3.0, 0.0}}),
                        0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesColPermutationToCsr)
{
    using i_type = typename TestFixture::i_type;
    using Csr = typename TestFixture::Csr;
    // clang-format off
    auto x = gko::initialize<Csr>(
                                  {{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  this->exec);
    // clang-format on
    auto y = Csr::create(this->exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};

    auto perm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, cdata),
        gko::matrix::column_permute);

    perm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                      l({{3.0, 0.0, 2.0},
                         {1.0, 0.0, 0.0},
                         {4.0, 2.5, 0.0}}),
                      0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesRowAndColPermutationToCsr)
{
    using i_type = typename TestFixture::i_type;
    using Csr = typename TestFixture::Csr;
    // clang-format off
    auto x = gko::initialize<Csr>(
                                  {{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  this->exec);
    // clang-format on
    auto y1 = Csr::create(this->exec, gko::dim<2>{3});
    auto y2 = Csr::create(this->exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};
    i_type rdata[] = {1, 2, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, rdata));
    auto cperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, cdata),
        gko::matrix::column_permute);

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


TYPED_TEST(Permutation, AppliesInverseRowPermutationToCsr)
{
    using i_type = typename TestFixture::i_type;
    using Csr = typename TestFixture::Csr;
    // clang-format off
    auto x = gko::initialize<Csr>({{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  this->exec);
    // clang-format on
    auto y = Csr::create(this->exec, gko::dim<2>{3});
    i_type rdata[] = {1, 2, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, rdata),
        gko::matrix::row_permute | gko::matrix::inverse_permute);

    rperm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                        l({{0.0, 4.0, 2.5},
                           {2.0, 3.0, 0.0},
                           {0.0, 1.0, 0.0}}),
                          0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesInverseColPermutationToCsr)
{
    using i_type = typename TestFixture::i_type;
    using Csr = typename TestFixture::Csr;
    // clang-format off
    auto x = gko::initialize<Csr>({{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  this->exec);
    // clang-format on
    auto y = Csr::create(this->exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};

    auto cperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, cdata),
        gko::matrix::inverse_permute | gko::matrix::column_permute);

    cperm->apply(x.get(), y.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y.get(),
                      l({{0.0, 2.0, 3.0},
                         {0.0, 0.0, 1.0},
                         {2.5, 0.0, 4.0}}),
                      0.0);
    // clang-format on
}


TYPED_TEST(Permutation, AppliesInverseRowAndColPermutationToCsr)
{
    using i_type = typename TestFixture::i_type;
    using Csr = typename TestFixture::Csr;
    // clang-format off
    auto x = gko::initialize<Csr>({{2.0, 3.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 4.0, 2.5}},
                                  this->exec);
    // clang-format on
    auto y1 = Csr::create(this->exec, gko::dim<2>{3});
    auto y2 = Csr::create(this->exec, gko::dim<2>{3});
    i_type cdata[] = {1, 2, 0};
    i_type rdata[] = {1, 2, 0};

    auto rperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, rdata),
        gko::matrix::row_permute | gko::matrix::inverse_permute);
    auto cperm = gko::matrix::Permutation<i_type>::create(
        this->exec, gko::dim<2>{3},
        gko::Array<i_type>::view(this->exec, 3, cdata),
        gko::matrix::inverse_permute | gko::matrix::column_permute);

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
