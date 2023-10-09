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

#include <ginkgo/core/matrix/permutation.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Permutation : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Vec = gko::matrix::Dense<value_type>;

    Permutation() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(Permutation, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(Permutation, Invert)
{
    using index_type = typename TestFixture::index_type;
    auto perm = gko::matrix::Permutation<index_type>::create(
        this->exec, gko::array<index_type>{this->exec, {1, 2, 0}});

    auto inv = perm->invert();

    EXPECT_EQ(inv->get_const_permutation()[0], 2);
    EXPECT_EQ(inv->get_const_permutation()[1], 0);
    EXPECT_EQ(inv->get_const_permutation()[2], 1);
}


TYPED_TEST(Permutation, Write)
{
    using index_type = typename TestFixture::index_type;
    auto perm = gko::matrix::Permutation<index_type>::create(
        this->exec, 3, gko::array<index_type>{this->exec, {1, 2, 0}});

    GKO_ASSERT_MTX_NEAR(
        perm, l<double>({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}}),
        0.0);
}


TYPED_TEST(Permutation, AppliesRowPermutationToDense)
{
    using index_type = typename TestFixture::index_type;
    using T = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{2});
    index_type rdata[] = {1, 0};

    auto perm = gko::matrix::Permutation<index_type>::create(
        this->exec, gko::make_array_view(this->exec, 2, rdata));

    perm->apply(x, y);
    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{4.0, 2.5},
                           {2.0, 3.0}}),
                        0.0);
    // clang-format on
}


}  // namespace
