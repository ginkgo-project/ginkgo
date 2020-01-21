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

#include <ginkgo/core/matrix/identity.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include <core/test/utils.hpp>


namespace {


template <typename T>
class Identity : public ::testing::Test {
protected:
    using value_type = T;
    using Id = gko::matrix::Identity<value_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Identity() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};


TYPED_TEST_CASE(Identity, gko::test::ValueTypes);


TYPED_TEST(Identity, AppliesLinearCombinationToVector)
{
    using Id = typename TestFixture::Id;
    using Vec = typename TestFixture::Vec;
    auto identity = Id::create(this->exec, 3);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({1.0}, this->exec);
    auto x = gko::initialize<Vec>({3.0, -1.0, 2.0}, this->exec);
    auto b = gko::initialize<Vec>({2.0, 1.0, 5.0}, this->exec);

    identity->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({7.0, 1.0, 12.0}), 0.0);
}


TYPED_TEST(Identity, AppliesLinearCombinationToMultipleVectors)
{
    using Id = typename TestFixture::Id;
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto identity = Id::create(this->exec, 3);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({1.0}, this->exec);
    auto x = gko::initialize<Vec>(
        3, {I<T>{3.0, 0.5}, I<T>{-1.0, 2.5}, I<T>{2.0, 3.4}}, this->exec);
    auto b = gko::initialize<Vec>(
        3, {I<T>{2.0, 3.0}, I<T>{1.0, 2.0}, I<T>{5.0, -1.0}}, this->exec);

    identity->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({{7.0, 6.5}, {1.0, 6.5}, {12.0, 1.4}}),
                        r<TypeParam>::value);
}


}  // namespace
