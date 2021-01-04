/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/base/composition.hpp>


#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueType>
class DummyLinOp : public gko::EnableLinOp<DummyLinOp<ValueType>>,
                   public gko::EnableCreateMethod<DummyLinOp<ValueType>> {
    friend class gko::EnablePolymorphicObject<DummyLinOp, gko::LinOp>;
    friend class gko::EnableCreateMethod<DummyLinOp>;

public:
    using value_type = ValueType;

    bool apply_uses_initial_guess() const override { return true; }

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override {}

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {}

    explicit DummyLinOp(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyLinOp>(exec)
    {}

    explicit DummyLinOp(std::shared_ptr<const gko::Executor> exec,
                        gko::dim<2> size)
        : gko::EnableLinOp<DummyLinOp>(exec, size)
    {}
};


template <typename T>
class Composition : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;
    using value_type = T;

    Composition()
        : exec{gko::ReferenceExecutor::create()},
          operators{
              gko::initialize<Mtx>(I<T>({2.0, 1.0}), exec),
              gko::initialize<Mtx>({I<T>({3.0, 2.0})}, exec),
              gko::initialize<Mtx>(
                  {I<T>({-1.0, 1.0, 2.0}), I<T>({5.0, -3.0, 0.0})}, exec),
              gko::initialize<Mtx>(
                  {I<T>({9.0, 4.0}), I<T>({6.0, -2.0}), I<T>({-3.0, 2.0})},
                  exec),
              gko::initialize<Mtx>({I<T>({1.0, 0.0}), I<T>({0.0, 1.0})}, exec),
              gko::initialize<Mtx>({I<T>({1.0, 0.0}), I<T>({0.0, 1.0})}, exec)},
          identity{
              gko::initialize<Mtx>({I<T>({1.0, 0.0}), I<T>({0.0, 1.0})}, exec)},
          product{gko::initialize<Mtx>({I<T>({-9.0, -2.0}), I<T>({27.0, 26.0})},
                                       exec)}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::vector<std::shared_ptr<gko::LinOp>> coefficients;
    std::vector<std::shared_ptr<gko::LinOp>> operators;
    std::shared_ptr<Mtx> identity;
    std::shared_ptr<Mtx> product;
};

TYPED_TEST_SUITE(Composition, gko::test::ValueTypes);


TYPED_TEST(Composition, AppliesSingleToVector)
{
    /*
        cmp = [ -9 -2 ]
              [ 27 26 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Composition<TypeParam>::create(this->product);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({-13.0, 79.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesSingleLinearCombinationToVector)
{
    /*
        cmp = [ -9 -2 ]
              [ 27 26 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Composition<TypeParam>::create(this->product);
    auto alpha = gko::initialize<Mtx>({3.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(alpha), lend(x), lend(beta), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({-40.0, 235.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ]
              [ 1 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Composition<TypeParam>::create(this->operators[0],
                                                   this->operators[1]);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({14.0, 7.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLinearCombinationToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ]
              [ 1 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Composition<TypeParam>::create(this->operators[0],
                                                   this->operators[1]);
    auto alpha = gko::initialize<Mtx>({3.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(alpha), lend(x), lend(beta), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({41.0, 19.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongerToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -9  -2 ]
              [ 1 ]             [ 27  26 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Composition<TypeParam>::create(
        this->operators[0], this->operators[1], this->product);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({238.0, 119.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongerLinearCombinationToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -9  -2 ]
              [ 1 ]             [ 27  26 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Composition<TypeParam>::create(
        this->operators[0], this->operators[1], this->product);
    auto alpha = gko::initialize<Mtx>({3.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(alpha), lend(x), lend(beta), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({713.0, 355.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongestToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -1  1  2 ] * [  9  4 ] * [ 1 0 ]^2
              [ 1 ]             [  5 -3  0 ]   [  6 -2 ]   [ 0 1 ]
                                               [ -3  2 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Composition<TypeParam>::create(this->operators.begin(),
                                                   this->operators.end());
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({238.0, 119.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongestLinearCombinationToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -1  1  2 ] * [  9  4 ] * [ 1 0 ]^2
              [ 1 ]             [  5 -3  0 ]   [  6 -2 ]   [ 0 1 ]
                                               [ -3  2 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Composition<TypeParam>::create(this->operators.begin(),
                                                   this->operators.end());
    auto alpha = gko::initialize<Mtx>({3.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(alpha), lend(x), lend(beta), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({713.0, 355.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongestToVectorMultipleRhs)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -1  1  2 ] * [  9  4 ] * [ 1 0 ]^2
              [ 1 ]             [  5 -3  0 ]   [  6 -2 ]   [ 0 1 ]
                                               [ -3  2 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Composition<TypeParam>::create(this->operators.begin(),
                                                   this->operators.end());
    auto x = clone(this->identity);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({{54.0, 92.0}, {27.0, 46.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongestLinearCombinationToVectorMultipleRhs)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -1  1  2 ] * [  9  4 ] * [ 1 0 ]^2
              [ 1 ]             [  5 -3  0 ]   [  6 -2 ]   [ 0 1 ]
                                               [ -3  2 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmp = gko::Composition<TypeParam>::create(this->operators.begin(),
                                                   this->operators.end());
    auto alpha = gko::initialize<Mtx>({3.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto x = clone(this->identity);
    auto res = clone(x);

    cmp->apply(lend(alpha), lend(x), lend(beta), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({{161.0, 276.0}, {81.0, 137.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesToVectorWithInitialGuess)
{
    /*
        cmp = I * DummyLinOp * I
    */
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto cmp = gko::Composition<TypeParam>::create(
        this->identity,
        DummyLinOp<value_type>::create(this->exec, this->identity->get_size()),
        this->identity);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({1.0, 2.0}), 0);
}


TYPED_TEST(Composition, AppliesToVectorWithInitialGuess2)
{
    /*
        cmp = I * DummyLinOp(2x3) * DummyLinOp(3x2) * I
    */
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto size1 = gko::dim<2>(3, 2);
    auto size2 = gko::dim<2>(2, 3);
    auto cmp = gko::Composition<TypeParam>::create(
        this->identity, DummyLinOp<value_type>::create(this->exec, size2),
        DummyLinOp<value_type>::create(this->exec, size1), this->identity);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({0.0, 0.0}), 0);
}


TYPED_TEST(Composition, AppliesToVectorWithInitialGuess3)
{
    /*
        cmp = I * DummyLinOp
    */
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto cmp = gko::Composition<TypeParam>::create(
        DummyLinOp<value_type>::create(this->exec, this->identity->get_size()),
        this->identity);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({1.0, 2.0}), 0);
}


TYPED_TEST(Composition, AppliesToVectorWithInitialGuess4)
{
    /*
        cmp = I * DummyLinOp(2x3) * DummyLinOp(3x2)
    */
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto size1 = gko::dim<2>(3, 2);
    auto size2 = gko::dim<2>(2, 3);
    auto cmp = gko::Composition<TypeParam>::create(
        this->identity, DummyLinOp<value_type>::create(this->exec, size2),
        DummyLinOp<value_type>::create(this->exec, size1));
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({0.0, 0.0}), 0);
}


TYPED_TEST(Composition, AppliesToVectorWithInitialGuess5)
{
    /*
        cmp = DummyLinOp(2x3) * DummyLinOp(3x2) * I
    */
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto size1 = gko::dim<2>(3, 2);
    auto size2 = gko::dim<2>(2, 3);
    auto cmp = gko::Composition<TypeParam>::create(
        DummyLinOp<value_type>::create(this->exec, size2),
        DummyLinOp<value_type>::create(this->exec, size1), this->identity);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(lend(x), lend(res));

    GKO_ASSERT_MTX_NEAR(res, l({1.0, 2.0}), 0);
}


}  // namespace
