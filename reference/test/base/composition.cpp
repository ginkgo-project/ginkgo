// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename ValueType>
class DummyLinOp : public gko::LinOp,
                   public gko::EnableCreateMethod<DummyLinOp<ValueType>> {
    friend class gko::EnableCreateMethod<DummyLinOp>;

public:
    using value_type = ValueType;

    bool apply_uses_initial_guess() const override { return true; }

protected:
    void apply_impl(const gko::AbstractMultiVector* b,
                    gko::AbstractMultiVector* x) const override
    {}

    void apply_impl(const gko::AbstractMultiVector* alpha,
                    const gko::AbstractMultiVector* b,
                    const gko::AbstractMultiVector* beta,
                    gko::AbstractMultiVector* x) const override
    {}

    explicit DummyLinOp(std::shared_ptr<const gko::Executor> exec)
        : gko::LinOp(exec)
    {}

    explicit DummyLinOp(std::shared_ptr<const gko::Executor> exec,
                        gko::dim<2> size)
        : gko::LinOp(exec, size)
    {}
};


template <typename T>
class Composition : public ::testing::Test {
protected:
    using Vec = gko::matrix::MultiVector<T>;
    using Mtx = gko::matrix::Dense<T>;
    using value_type = T;

    Composition() : exec{gko::ReferenceExecutor::create()}
    {
        operators = {
            gko::initialize<Mtx>(I<T>({2.0, 1.0}), exec),
            gko::initialize<Mtx>({I<T>({3.0, 2.0})}, exec),
            gko::initialize<Mtx>(
                {I<T>({-1.0, 1.0, 2.0}), I<T>({5.0, -3.0, 0.0})}, exec),
            gko::initialize<Mtx>(
                {I<T>({9.0, 4.0}), I<T>({6.0, -2.0}), I<T>({-3.0, 2.0})}, exec),
            gko::initialize<Mtx>({I<T>({1.0, 0.0}), I<T>({0.0, 1.0})}, exec),
            gko::initialize<Mtx>({I<T>({1.0, 0.0}), I<T>({0.0, 1.0})}, exec)};
        identity =
            gko::initialize<Mtx>({I<T>({1.0, 0.0}), I<T>({0.0, 1.0})}, exec);
        product = gko::initialize<Mtx>({I<T>({-9.0, -2.0}), I<T>({27.0, 26.0})},
                                       exec);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::vector<std::shared_ptr<Vec>> coefficients;
    std::vector<std::shared_ptr<gko::LinOp>> operators;
    std::shared_ptr<Mtx> identity;
    std::shared_ptr<Mtx> product;
};

TYPED_TEST_SUITE(Composition, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Composition, AppliesSingleToVector)
{
    /*
        cmp = [ -9 -2 ]
              [ 27 26 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Composition<TypeParam>::create(this->product);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({-13.0, 79.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesSingleToMixedVector)
{
    /*
        cmp = [ -9 -2 ]
              [ 27 26 ]
    */
    using Vec = gko::matrix::MultiVector<gko::next_precision<TypeParam>>;
    using value_type = typename Vec::value_type;
    auto cmp = gko::Composition<TypeParam>::create(this->product);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({-13.0, 79.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Composition, AppliesSingleToComplexVector)
{
    /*
        cmp = [ -9 -2 ]
              [ 27 26 ]
    */
    using value_type = gko::to_complex<TypeParam>;
    using Vec = gko::matrix::MultiVector<value_type>;
    auto cmp = gko::Composition<TypeParam>::create(this->product);
    auto x = gko::initialize<Vec>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res,
                        l({value_type{-13.0, 26.0}, value_type{79.0, -158.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesSingleLinearCombinationToVector)
{
    /*
        cmp = [ -9 -2 ]
              [ 27 26 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Composition<TypeParam>::create(this->product);
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({-40.0, 235.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesSingleLinearCombinationToMixedVector)
{
    /*
        cmp = [ -9 -2 ]
              [ 27 26 ]
    */
    using value_type = gko::next_precision<TypeParam>;
    using Vec = gko::matrix::MultiVector<value_type>;
    auto cmp = gko::Composition<TypeParam>::create(this->product);
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({-40.0, 235.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Composition, AppliesSingleLinearCombinationToComplexVector)
{
    /*
        cmp = [ -9 -2 ]
              [ 27 26 ]
    */
    using MultiVector = typename TestFixture::Vec;
    using MultiVectorComplex = gko::to_complex<MultiVector>;
    using value_type = typename MultiVectorComplex::value_type;
    auto cmp = gko::Composition<TypeParam>::create(this->product);
    auto alpha = gko::initialize<MultiVector>({3.0}, this->exec);
    auto beta = gko::initialize<MultiVector>({-1.0}, this->exec);
    auto x = gko::initialize<MultiVectorComplex>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}}, this->exec);
    auto res = clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res,
                        l({value_type{-40.0, 80.0}, value_type{235.0, -470.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ]
              [ 1 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Composition<TypeParam>::create(this->operators[0],
                                                   this->operators[1]);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({14.0, 7.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLinearCombinationToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ]
              [ 1 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Composition<TypeParam>::create(this->operators[0],
                                                   this->operators[1]);
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({41.0, 19.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongerToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -9  -2 ]
              [ 1 ]             [ 27  26 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Composition<TypeParam>::create(
        this->operators[0], this->operators[1], this->product);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({238.0, 119.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongerLinearCombinationToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -9  -2 ]
              [ 1 ]             [ 27  26 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Composition<TypeParam>::create(
        this->operators[0], this->operators[1], this->product);
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({713.0, 355.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongestToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -1  1  2 ] * [  9  4 ] * [ 1 0 ]^2
              [ 1 ]             [  5 -3  0 ]   [  6 -2 ]   [ 0 1 ]
                                               [ -3  2 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Composition<TypeParam>::create(this->operators.begin(),
                                                   this->operators.end());
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({238.0, 119.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongestLinearCombinationToVector)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -1  1  2 ] * [  9  4 ] * [ 1 0 ]^2
              [ 1 ]             [  5 -3  0 ]   [  6 -2 ]   [ 0 1 ]
                                               [ -3  2 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Composition<TypeParam>::create(this->operators.begin(),
                                                   this->operators.end());
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({713.0, 355.0}), r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesLongestToVectorMultipleRhs)
{
    /*
        cmp = [ 2 ] * [ 3 2 ] * [ -1  1  2 ] * [  9  4 ] * [ 1 0 ]^2
              [ 1 ]             [  5 -3  0 ]   [  6 -2 ]   [ 0 1 ]
                                               [ -3  2 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Composition<TypeParam>::create(this->operators.begin(),
                                                   this->operators.end());
    auto x = clone(this->identity->as_multivector_view());
    auto res = clone(x);

    cmp->apply(x, res);

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
    using Vec = typename TestFixture::Vec;
    auto cmp = gko::Composition<TypeParam>::create(this->operators.begin(),
                                                   this->operators.end());
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = clone(this->identity->as_multivector_view());
    auto res = clone(x);

    cmp->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({{161.0, 276.0}, {81.0, 137.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Composition, AppliesToVectorWithInitialGuess)
{
    /*
        cmp = I * DummyLinOp * I
    */
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto cmp = gko::Composition<TypeParam>::create(
        this->identity,
        DummyLinOp<value_type>::create(this->exec, this->identity->get_size()),
        this->identity);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({1.0, 2.0}), 0);
}


TYPED_TEST(Composition, AppliesToVectorWithInitialGuess2)
{
    /*
        cmp = I * DummyLinOp(2x3) * DummyLinOp(3x2) * I
    */
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto size1 = gko::dim<2>(3, 2);
    auto size2 = gko::dim<2>(2, 3);
    auto cmp = gko::Composition<TypeParam>::create(
        this->identity, DummyLinOp<value_type>::create(this->exec, size2),
        DummyLinOp<value_type>::create(this->exec, size1), this->identity);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({0.0, 0.0}), 0);
}


TYPED_TEST(Composition, AppliesToVectorWithInitialGuess3)
{
    /*
        cmp = I * DummyLinOp
    */
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto cmp = gko::Composition<TypeParam>::create(
        DummyLinOp<value_type>::create(this->exec, this->identity->get_size()),
        this->identity);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({1.0, 2.0}), 0);
}


TYPED_TEST(Composition, AppliesToVectorWithInitialGuess4)
{
    /*
        cmp = I * DummyLinOp(2x3) * DummyLinOp(3x2)
    */
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto size1 = gko::dim<2>(3, 2);
    auto size2 = gko::dim<2>(2, 3);
    auto cmp = gko::Composition<TypeParam>::create(
        this->identity, DummyLinOp<value_type>::create(this->exec, size2),
        DummyLinOp<value_type>::create(this->exec, size1));
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({0.0, 0.0}), 0);
}


TYPED_TEST(Composition, AppliesToVectorWithInitialGuess5)
{
    /*
        cmp = DummyLinOp(2x3) * DummyLinOp(3x2) * I
    */
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto size1 = gko::dim<2>(3, 2);
    auto size2 = gko::dim<2>(2, 3);
    auto cmp = gko::Composition<TypeParam>::create(
        DummyLinOp<value_type>::create(this->exec, size2),
        DummyLinOp<value_type>::create(this->exec, size1), this->identity);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmp->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({1.0, 2.0}), 0);
}


}  // namespace
