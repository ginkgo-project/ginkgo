// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/combination.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class Combination : public ::testing::Test {
protected:
    using Vec = gko::matrix::MultiVector<T>;
    using Mtx = gko::matrix::Dense<T>;

    Combination()
        : exec{gko::ReferenceExecutor::create()},
          coefficients{gko::initialize<Vec>({1}, exec),
                       gko::initialize<Vec>({2}, exec)}
    {
        operators = {
            gko::initialize<Mtx>({I<T>({2.0, 3.0}), I<T>({1.0, 4.0})}, exec),
            gko::initialize<Mtx>({I<T>({3.0, 2.0}), I<T>({2.0, 0.0})}, exec)};
    }

    std::shared_ptr<const gko::Executor> exec;
    std::vector<std::shared_ptr<Vec>> coefficients;
    std::vector<std::shared_ptr<gko::LinOp>> operators;
};

TYPED_TEST_SUITE(Combination, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Combination, AppliesToVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmb->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({22.0, 13.0}), r<TypeParam>::value);
}


TYPED_TEST(Combination, AppliesToMixedVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    using value_type = gko::next_precision<TypeParam>;
    using Vec = gko::matrix::MultiVector<value_type>;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmb->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({22.0, 13.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Combination, AppliesToComplexVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    using Vec = gko::to_complex<typename TestFixture::Vec>;
    using T = typename Vec::value_type;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto x = gko::initialize<Vec>({T{1.0, -2.0}, T{2.0, -4.0}}, this->exec);
    auto res = clone(x);

    cmb->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({T{22.0, -44.0}, T{13.0, -26.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Combination, AppliesLinearCombinationToVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    using Vec = typename TestFixture::Vec;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmb->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({65.0, 37.0}), r<TypeParam>::value);
}


TYPED_TEST(Combination, AppliesLinearCombinationToMixedVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    using value_type = gko::next_precision<TypeParam>;
    using Vec = gko::matrix::MultiVector<value_type>;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1.0, 2.0}, this->exec);
    auto res = clone(x);

    cmb->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({65.0, 37.0}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Combination, AppliesLinearCombinationToComplexVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    using MultiVector = typename TestFixture::Vec;
    using MultiVectorComplex = gko::to_complex<MultiVector>;
    using T = typename MultiVectorComplex::value_type;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto alpha = gko::initialize<MultiVector>({3.0}, this->exec);
    auto beta = gko::initialize<MultiVector>({-1.0}, this->exec);
    auto x = gko::initialize<MultiVectorComplex>({T{1.0, -2.0}, T{2.0, -4.0}},
                                                 this->exec);
    auto res = clone(x);

    cmb->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({T{65.0, -130.0}, T{37.0, -74.0}}),
                        r<TypeParam>::value);
}


}  // namespace
