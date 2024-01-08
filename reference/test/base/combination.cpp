// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/combination.hpp>


#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Combination : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<T>;

    Combination()
        : exec{gko::ReferenceExecutor::create()},
          coefficients{gko::initialize<Mtx>({1}, exec),
                       gko::initialize<Mtx>({2}, exec)},
          operators{
              gko::initialize<Mtx>({I<T>({2.0, 3.0}), I<T>({1.0, 4.0})}, exec),
              gko::initialize<Mtx>({I<T>({3.0, 2.0}), I<T>({2.0, 0.0})}, exec)}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::vector<std::shared_ptr<gko::LinOp>> coefficients;
    std::vector<std::shared_ptr<gko::LinOp>> operators;
};

TYPED_TEST_SUITE(Combination, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Combination, CopiesOnSameExecutor)
{
    using Mtx = typename TestFixture::Mtx;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto out = cmb->create_default();

    cmb->convert_to(out);

    ASSERT_EQ(out->get_size(), cmb->get_size());
    ASSERT_EQ(out->get_executor(), cmb->get_executor());
    ASSERT_EQ(out->get_operators().size(), cmb->get_operators().size());
    ASSERT_EQ(out->get_operators().size(), 2);
    ASSERT_EQ(out->get_coefficients().size(), cmb->get_coefficients().size());
    ASSERT_EQ(out->get_coefficients().size(), 2);
    ASSERT_EQ(out->get_operators()[0], cmb->get_operators()[0]);
    ASSERT_EQ(out->get_operators()[1], cmb->get_operators()[1]);
    ASSERT_EQ(out->get_coefficients()[0], cmb->get_coefficients()[0]);
    ASSERT_EQ(out->get_coefficients()[1], cmb->get_coefficients()[1]);
}


TYPED_TEST(Combination, MovesOnSameExecutor)
{
    using Mtx = typename TestFixture::Mtx;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto cmb2 = cmb->clone();
    auto out = cmb->create_default();

    cmb->move_to(out);

    ASSERT_EQ(out->get_size(), cmb2->get_size());
    ASSERT_EQ(out->get_executor(), cmb2->get_executor());
    ASSERT_EQ(out->get_operators().size(), cmb2->get_operators().size());
    ASSERT_EQ(out->get_operators().size(), 2);
    ASSERT_EQ(out->get_coefficients().size(), cmb2->get_coefficients().size());
    ASSERT_EQ(out->get_coefficients().size(), 2);
    ASSERT_EQ(out->get_operators()[0], cmb2->get_operators()[0]);
    ASSERT_EQ(out->get_operators()[1], cmb2->get_operators()[1]);
    ASSERT_EQ(out->get_coefficients()[0], cmb2->get_coefficients()[0]);
    ASSERT_EQ(out->get_coefficients()[1], cmb2->get_coefficients()[1]);
    // empty size, empty operators, same executor
    ASSERT_EQ(cmb->get_size(), gko::dim<2>{});
    ASSERT_EQ(cmb->get_executor(), cmb2->get_executor());
    ASSERT_EQ(cmb->get_operators().size(), 0);
    ASSERT_EQ(cmb->get_coefficients().size(), 0);
}


TYPED_TEST(Combination, AppliesToVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
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
    using Mtx = gko::matrix::Dense<value_type>;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
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
    using Mtx = gko::to_complex<typename TestFixture::Mtx>;
    using T = typename Mtx::value_type;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto x = gko::initialize<Mtx>({T{1.0, -2.0}, T{2.0, -4.0}}, this->exec);
    auto res = clone(x);

    cmb->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res, l({T{22.0, -44.0}, T{13.0, -26.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Combination, AppliesToMixedComplexVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    using value_type = gko::to_complex<gko::next_precision<TypeParam>>;
    using Mtx = gko::matrix::Dense<value_type>;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto x = gko::initialize<Mtx>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}}, this->exec);
    auto res = clone(x);

    cmb->apply(x, res);

    GKO_ASSERT_MTX_NEAR(res,
                        l({value_type{22.0, -44.0}, value_type{13.0, -26.0}}),
                        (r_mixed<value_type, TypeParam>()));
}


TYPED_TEST(Combination, AppliesLinearCombinationToVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    using Mtx = typename TestFixture::Mtx;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto alpha = gko::initialize<Mtx>({3.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
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
    using Mtx = gko::matrix::Dense<value_type>;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto alpha = gko::initialize<Mtx>({3.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto x = gko::initialize<Mtx>({1.0, 2.0}, this->exec);
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
    using Dense = typename TestFixture::Mtx;
    using DenseComplex = gko::to_complex<Dense>;
    using T = typename DenseComplex::value_type;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto alpha = gko::initialize<Dense>({3.0}, this->exec);
    auto beta = gko::initialize<Dense>({-1.0}, this->exec);
    auto x =
        gko::initialize<DenseComplex>({T{1.0, -2.0}, T{2.0, -4.0}}, this->exec);
    auto res = clone(x);

    cmb->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({T{65.0, -130.0}, T{37.0, -74.0}}),
                        r<TypeParam>::value);
}


TYPED_TEST(Combination, AppliesLinearCombinationToMixedComplexVector)
{
    /*
        cmb = [ 8 7 ]
              [ 5 4 ]
    */
    using MixedDense = gko::matrix::Dense<gko::next_precision<TypeParam>>;
    using MixedDenseComplex = gko::to_complex<MixedDense>;
    using value_type = typename MixedDenseComplex::value_type;
    auto cmb = gko::Combination<TypeParam>::create(
        this->coefficients[0], this->operators[0], this->coefficients[1],
        this->operators[1]);
    auto alpha = gko::initialize<MixedDense>({3.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto x = gko::initialize<MixedDenseComplex>(
        {value_type{1.0, -2.0}, value_type{2.0, -4.0}}, this->exec);
    auto res = clone(x);

    cmb->apply(alpha, x, beta, res);

    GKO_ASSERT_MTX_NEAR(res,
                        l({value_type{65.0, -130.0}, value_type{37.0, -74.0}}),
                        (r_mixed<value_type, TypeParam>()));
}


}  // namespace
