// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/identity.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Identity : public ::testing::Test {
protected:
    using value_type = T;
    using Id = gko::matrix::Identity<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<gko::next_precision<value_type>>;
    using ComplexVec = gko::to_complex<Vec>;
    using MixedComplexVec = gko::to_complex<MixedVec>;

    Identity() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};


TYPED_TEST_SUITE(Identity, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Identity, AppliesToVector)
{
    using Id = typename TestFixture::Id;
    using Vec = typename TestFixture::Vec;
    auto identity = Id::create(this->exec, 3);
    auto x = gko::initialize<Vec>({3.0, -1.0, 2.0}, this->exec);
    auto b = gko::initialize<Vec>({2.0, 1.0, 5.0}, this->exec);

    identity->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({2.0, 1.0, 5.0}), 0.0);
}


TYPED_TEST(Identity, AppliesToMultipleVectors)
{
    using Id = typename TestFixture::Id;
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto identity = Id::create(this->exec, 3);
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2}, 3);
    auto b = gko::initialize<Vec>(
        3, {I<T>{2.0, 3.0}, I<T>{1.0, 2.0}, I<T>{5.0, -1.0}}, this->exec);

    identity->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({{2.0, 3.0}, {1.0, 2.0}, {5.0, -1.0}}), 0.0);
}


TYPED_TEST(Identity, AppliesToMixedVector)
{
    using Id = typename TestFixture::Id;
    using MixedVec = typename TestFixture::MixedVec;
    auto identity = Id::create(this->exec, 3);
    auto x = gko::initialize<MixedVec>({3.0, -1.0, 2.0}, this->exec);
    auto b = gko::initialize<MixedVec>({2.0, 1.0, 5.0}, this->exec);

    identity->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({2.0, 1.0, 5.0}), 0.0);
}


TYPED_TEST(Identity, AppliesLinearCombinationToVector)
{
    using Id = typename TestFixture::Id;
    using Vec = typename TestFixture::Vec;
    auto identity = Id::create(this->exec, 3);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({1.0}, this->exec);
    auto x = gko::initialize<Vec>({3.0, -1.0, 2.0}, this->exec);
    auto b = gko::initialize<Vec>({2.0, 1.0, 5.0}, this->exec);

    identity->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({7.0, 1.0, 12.0}), 0.0);
}


TYPED_TEST(Identity, AppliesLinearCombinationToMixedVector)
{
    using Id = typename TestFixture::Id;
    using MixedVec = typename TestFixture::MixedVec;
    auto identity = Id::create(this->exec, 3);
    auto alpha = gko::initialize<MixedVec>({2.0}, this->exec);
    auto beta = gko::initialize<MixedVec>({1.0}, this->exec);
    auto x = gko::initialize<MixedVec>({3.0, -1.0, 2.0}, this->exec);
    auto b = gko::initialize<MixedVec>({2.0, 1.0, 5.0}, this->exec);

    identity->apply(alpha, b, beta, x);

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
        3, {I<T>{3.0, 0.5}, I<T>{-1.0, 2.5}, I<T>{2.0, 3.5}}, this->exec);
    auto b = gko::initialize<Vec>(
        3, {I<T>{2.0, 3.0}, I<T>{1.0, 2.0}, I<T>{5.0, -1.0}}, this->exec);

    identity->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({{7.0, 6.5}, {1.0, 6.5}, {12.0, 1.5}}), 0.0);
}


TYPED_TEST(Identity, AppliesToComplex)
{
    using Id = typename TestFixture::Id;
    using ComplexVec = typename TestFixture::ComplexVec;
    auto identity = Id::create(this->exec, 3);
    auto x = gko::initialize<ComplexVec>({3.0, -1.0, 2.0}, this->exec);
    auto b = gko::initialize<ComplexVec>({2.0, 1.0, 5.0}, this->exec);

    identity->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({2.0, 1.0, 5.0}), 0.0);
}


TYPED_TEST(Identity, AppliesToMixedComplex)
{
    using Id = typename TestFixture::Id;
    using MixedComplexVec = typename TestFixture::MixedComplexVec;
    auto identity = Id::create(this->exec, 3);
    auto x = gko::initialize<MixedComplexVec>({3.0, -1.0, 2.0}, this->exec);
    auto b = gko::initialize<MixedComplexVec>({2.0, 1.0, 5.0}, this->exec);

    identity->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, l({2.0, 1.0, 5.0}), 0.0);
}


TYPED_TEST(Identity, AppliesLinearCombinationToComplex)
{
    using Id = typename TestFixture::Id;
    using Vec = typename TestFixture::Vec;
    using ComplexVec = typename TestFixture::ComplexVec;
    auto identity = Id::create(this->exec, 3);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({1.0}, this->exec);
    auto x = gko::initialize<ComplexVec>({3.0, -1.0, 2.0}, this->exec);
    auto b = gko::initialize<ComplexVec>({2.0, 1.0, 5.0}, this->exec);

    identity->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({7.0, 1.0, 12.0}), 0.0);
}


TYPED_TEST(Identity, AppliesLinearCombinationToMixedComplex)
{
    using Id = typename TestFixture::Id;
    using MixedVec = typename TestFixture::MixedVec;
    using MixedComplexVec = typename TestFixture::MixedComplexVec;
    auto identity = Id::create(this->exec, 3);
    auto alpha = gko::initialize<MixedVec>({2.0}, this->exec);
    auto beta = gko::initialize<MixedVec>({1.0}, this->exec);
    auto x = gko::initialize<MixedComplexVec>({3.0, -1.0, 2.0}, this->exec);
    auto b = gko::initialize<MixedComplexVec>({2.0, 1.0, 5.0}, this->exec);

    identity->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({7.0, 1.0, 12.0}), 0.0);
}


}  // namespace
