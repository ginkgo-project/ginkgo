// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cmath>
#include <iostream>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


template <typename ValueType>
class UniformRealDistribution : public testing::Test {
public:
    using dist_type = gko::test::uniform_real_distribution<ValueType>;
    using value_type = ValueType;

    dist_type dist{1, 2};
    std::default_random_engine engine{};
};

TYPED_TEST_SUITE(UniformRealDistribution, gko::test::RealValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(UniformRealDistribution, HasResultTypeSameAsValueType)
{
    using value_type = typename TestFixture::value_type;
    using result_type = typename TestFixture::dist_type::result_type;

    testing::StaticAssertTypeEq<result_type, value_type>();
    testing::StaticAssertTypeEq<decltype(this->dist(this->engine)),
                                value_type>();
}


TYPED_TEST(UniformRealDistribution, CanDefaultCreate)
{
    using dist_type = typename TestFixture::dist_type;
    using value_type = typename TestFixture::value_type;

    auto dist = dist_type();

    ASSERT_EQ(dist.min(), value_type(0));
    ASSERT_EQ(dist.max(), value_type(1));
}


TYPED_TEST(UniformRealDistribution, CanCreateWithOneParameter)
{
    using dist_type = typename TestFixture::dist_type;
    using value_type = typename TestFixture::value_type;

    auto dist = dist_type(value_type(-1));

    ASSERT_EQ(dist.min(), value_type(-1));
    ASSERT_EQ(dist.max(), value_type(1));
}


TYPED_TEST(UniformRealDistribution, CanCreateWithTwoParameter)
{
    using dist_type = typename TestFixture::dist_type;
    using value_type = typename TestFixture::value_type;

    auto dist = dist_type(value_type(-1), value_type(2));

    ASSERT_EQ(dist.min(), value_type(-1));
    ASSERT_EQ(dist.max(), value_type(2));
}


TYPED_TEST(UniformRealDistribution, CanGenerateResult)
{
    auto r1 = this->dist(this->engine);
    auto r2 = this->dist(this->engine);

    ASSERT_NE(r1, r2);
}


template <typename ValueType>
class NormalDistribution : public testing::Test {
public:
    using dist_type = gko::test::normal_distribution<ValueType>;
    using value_type = ValueType;

    dist_type dist{1, 2};
    std::default_random_engine engine{};
};

TYPED_TEST_SUITE(NormalDistribution, gko::test::RealValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(NormalDistribution, HasResultTypeSameAsValueType)
{
    using value_type = typename TestFixture::value_type;
    using result_type = typename TestFixture::dist_type::result_type;

    testing::StaticAssertTypeEq<result_type, value_type>();
    testing::StaticAssertTypeEq<decltype(this->dist(this->engine)),
                                value_type>();
}


TYPED_TEST(NormalDistribution, CanDefaultCreate)
{
    using dist_type = typename TestFixture::dist_type;
    using value_type = typename TestFixture::value_type;

    ASSERT_NO_THROW(dist_type());
}


TYPED_TEST(NormalDistribution, CanCreateWithOneParameter)
{
    using dist_type = typename TestFixture::dist_type;
    using value_type = typename TestFixture::value_type;

    ASSERT_NO_THROW(dist_type(value_type(-1)));
}


TYPED_TEST(NormalDistribution, CanCreateWithTwoParameter)
{
    using dist_type = typename TestFixture::dist_type;
    using value_type = typename TestFixture::value_type;

    ASSERT_NO_THROW(dist_type(value_type(-1), value_type(2)));
}


TYPED_TEST(NormalDistribution, CanGenerateResult)
{
    auto r1 = this->dist(this->engine);
    auto r2 = this->dist(this->engine);

    ASSERT_NE(r1, r2);
}
