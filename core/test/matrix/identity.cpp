// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/identity.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Identity : public ::testing::Test {
protected:
    using value_type = T;
    using Id = gko::matrix::Identity<T>;
    using Vec = gko::matrix::Dense<T>;

    Identity() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(Identity, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Identity, CanBeEmpty)
{
    using Id = typename TestFixture::Id;
    auto empty = Id::create(this->exec);
    ASSERT_EQ(empty->get_size(), gko::dim<2>(0, 0));
}


TYPED_TEST(Identity, CanBeConstructedWithSize)
{
    using Id = typename TestFixture::Id;
    auto identity = Id::create(this->exec, 5);

    ASSERT_EQ(identity->get_size(), gko::dim<2>(5, 5));
}


TYPED_TEST(Identity, CanBeConstructedWithSquareSize)
{
    using Id = typename TestFixture::Id;
    auto identity = Id::create(this->exec, gko::dim<2>(5, 5));

    ASSERT_EQ(identity->get_size(), gko::dim<2>(5, 5));
}


TYPED_TEST(Identity, FailsConstructionWithRectangularSize)
{
    using Id = typename TestFixture::Id;

    ASSERT_THROW(Id::create(this->exec, gko::dim<2>(5, 4)),
                 gko::DimensionMismatch);
}


template <typename T>
class IdentityFactory : public ::testing::Test {
protected:
    using value_type = T;
};

TYPED_TEST_SUITE(IdentityFactory, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(IdentityFactory, CanGenerateIdentityMatrix)
{
    auto exec = gko::ReferenceExecutor::create();
    auto id_factory = gko::matrix::IdentityFactory<TypeParam>::create(exec);
    auto mtx = gko::matrix::Dense<TypeParam>::create(exec, gko::dim<2>{5, 5});

    auto id = id_factory->generate(std::move(mtx));

    ASSERT_EQ(id->get_size(), gko::dim<2>(5, 5));
}


TYPED_TEST(IdentityFactory, FailsToGenerateRectangularIdentityMatrix)
{
    auto exec = gko::ReferenceExecutor::create();
    auto id_factory = gko::matrix::IdentityFactory<TypeParam>::create(exec);
    auto mtx = gko::matrix::Dense<TypeParam>::create(exec, gko::dim<2>{5, 4});

    ASSERT_THROW(id_factory->generate(std::move(mtx)), gko::DimensionMismatch);
}


}  // namespace
