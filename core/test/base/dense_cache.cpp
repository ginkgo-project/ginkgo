// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/dense_cache.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueType>
class DenseCache : public ::testing::Test {
protected:
    using value_type = ValueType;

    DenseCache() {}

    void SetUp() { ref = gko::ReferenceExecutor::create(); }

    void TearDown() {}

    void gen_cache(gko::dim<2> size) { cache.init(ref, size); }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::detail::DenseCache<value_type> cache;
};


TYPED_TEST_SUITE(DenseCache, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(DenseCache, CanDefaultConstruct)
{
    using value_type = typename TestFixture::value_type;
    gko::detail::DenseCache<value_type> cache;

    ASSERT_EQ(cache.get(), nullptr);
}


TYPED_TEST(DenseCache, CanInitWithSize)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{4, 7};

    this->cache.init(this->ref, size);

    ASSERT_NE(this->cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(), size);
    ASSERT_EQ(this->cache->get_executor(), this->ref);
}


TYPED_TEST(DenseCache, SecondInitWithSameSizeIsNoOp)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{4, 7};
    this->cache.init(this->ref, size);
    auto first_ptr = this->cache.get();

    this->cache.init(this->ref, size);

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_EQ(first_ptr, this->cache.get());
}


TYPED_TEST(DenseCache, SecondInitWithDifferentSizeInitializes)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{4, 7};
    gko::dim<2> second_size{7, 4};
    this->cache.init(this->ref, size);
    auto first_ptr = this->cache.get();

    this->cache.init(this->ref, second_size);

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_NE(first_ptr, this->cache.get());
}


TYPED_TEST(DenseCache, CanInitFromDense)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{5, 2};
    auto dense = gko::matrix::Dense<value_type>::create(this->ref, size);

    this->cache.init_from(dense.get());

    ASSERT_NE(this->cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(), size);
    ASSERT_EQ(this->cache->get_executor(), dense->get_executor());
}


TYPED_TEST(DenseCache, SecondInitFromSameDenseIsNoOp)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{4, 7};
    auto dense = gko::matrix::Dense<value_type>::create(this->ref, size);
    this->cache.init_from(dense.get());
    auto first_ptr = this->cache.get();

    this->cache.init_from(dense.get());

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_EQ(first_ptr, this->cache.get());
}


TYPED_TEST(DenseCache, SecondInitFromDifferentDenseWithSameSizeIsNoOp)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{4, 7};
    auto first_dense = gko::matrix::Dense<value_type>::create(this->ref, size);
    auto second_dense = gko::matrix::Dense<value_type>::create(this->ref, size);
    this->cache.init_from(first_dense.get());
    auto first_ptr = this->cache.get();

    this->cache.init_from(second_dense.get());

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_EQ(first_ptr, this->cache.get());
}


TYPED_TEST(DenseCache, SecondInitFromDifferentDenseWithDifferentSizeInitializes)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{4, 7};
    gko::dim<2> second_size{7, 4};
    auto first_dense = gko::matrix::Dense<value_type>::create(this->ref, size);
    auto second_dense =
        gko::matrix::Dense<value_type>::create(this->ref, second_size);
    this->cache.init_from(first_dense.get());
    auto first_ptr = this->cache.get();

    this->cache.init_from(second_dense.get());

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_NE(first_ptr, this->cache.get());
}


TYPED_TEST(DenseCache, VectorIsNotCopied)
{
    using value_type = typename TestFixture::value_type;
    this->gen_cache({1, 1});
    gko::detail::DenseCache<value_type> cache(this->cache);

    ASSERT_EQ(cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(), gko::dim<2>(1, 1));
}


TYPED_TEST(DenseCache, VectorIsNotMoved)
{
    using value_type = typename TestFixture::value_type;
    this->gen_cache({1, 1});
    gko::detail::DenseCache<value_type> cache(std::move(this->cache));

    ASSERT_EQ(cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(), gko::dim<2>(1, 1));
}


TYPED_TEST(DenseCache, VectorIsNotCopyAssigned)
{
    using value_type = typename TestFixture::value_type;
    this->gen_cache({1, 1});
    gko::detail::DenseCache<value_type> cache;
    cache = this->cache;

    ASSERT_EQ(cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(), gko::dim<2>(1, 1));
}


TYPED_TEST(DenseCache, VectorIsNotMoveAssigned)
{
    using value_type = typename TestFixture::value_type;
    this->gen_cache({1, 1});
    gko::detail::DenseCache<value_type> cache;
    cache = std::move(this->cache);

    ASSERT_EQ(cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(), gko::dim<2>(1, 1));
}


}  // namespace
