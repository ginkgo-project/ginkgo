// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"


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


template <typename ValueType>
class GenericDenseCache : public ::testing::Test {
protected:
    using value_type = ValueType;

    GenericDenseCache() : ref(gko::ReferenceExecutor::create()), size(4, 7) {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    gko::dim<2> size;
    gko::detail::GenericDenseCache cache;
};

TYPED_TEST_SUITE(GenericDenseCache, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(GenericDenseCache, GenericCanInitWithSize)
{
    using value_type = typename TestFixture::value_type;

    auto buffer = this->cache.template get<value_type>(this->ref, this->size);

    ASSERT_NE(buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(buffer->get_size(), this->size);
    ASSERT_EQ(buffer->get_executor(), this->ref);
}


TYPED_TEST(GenericDenseCache, SecondInitWithSameSizeIsNoOp)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = this->cache.template get<value_type>(this->ref, this->size);
    auto array_ptr = this->cache.workspace.get_const_data();
    auto array_size = this->cache.workspace.get_size();

    auto second_buffer =
        this->cache.template get<value_type>(this->ref, this->size);

    ASSERT_NE(second_buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_size(), this->size);
    ASSERT_EQ(second_buffer->get_executor(), this->ref);
    ASSERT_EQ(array_ptr, this->cache.workspace.get_const_data());
    ASSERT_EQ(array_size, this->cache.workspace.get_size());
}


TYPED_TEST(GenericDenseCache, SecondInitWithTheSmallEqSizeIsNoOp)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> second_size{7, 4};
    auto buffer = this->cache.template get<value_type>(this->ref, this->size);
    auto array_ptr = this->cache.workspace.get_const_data();
    auto array_size = this->cache.workspace.get_size();

    auto second_buffer =
        this->cache.template get<value_type>(this->ref, second_size);

    ASSERT_NE(second_buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_size(), second_size);
    ASSERT_EQ(second_buffer->get_executor(), this->ref);
    ASSERT_EQ(array_ptr, this->cache.workspace.get_const_data());
    ASSERT_EQ(array_size, this->cache.workspace.get_size());
}


TYPED_TEST(GenericDenseCache, SecondInitWithTheLargerSizeRecreate)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> second_size{7, 5};
    auto buffer = this->cache.template get<value_type>(this->ref, this->size);
    auto array_ptr = this->cache.workspace.get_const_data();
    auto array_size = this->cache.workspace.get_size();

    auto second_buffer =
        this->cache.template get<value_type>(this->ref, second_size);

    ASSERT_NE(second_buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_size(), second_size);
    ASSERT_EQ(second_buffer->get_executor(), this->ref);
    ASSERT_NE(array_ptr, this->cache.workspace.get_const_data());
    ASSERT_GT(this->cache.workspace.get_size(), array_size);
}


TYPED_TEST(GenericDenseCache, GenericCanInitWithSizeAndType)
{
    using value_type = typename TestFixture::value_type;
    using another_type = gko::next_precision<value_type>;
    auto buffer = this->cache.template get<value_type>(this->ref, this->size);
    auto array_ptr = this->cache.workspace.get_const_data();
    auto array_size = this->cache.workspace.get_size();

    auto second_buffer =
        this->cache.template get<another_type>(this->ref, this->size);

    ASSERT_NE(second_buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_size(), this->size);
    ASSERT_EQ(second_buffer->get_executor(), this->ref);
    if (sizeof(another_type) > sizeof(value_type)) {
        // the requring workspace will be bigger if the type is larger.
        ASSERT_NE(array_ptr, this->cache.workspace.get_const_data());
        ASSERT_GT(this->cache.workspace.get_size(), array_size);
    } else {
        ASSERT_EQ(array_ptr, this->cache.workspace.get_const_data());
        ASSERT_EQ(array_size, this->cache.workspace.get_size());
    }
}


TYPED_TEST(GenericDenseCache, GenericCanInitWithDifferentExecutor)
{
    using value_type = typename TestFixture::value_type;
    auto another_ref = gko::ReferenceExecutor::create();
    auto buffer = this->cache.template get<value_type>(this->ref, this->size);
    auto array_ptr = this->cache.workspace.get_const_data();
    auto array_size = this->cache.workspace.get_size();

    auto second_buffer =
        this->cache.template get<value_type>(another_ref, this->size);
    ASSERT_NE(second_buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_size(), this->size);
    ASSERT_EQ(second_buffer->get_executor(), another_ref);
    // Different executor always regenerate different workspace
    ASSERT_NE(array_ptr, this->cache.workspace.get_const_data());
}


TYPED_TEST(GenericDenseCache, WorkspaceIsNotCopied)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = this->cache.template get<value_type>(this->ref, this->size);
    gko::detail::GenericDenseCache cache(this->cache);

    ASSERT_EQ(cache.workspace.get_size(), 0);
    ASSERT_EQ(cache.workspace.get_executor(), nullptr);
}


TYPED_TEST(GenericDenseCache, WorkspaceIsNotMoved)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = this->cache.template get<value_type>(this->ref, this->size);
    gko::detail::GenericDenseCache cache(std::move(this->cache));

    ASSERT_EQ(cache.workspace.get_size(), 0);
    ASSERT_EQ(cache.workspace.get_executor(), nullptr);
}


TYPED_TEST(GenericDenseCache, WorkspaceIsNotCopyAssigned)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = this->cache.template get<value_type>(this->ref, this->size);
    gko::detail::GenericDenseCache cache;
    cache = this->cache;

    ASSERT_EQ(cache.workspace.get_size(), 0);
    ASSERT_EQ(cache.workspace.get_executor(), nullptr);
}


TYPED_TEST(GenericDenseCache, WorkspaceIsNotMoveAssigned)
{
    using value_type = typename TestFixture::value_type;
    auto buffer = this->cache.template get<value_type>(this->ref, this->size);
    gko::detail::GenericDenseCache cache;
    cache = std::move(this->cache);

    ASSERT_EQ(cache.workspace.get_size(), 0);
    ASSERT_EQ(cache.workspace.get_executor(), nullptr);
}
