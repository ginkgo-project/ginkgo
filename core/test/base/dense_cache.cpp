// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/dense_cache.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueType>
struct DenseCacheConfig {
    using cache_type = gko::detail::DenseCache<ValueType>;
    using stored_type = gko::matrix::Dense<ValueType>;

    stored_type* get() { return cache.get(); }

    void init_from(const stored_type* template_vec)
    {
        cache.init_from(template_vec);
    }

    void init(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
    {
        cache.init(std::move(exec), size);
    }

    cache_type cache;
};


struct AnyDenseCacheConfig {
    using cache_type = gko::detail::AnyDenseCache;
    using stored_type = gko::matrix::Dense<double>;

    stored_type* get() { return cache.get<double>(); }

    void init_from(const stored_type* template_vec)
    {
        cache.init_from(template_vec);
    }

    void init(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
    {
        cache.init<double>(std::move(exec), size);
    }

    cache_type cache;
};


template <typename T>
class Cache : public ::testing::Test {
public:
    using Config = T;
    std::shared_ptr<gko::ReferenceExecutor> ref =
        gko::ReferenceExecutor::create();

    void gen_cache(gko::dim<2> size) { config.init(ref, size); }

    Config config;
};

using CacheTypes =
    ::testing::Types<DenseCacheConfig<double>, DenseCacheConfig<float>,
                     DenseCacheConfig<std::complex<double>>,
                     DenseCacheConfig<std::complex<float>>,
                     AnyDenseCacheConfig>;
TYPED_TEST_SUITE(Cache, CacheTypes, TypenameNameGenerator);


TYPED_TEST(Cache, CanDefaultConstruct)
{
    ASSERT_EQ(this->config.get(), nullptr);
}


TYPED_TEST(Cache, CanInitWithSize)
{
    gko::dim<2> size{4, 7};

    this->config.init(this->ref, size);

    ASSERT_NE(this->config.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->config.get()->get_size(), size);
    ASSERT_EQ(this->config.get()->get_executor(), this->ref);
}


TYPED_TEST(Cache, SecondInitWithSameSizeIsNoOp)
{
    gko::dim<2> size{4, 7};
    this->config.init(this->ref, size);
    auto first_ptr = this->config.get();

    this->config.init(this->ref, size);

    ASSERT_NE(this->config.get(), nullptr);
    ASSERT_EQ(first_ptr, this->config.get());
}


TYPED_TEST(Cache, SecondInitWithDifferentSizeInitializes)
{
    gko::dim<2> size{4, 7};
    gko::dim<2> second_size{7, 4};
    this->config.init(this->ref, size);
    auto first_ptr = this->config.get();

    this->config.init(this->ref, second_size);

    ASSERT_NE(this->config.get(), nullptr);
    ASSERT_NE(first_ptr, this->config.get());
}


TYPED_TEST(Cache, CanInitFromDense)
{
    using Config = typename TestFixture::Config;
    gko::dim<2> size{5, 2};
    auto dense = Config::stored_type::create(this->ref, size);

    this->config.init_from(dense.get());

    ASSERT_NE(this->config.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->config.get()->get_size(), size);
    ASSERT_EQ(this->config.get()->get_executor(), dense->get_executor());
}


TYPED_TEST(Cache, SecondInitFromSameDenseIsNoOp)
{
    using Config = typename TestFixture::Config;
    gko::dim<2> size{4, 7};
    auto dense = Config::stored_type::create(this->ref, size);
    this->config.init_from(dense.get());
    auto first_ptr = this->config.get();

    this->config.init_from(dense.get());

    ASSERT_NE(this->config.get(), nullptr);
    ASSERT_EQ(first_ptr, this->config.get());
}


TYPED_TEST(Cache, SecondInitFromDifferentDenseWithSameSizeIsNoOp)
{
    using Config = typename TestFixture::Config;
    gko::dim<2> size{4, 7};
    auto first_dense = Config::stored_type::create(this->ref, size);
    auto second_dense = Config::stored_type::create(this->ref, size);
    this->config.init_from(first_dense.get());
    auto first_ptr = this->config.get();

    this->config.init_from(second_dense.get());

    ASSERT_NE(this->config.get(), nullptr);
    ASSERT_EQ(first_ptr, this->config.get());
}


TYPED_TEST(Cache, SecondInitFromDifferentDenseWithDifferentSizeInitializes)
{
    using Config = typename TestFixture::Config;
    gko::dim<2> size{4, 7};
    gko::dim<2> second_size{7, 4};
    auto first_dense = Config::stored_type::create(this->ref, size);
    auto second_dense = Config::stored_type::create(this->ref, second_size);
    this->config.init_from(first_dense.get());
    auto first_ptr = this->config.get();

    this->config.init_from(second_dense.get());

    ASSERT_NE(this->config.get(), nullptr);
    ASSERT_NE(first_ptr, this->config.get());
}


TYPED_TEST(Cache, VectorIsNotCopied)
{
    using Config = typename TestFixture::Config;
    this->gen_cache({1, 1});
    Config config{this->config.cache};

    ASSERT_EQ(config.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->config.get()->get_size(),
                                gko::dim<2>(1, 1));
}


TYPED_TEST(Cache, VectorIsNotMoved)
{
    using Config = typename TestFixture::Config;
    this->gen_cache({1, 1});
    Config config{std::move(this->config.cache)};

    ASSERT_EQ(config.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->config.get()->get_size(),
                                gko::dim<2>(1, 1));
}


TYPED_TEST(Cache, VectorIsNotCopyAssigned)
{
    using Config = typename TestFixture::Config;
    this->gen_cache({1, 1});
    Config config;
    config.cache = this->config.cache;

    ASSERT_EQ(config.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->config.get()->get_size(),
                                gko::dim<2>(1, 1));
}


TYPED_TEST(Cache, VectorIsNotMoveAssigned)
{
    using Config = typename TestFixture::Config;
    this->gen_cache({1, 1});
    Config config;
    config.cache = std::move(this->config.cache);

    ASSERT_EQ(config.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->config.get()->get_size(),
                                gko::dim<2>(1, 1));
}


class AnyDenseCache : public ::testing::Test {
public:
    std::shared_ptr<gko::Executor> ref = gko::ReferenceExecutor::create();
    gko::detail::AnyDenseCache cache;
};


TEST_F(AnyDenseCache, GetWithNonMatchingValueTypeReturnsNullptr)
{
    cache.init<double>(ref, gko::dim<2>{1, 1});

    ASSERT_EQ(cache.get<float>(), nullptr);
}


TEST_F(AnyDenseCache, InitWithNonMatchingValueTypeInitializes)
{
    gko::dim<2> size{4, 7};
    cache.init<double>(ref, size);

    cache.init<float>(this->ref, size);

    ASSERT_NE(cache.get<float>(), nullptr);
    ASSERT_EQ(cache.get<double>(), nullptr);
}


TEST_F(AnyDenseCache, InitFromWithNonMatchingValueTypeInitializes)
{
    gko::dim<2> size{4, 7};
    auto first_dense = gko::matrix::Dense<double>::create(this->ref, size);
    auto second_dense = gko::matrix::Dense<float>::create(this->ref, size);
    cache.init_from(first_dense.get());

    cache.init_from(second_dense.get());

    ASSERT_NE(cache.get<float>(), nullptr);
    ASSERT_EQ(cache.get<double>(), nullptr);
}


}  // namespace
