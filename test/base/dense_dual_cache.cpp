// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/dense_cache.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "test/utils/common_fixture.hpp"


struct copy_logger : gko::log::Logger {
    void on_copy_started(const gko::Executor* exec_from,
                         const gko::Executor* exec_to,
                         const gko::uintptr& loc_from,
                         const gko::uintptr& loc_to,
                         const gko::size_type& num_bytes) const override
    {
        ++count;
    }

    mutable int count = 0;
};


struct alloc_logger : gko::log::Logger {
    void on_allocation_started(const gko::Executor* exec,
                               const gko::size_type& num_bytes) const override
    {
        execs.insert(exec);
        ++count;
    }

    mutable int count = 0;
    mutable std::set<const gko::Executor*> execs;
};


template <typename ValueType>
class DenseDualCache : public CommonTestFixture {
protected:
    using value_type = ValueType;
    using dense_dual_cache_t = gko::detail::DenseDualCache<value_type>;

    void assert_initialized() const
    {
        ASSERT_NE(this->cache.get_const(this->exec), nullptr);
        ASSERT_NE(this->cache.get_const(this->host), nullptr);
        ASSERT_EQ(this->cache.get(this->exec)->get_executor(), this->exec);
        ASSERT_EQ(this->cache.get(this->host)->get_executor(), this->host);
    }

    dense_dual_cache_t cache;
    // this->ref != this->exec->get_master if exec is a CPU executor
    std::shared_ptr<gko::Executor> host = exec->get_master();
};


TYPED_TEST_SUITE(DenseDualCache, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(DenseDualCache, CanDefaultConstruct)
{
    using value_type = typename TestFixture::value_type;
    gko::detail::DenseDualCache<value_type> cache;

    ASSERT_EQ(cache.get_const(this->exec), nullptr);
    ASSERT_EQ(cache.get_const(this->host), nullptr);
}


TYPED_TEST(DenseDualCache, CanInitWithSize)
{
    gko::dim<2> size{4, 7};

    this->cache.init(this->exec, size);

    this->assert_initialized();
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache.get_const(this->exec)->get_size(),
                                size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache.get_const(this->host)->get_size(),
                                size);
}


TYPED_TEST(DenseDualCache, InitializesDeviceOnly)
{
    gko::dim<2> size{4, 7};
    auto log = std::make_shared<alloc_logger>();
    this->exec->add_logger(log);
    this->ref->add_logger(log);

    auto count_before = log->count;
    this->cache.init(this->exec, size);
    auto count_after = log->count;

    ASSERT_EQ(count_before + 1, count_after);
    ASSERT_NE(log->execs.find(this->exec.get()), log->execs.end());
    if (!std::is_same_v<gko::ReferenceExecutor, gko::EXEC_TYPE>) {
        ASSERT_EQ(log->execs.find(this->host.get()), log->execs.end());
    }
}


TYPED_TEST(DenseDualCache, InitializesHostWhenNecessary)
{
    gko::dim<2> size{4, 7};
    auto log = std::make_shared<alloc_logger>();
    this->exec->add_logger(log);
    this->ref->add_logger(log);
    this->cache.init(this->exec, size);

    auto count_before = log->count;
    auto host = this->cache.get(this->host);
    auto count_after = log->count;

    if (!std::is_same_v<gko::ReferenceExecutor, gko::EXEC_TYPE>) {
        ASSERT_EQ(count_before + 1, count_after);
        ASSERT_NE(log->execs.find(this->host.get()), log->execs.end());
    } else {
        ASSERT_EQ(count_before, count_after);
    }
}


TYPED_TEST(DenseDualCache, SecondInitWithSameSizeIsNoOp)
{
    gko::dim<2> size{4, 7};
    this->cache.init(this->exec, size);
    auto first_device_ptr = this->cache.get_const(this->exec);
    auto first_host_ptr = this->cache.get_const(this->host);

    this->cache.init(this->exec, size);

    ASSERT_EQ(first_device_ptr, this->cache.get_const(this->exec));
    ASSERT_EQ(first_host_ptr, this->cache.get_const(this->host));
}


TYPED_TEST(DenseDualCache, SecondInitWithDifferentSizeInitializes)
{
    gko::dim<2> size{4, 7};
    gko::dim<2> second_size{7, 4};
    this->cache.init(this->exec, size);
    auto first_device_ptr = this->cache.get_const(this->exec);
    auto first_host_ptr = this->cache.get_const(this->host);

    this->cache.init(this->exec, second_size);

    ASSERT_NE(this->cache.get_const(this->exec), nullptr);
    ASSERT_NE(this->cache.get_const(this->host), nullptr);
    ASSERT_NE(first_device_ptr, this->cache.get_const(this->exec));
    ASSERT_NE(first_host_ptr, this->cache.get_const(this->host));
}


TYPED_TEST(DenseDualCache, CanInitFromDense)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{5, 2};
    auto dense = gko::matrix::Dense<value_type>::create(this->exec, size);

    this->cache.init_from(dense.get());

    this->assert_initialized();
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache.get_const(this->exec)->get_size(),
                                size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache.get_const(this->host)->get_size(),
                                size);
}


TYPED_TEST(DenseDualCache, SecondInitFromSameDenseIsNoOp)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{4, 7};
    auto dense = gko::matrix::Dense<value_type>::create(this->exec, size);
    this->cache.init_from(dense.get());
    auto first_device_ptr = this->cache.get_const(this->exec);
    auto first_host_ptr = this->cache.get_const(this->host);

    this->cache.init_from(dense.get());

    ASSERT_EQ(first_device_ptr, this->cache.get_const(this->exec));
    ASSERT_EQ(first_host_ptr, this->cache.get_const(this->host));
}


TYPED_TEST(DenseDualCache, SecondInitFromDifferentDenseWithSameSizeIsNoOp)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{4, 7};
    auto first_dense = gko::matrix::Dense<value_type>::create(this->exec, size);
    auto second_dense =
        gko::matrix::Dense<value_type>::create(this->exec, size);
    this->cache.init_from(first_dense.get());
    auto first_device_ptr = this->cache.get_const(this->exec);
    auto first_host_ptr = this->cache.get_const(this->host);

    this->cache.init_from(second_dense.get());

    ASSERT_NE(this->cache.get_const(this->exec), nullptr);
    ASSERT_NE(this->cache.get_const(this->host), nullptr);
    ASSERT_EQ(first_device_ptr, this->cache.get_const(this->exec));
    ASSERT_EQ(first_host_ptr, this->cache.get_const(this->host));
}


TYPED_TEST(DenseDualCache,
           SecondInitFromDifferentDenseWithDifferentSizeInitializes)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> size{4, 7};
    gko::dim<2> second_size{7, 4};
    auto first_dense = gko::matrix::Dense<value_type>::create(this->exec, size);
    auto second_dense =
        gko::matrix::Dense<value_type>::create(this->exec, second_size);
    this->cache.init_from(first_dense.get());
    auto first_device_ptr = this->cache.get_const(this->exec);
    auto first_host_ptr = this->cache.get_const(this->host);

    this->cache.init_from(second_dense.get());

    ASSERT_NE(this->cache.get_const(this->exec), nullptr);
    ASSERT_NE(this->cache.get_const(this->host), nullptr);
    ASSERT_NE(first_device_ptr, this->cache.get_const(this->exec));
    ASSERT_NE(first_host_ptr, this->cache.get_const(this->host));
}


TYPED_TEST(DenseDualCache, VectorIsNotCopied)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->exec, {1, 1});
    gko::detail::DenseDualCache<value_type> cache(this->cache);

    ASSERT_EQ(cache.get_const(this->exec), nullptr);
    ASSERT_EQ(cache.get_const(this->host), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache.get_const(this->exec)->get_size(),
                                gko::dim<2>(1, 1));
}


TYPED_TEST(DenseDualCache, VectorIsNotMoved)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->exec, {1, 1});
    gko::detail::DenseDualCache<value_type> cache(std::move(this->cache));

    ASSERT_EQ(cache.get_const(this->exec), nullptr);
    ASSERT_EQ(cache.get_const(this->host), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache.get_const(this->exec)->get_size(),
                                gko::dim<2>(1, 1));
}


TYPED_TEST(DenseDualCache, VectorIsNotCopyAssigned)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->exec, {1, 1});
    gko::detail::DenseDualCache<value_type> cache;
    cache = this->cache;

    ASSERT_EQ(cache.get_const(this->exec), nullptr);
    ASSERT_EQ(cache.get_const(this->host), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache.get_const(this->exec)->get_size(),
                                gko::dim<2>(1, 1));
}


TYPED_TEST(DenseDualCache, VectorIsNotMoveAssigned)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->exec, {1, 1});
    gko::detail::DenseDualCache<value_type> cache;
    cache = std::move(this->cache);

    ASSERT_EQ(cache.get_const(this->exec), nullptr);
    ASSERT_EQ(cache.get_const(this->host), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache.get_const(this->exec)->get_size(),
                                gko::dim<2>(1, 1));
}


TYPED_TEST(DenseDualCache, SynchronizesAfterWriteToHost)
{
    gko::dim<2> size{4, 7};
    this->cache.init(this->exec, size);

    auto host = this->cache.get(this->host);
    host->fill(1.0);
    auto device = this->cache.get_const(this->exec);

    GKO_ASSERT_MTX_NEAR(host, device, 0.0);
}


TYPED_TEST(DenseDualCache, NoSynchronizeAfterRead)
{
    gko::dim<2> size{4, 7};
    auto log = std::make_shared<copy_logger>();
    this->exec->add_logger(log);
    this->ref->add_logger(log);
    this->cache.init(this->exec, size);
    auto device = this->cache.get(this->exec);
    device->fill(1.0);
    auto host = this->cache.get_const(this->host);

    {
        SCOPED_TRACE("Access host");
        auto counter_before = log->count;
        auto host2 = this->cache.get_const(this->host);
        auto counter_after = log->count;

        ASSERT_EQ(counter_before, counter_after);
    }
    {
        SCOPED_TRACE("Access device");
        auto counter_before = log->count;
        auto device2 = this->cache.get_const(this->exec);
        auto counter_after = log->count;

        ASSERT_EQ(counter_before, counter_after);
    }
}
