// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/memory.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"


namespace {


#if !GKO_HAVE_UMPIRE


TEST(UmpireAllocatorWithoutUmpire, CpuConstructorThrowsNotCompiled)
{
    ASSERT_THROW(gko::UmpireCpuAllocator{"any_name"}, gko::NotCompiled);
}


TEST(UmpireAllocatorWithoutUmpire, CudaConstructorThrowsNotCompiled)
{
    ASSERT_THROW(gko::UmpireCudaAllocator{"any_name"}, gko::NotCompiled);
}


TEST(UmpireAllocatorWithoutUmpire, HipConstructorThrowsNotCompiled)
{
    ASSERT_THROW(gko::UmpireHipAllocator{"any_name"}, gko::NotCompiled);
}


TEST(UmpireAllocatorWithoutUmpire, PoolFactoryThrowsNotCompiled)
{
    ASSERT_THROW(gko::make_umpire_cpu_pool("any_name"), gko::NotCompiled);
}


#endif  // !GKO_HAVE_UMPIRE


#if GKO_HAVE_UMPIRE


// Small initial block: the pools below live until process exit, so the
// 512 MiB default would be wasteful on a shared host.
gko::umpire_pool_config small_pool_config()
{
    gko::umpire_pool_config config;
    config.initial_block_size = gko::size_type{1} << 20;  // 1 MiB
    return config;
}


// Runs `f`, expected to throw a gko::Error, and returns its message.
template <typename Callable>
std::string error_message(Callable&& f)
{
    try {
        f();
    } catch (const gko::Error& e) {
        return e.what();
    }
    ADD_FAILURE() << "expected a gko::Error";
    return {};
}


bool contains(const std::string& text, const std::string& phrase)
{
    return text.find(phrase) != std::string::npos;
}


// *DeathTest suites run first, before any test has started OpenMP threads.
TEST(UmpireAllocatorDeathTest, DeallocatingForeignPointerExits)
{
    auto alloc =
        gko::make_umpire_cpu_pool("gko_test_pool_foreign", small_pool_config());
    int not_from_the_pool = 0;

    // Umpire throws on the foreign pointer; the noexcept deallocate() exits.
    EXPECT_EXIT(alloc->deallocate(&not_from_the_pool),
                ::testing::ExitedWithCode(1), "");
}


TEST(UmpireAllocator, UnknownNameThrows)
{
    const auto message = error_message(
        [] { gko::UmpireCpuAllocator{"definitely_not_registered"}; });

    ASSERT_TRUE(contains(message,
                         "No Umpire allocator named "
                         "'definitely_not_registered'"))
        << message;
}


TEST(UmpireAllocator, PoolFactoryCreatesUsableAllocator)
{
    auto alloc =
        gko::make_umpire_cpu_pool("gko_test_pool_basic", small_pool_config());

    ASSERT_EQ(alloc->get_name(), "gko_test_pool_basic");
}


TEST(UmpireAllocator, PoolFactoryRejectsUnknownResource)
{
    auto config = small_pool_config();
    config.resource = "definitely_not_a_resource";

    ASSERT_THROW(
        gko::make_umpire_cpu_pool("gko_test_pool_bad_resource", config),
        gko::Error);
}


TEST(UmpireAllocator, RejectsNonPowerOfTwoAlignment)
{
    auto config = small_pool_config();
    config.alignment = 100;

    ASSERT_THROW(
        gko::make_umpire_cpu_pool("gko_test_pool_bad_alignment", config),
        gko::Error);
}


TEST(UmpireAllocator, AllocatesAndDeallocates)
{
    auto alloc = gko::make_umpire_cpu_pool("gko_test_pool_roundtrip",
                                           small_pool_config());
    const gko::size_type num_bytes = 1024;

    auto ptr = alloc->allocate(num_bytes);

    ASSERT_NE(ptr, nullptr);
    ASSERT_GE(alloc->get_current_size(), num_bytes);
    alloc->deallocate(ptr);
}


TEST(UmpireAllocator, HandlesZeroByteAllocation)
{
    auto alloc =
        gko::make_umpire_cpu_pool("gko_test_pool_zero", small_pool_config());

    // Umpire returns a non-null pointer here, not nullptr.
    auto ptr = alloc->allocate(0);

    ASSERT_NE(ptr, nullptr);
    alloc->deallocate(ptr);
}


TEST(UmpireAllocator, ReusesAlreadyRegisteredPool)
{
    auto first =
        gko::make_umpire_cpu_pool("gko_test_pool_shared", small_pool_config());
    auto different = small_pool_config();
    different.initial_block_size = 1024;
    std::shared_ptr<gko::UmpireCpuAllocator> second;

    ASSERT_NO_THROW(
        second = gko::make_umpire_cpu_pool("gko_test_pool_shared", different));
    auto ptr = first->allocate(4096);

    ASSERT_GE(second->get_current_size(), 4096);
    first->deallocate(ptr);
}


TEST(UmpireAllocator, ActualSizeIsAtLeastCurrentSize)
{
    auto alloc =
        gko::make_umpire_cpu_pool("gko_test_pool_sizes", small_pool_config());
    auto ptr = alloc->allocate(4096);

    ASSERT_GE(alloc->get_actual_size(), alloc->get_current_size());

    alloc->deallocate(ptr);
}


TEST(UmpireAllocator, SizeReturnsToPriorValueAfterDeallocate)
{
    auto alloc = gko::make_umpire_cpu_pool("gko_test_pool_prior_size",
                                           small_pool_config());
    const auto size_before = alloc->get_current_size();

    auto ptr = alloc->allocate(4096);

    ASSERT_GE(alloc->get_current_size(), size_before + 4096);

    alloc->deallocate(ptr);

    ASSERT_EQ(alloc->get_current_size(), size_before);
}


TEST(UmpireAllocator, TracksHighWatermarkAndAllocationCount)
{
    auto alloc =
        gko::make_umpire_cpu_pool("gko_test_pool_stats", small_pool_config());
    const auto count_before = alloc->get_allocation_count();

    auto ptr = alloc->allocate(4096);

    ASSERT_EQ(alloc->get_allocation_count(), count_before + 1);
    ASSERT_GE(alloc->get_high_watermark(), gko::size_type{4096});

    alloc->deallocate(ptr);

    ASSERT_EQ(alloc->get_allocation_count(), count_before);
    ASSERT_GE(alloc->get_high_watermark(), gko::size_type{4096});
}


TEST(UmpireAllocator, ReleaseReturnsUnusedBlocks)
{
    const auto config = small_pool_config();
    auto alloc = gko::make_umpire_cpu_pool("gko_test_pool_release", config);
    // larger than the first block, so the pool has to grow
    auto ptr = alloc->allocate(2 * config.initial_block_size);
    alloc->deallocate(ptr);
    const auto actual_before = alloc->get_actual_size();

    alloc->release();

    ASSERT_GE(actual_before, 2 * config.initial_block_size);
    ASSERT_LT(alloc->get_actual_size(), actual_before);
}


TEST(UmpireAllocator, FailedAllocationThrowsAllocationError)
{
    auto alloc =
        gko::make_umpire_cpu_pool("gko_test_pool_oom", small_pool_config());
    // 1 EiB: no host can provide it.
    const auto huge = gko::size_type{1} << 60;

    ASSERT_THROW(alloc->allocate(huge), gko::AllocationError);
    ASSERT_TRUE(contains(error_message([&] { alloc->allocate(huge); }),
                         "umpire:gko_test_pool_oom"));
}


TEST(UmpireAllocator, ThreadSafePoolIsUsable)
{
    auto config = small_pool_config();
    config.thread_safe = true;
    auto alloc = gko::make_umpire_cpu_pool("gko_test_pool_threadsafe", config);

    auto ptr = alloc->allocate(4096);

    ASSERT_NE(ptr, nullptr);
    ASSERT_GE(alloc->get_current_size(), gko::size_type{4096});
    alloc->deallocate(ptr);

    // the unwrapped pool is registered as "<name>_pool"
    ASSERT_NO_THROW(gko::UmpireCpuAllocator{"gko_test_pool_threadsafe_pool"});
}


TEST(UmpireAllocator, HostPoolRejectedByCudaAllocator)
{
    gko::make_umpire_cpu_pool("gko_test_pool_host_only", small_pool_config());

    const auto message = error_message(
        [] { gko::UmpireCudaAllocator{"gko_test_pool_host_only"}; });

    ASSERT_TRUE(
        contains(message, "'gko_test_pool_host_only' is not device-accessible"))
        << message;
}


TEST(UmpireAllocator, PoolFactoryRejectsWrongResourceKindBeforeRegistering)
{
    // Fresh name per run, so --gtest_repeat does not reuse a registered pool.
    static int run = 0;
    const auto name = "gko_test_pool_wrong_kind_" + std::to_string(run++);
    auto device_config = small_pool_config();
    device_config.resource = "HOST";

    const auto message =
        error_message([&] { gko::make_umpire_cuda_pool(name, device_config); });

    ASSERT_TRUE(contains(message, "memory resource 'HOST' for pool '" + name +
                                      "' is not device-accessible"))
        << message;
    // A doubled first block shows the corrected call made its own pool.
    auto host_config = small_pool_config();
    host_config.resource = "HOST";
    host_config.initial_block_size = 2 * device_config.initial_block_size;
    std::shared_ptr<gko::UmpireCpuAllocator> alloc;
    ASSERT_NO_THROW(alloc = gko::make_umpire_cpu_pool(name, host_config));
    auto ptr = alloc->allocate(256);
    ASSERT_GE(alloc->get_actual_size(), host_config.initial_block_size);
    alloc->deallocate(ptr);
}


TEST(UmpireAllocator, WorksAsOmpExecutorAllocator)
{
    auto alloc =
        gko::make_umpire_cpu_pool("gko_test_pool_exec", small_pool_config());
    auto exec = gko::OmpExecutor::create(alloc);
    const auto size_before = alloc->get_current_size();

    auto array = gko::array<int>(exec, 128);

    ASSERT_GE(alloc->get_current_size(), size_before + 128 * sizeof(int));
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(array.get_data()) % 256, 0u);

    std::fill_n(array.get_data(), 128, 42);
    for (gko::size_type i = 0; i < 128; ++i) {
        ASSERT_EQ(array.get_data()[i], 42);
    }
}


TEST(UmpireAllocator, CsrApplyOnPoolMemoryMatchesDefaultAllocator)
{
    using value_type = double;
    using index_type = gko::int32;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    auto alloc =
        gko::make_umpire_cpu_pool("gko_test_pool_csr", small_pool_config());
    auto umpire_exec = gko::ReferenceExecutor::create(alloc);
    auto default_exec = gko::ReferenceExecutor::create();
    // nonsymmetric, with an empty row
    auto make_matrix = [](std::shared_ptr<const gko::Executor> exec) {
        return gko::initialize<Csr>({{2.0, -1.0, 0.0, 0.5},
                                     {-1.0, 2.0, -1.0, 0.0},
                                     {0.0, 0.0, 0.0, 0.0},
                                     {0.25, 0.0, -1.0, 2.0}},
                                    exec);
    };
    auto make_vector = [](std::shared_ptr<const gko::Executor> exec) {
        return gko::initialize<Dense>({1.0, 2.0, -3.0, 4.0}, exec);
    };
    const auto size_before = alloc->get_current_size();
    auto mtx = make_matrix(umpire_exec);
    auto b = make_vector(umpire_exec);
    auto x = Dense::create(umpire_exec, gko::dim<2>{4, 1});
    auto default_mtx = make_matrix(default_exec);
    auto default_b = make_vector(default_exec);
    auto default_x = Dense::create(default_exec, gko::dim<2>{4, 1});
    ASSERT_GT(alloc->get_current_size(), size_before);

    mtx->apply(b, x);
    default_mtx->apply(default_b, default_x);

    GKO_ASSERT_MTX_NEAR(x, default_x, r<value_type>::value);
}


#endif  // GKO_HAVE_UMPIRE


}  // namespace
