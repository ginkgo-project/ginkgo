// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <memory>
#include <type_traits>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/memory.hpp>

#include "hip/test/utils.hip.hpp"


namespace {


class Memory : public HipTestFixture {
protected:
    Memory()
        : host_exec_with_pinned{gko::OmpExecutor::create(
              std::make_shared<gko::HipHostAllocator>(0))},
          host_exec_with_unified{gko::OmpExecutor::create(
              std::make_shared<gko::HipUnifiedAllocator>(0))},
          exec_with_normal{gko::HipExecutor::create(
              0, ref, std::make_shared<gko::HipAllocator>(),
              exec->get_stream())},
          exec_with_async{gko::HipExecutor::create(
              0, host_exec_with_pinned,
              std::make_shared<gko::HipAsyncAllocator>(exec->get_stream()),
              exec->get_stream())},
          exec_with_unified{gko::HipExecutor::create(
              0, host_exec_with_unified,
              std::make_shared<gko::HipUnifiedAllocator>(0),
              exec->get_stream())}
    {}

    std::shared_ptr<gko::OmpExecutor> host_exec_with_pinned;
    std::shared_ptr<gko::OmpExecutor> host_exec_with_unified;
    std::shared_ptr<gko::HipExecutor> exec_with_normal;
    std::shared_ptr<gko::HipExecutor> exec_with_async;
    std::shared_ptr<gko::HipExecutor> exec_with_unified;
};


TEST_F(Memory, DeviceAllocationWorks)
{
    gko::array<int> data{exec_with_normal, {1, 2}};

    GKO_ASSERT_ARRAY_EQ(data, I<int>({1, 2}));
}


TEST_F(Memory, AsyncDeviceAllocationWorks)
{
    gko::array<int> data{exec_with_async, {1, 2}};

    GKO_ASSERT_ARRAY_EQ(data, I<int>({1, 2}));
}


TEST_F(Memory, UnifiedDeviceAllocationWorks)
{
    gko::array<int> data{exec_with_unified, {1, 2}};
    exec->synchronize();

    ASSERT_EQ(data.get_const_data()[0], 1);
    ASSERT_EQ(data.get_const_data()[1], 2);
}


TEST_F(Memory, HostUnifiedAllocationWorks)
{
    gko::array<int> data{host_exec_with_unified, {1, 2}};

    ASSERT_EQ(data.get_const_data()[0], 1);
    ASSERT_EQ(data.get_const_data()[1], 2);
}


TEST_F(Memory, HostPinnedAllocationWorks)
{
    gko::array<int> data{host_exec_with_pinned, {1, 2}};

    ASSERT_EQ(data.get_const_data()[0], 1);
    ASSERT_EQ(data.get_const_data()[1], 2);
}


#if GKO_HAVE_UMPIRE


// Small initial block: the pools live until process exit, and the 512 MiB
// default would hold 1 GiB of device memory on a shared GPU.
gko::umpire_pool_config small_pool_config()
{
    gko::umpire_pool_config config;
    config.initial_block_size = gko::size_type{1} << 20;  // 1 MiB
    return config;
}


TEST(UmpireHipAllocator, AllocatesAndDeallocatesDeviceMemory)
{
    auto alloc =
        gko::make_umpire_hip_pool("gko_hip_test_pool", small_pool_config());
    const gko::size_type num_bytes = 4096;

    auto ptr = alloc->allocate(num_bytes);

    ASSERT_NE(ptr, nullptr);
    ASSERT_GE(alloc->get_current_size(), num_bytes);
    alloc->deallocate(ptr);
}


TEST(UmpireHipAllocator, PoolPointersAreUsableByKernels)
{
    auto alloc = gko::make_umpire_hip_pool("gko_hip_test_pool_exec",
                                           small_pool_config());
    auto ref = gko::ReferenceExecutor::create();
    auto exec = gko::HipExecutor::create(0, ref, alloc);

    auto device_data = gko::array<int>(exec, 5);
    device_data.fill(7);

    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(device_data.get_data()) % 256,
              0u);

    auto result = gko::array<int>(ref, device_data);

    GKO_ASSERT_ARRAY_EQ(result, I<int>({7, 7, 7, 7, 7}));
}


TEST(UmpireHipAllocator, RejectsPoolBoundToAnotherDevice)
{
    if (gko::HipExecutor::get_num_devices() < 2) {
        GTEST_SKIP() << "needs at least two HIP devices";
    }
    auto alloc = std::make_shared<gko::UmpireHipAllocator>("DEVICE::0");
    auto ref = gko::ReferenceExecutor::create();

    ASSERT_THROW(gko::HipExecutor::create(1, ref, alloc), gko::Error);
}


#endif  // GKO_HAVE_UMPIRE


}  // namespace
