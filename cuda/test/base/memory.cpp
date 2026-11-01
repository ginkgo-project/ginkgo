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

#include "cuda/test/utils.hpp"


namespace {


class Memory : public CudaTestFixture {
protected:
    Memory()
        : host_exec_with_pinned{gko::OmpExecutor::create(
              std::make_shared<gko::CudaHostAllocator>(exec->get_device_id()))},
          host_exec_with_unified{gko::OmpExecutor::create(
              std::make_shared<gko::CudaUnifiedAllocator>(
                  exec->get_device_id()))},
          exec_with_normal{gko::CudaExecutor::create(
              exec->get_device_id(), ref,
              std::make_shared<gko::CudaAllocator>(), exec->get_stream())},
          exec_with_async{gko::CudaExecutor::create(
              exec->get_device_id(), host_exec_with_pinned,
              std::make_shared<gko::CudaAsyncAllocator>(exec->get_stream()),
              exec->get_stream())},
          exec_with_unified{gko::CudaExecutor::create(
              exec->get_device_id(), host_exec_with_unified,
              std::make_shared<gko::CudaUnifiedAllocator>(
                  exec->get_device_id()),
              exec->get_stream())}
    {}

    std::shared_ptr<gko::OmpExecutor> host_exec_with_pinned;
    std::shared_ptr<gko::OmpExecutor> host_exec_with_unified;
    std::shared_ptr<gko::CudaExecutor> exec_with_normal;
    std::shared_ptr<gko::CudaExecutor> exec_with_async;
    std::shared_ptr<gko::CudaExecutor> exec_with_unified;
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


TEST(UmpireCudaAllocator, AllocatesAndDeallocatesDeviceMemory)
{
    auto alloc =
        gko::make_umpire_cuda_pool("gko_cuda_test_pool", small_pool_config());
    const gko::size_type num_bytes = 4096;

    auto ptr = alloc->allocate(num_bytes);

    ASSERT_NE(ptr, nullptr);
    ASSERT_GE(alloc->get_current_size(), num_bytes);
    alloc->deallocate(ptr);
}


TEST(UmpireCudaAllocator, PoolPointersAreUsableByKernels)
{
    auto alloc = gko::make_umpire_cuda_pool("gko_cuda_test_pool_exec",
                                            small_pool_config());
    auto ref = gko::ReferenceExecutor::create();
    auto exec = gko::CudaExecutor::create(0, ref, alloc);

    auto device_data = gko::array<int>(exec, 5);
    device_data.fill(7);

    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(device_data.get_data()) % 256,
              0u);

    auto result = gko::array<int>(ref, device_data);

    GKO_ASSERT_ARRAY_EQ(result, I<int>({7, 7, 7, 7, 7}));
}


TEST(UmpireCudaAllocator, RejectsPoolBoundToAnotherDevice)
{
    if (gko::CudaExecutor::get_num_devices() < 2) {
        GTEST_SKIP() << "needs at least two CUDA devices";
    }
    auto alloc = std::make_shared<gko::UmpireCudaAllocator>("DEVICE::0");
    auto ref = gko::ReferenceExecutor::create();

    ASSERT_THROW(gko::CudaExecutor::create(1, ref, alloc), gko::Error);
}


#endif  // GKO_HAVE_UMPIRE


}  // namespace
