// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/memory.hpp>


#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>


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


}  // namespace
