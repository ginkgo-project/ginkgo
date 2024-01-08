// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/allocator.hpp"


#include <vector>


#include <gtest/gtest.h>


namespace {


TEST(ExecutorAllocator, Works)
{
    auto exec = gko::ReferenceExecutor::create();
    auto alloc = gko::ExecutorAllocator<int>(exec);

    int* ptr{};
    ASSERT_NO_THROW(ptr = alloc.allocate(10));
    // This test can only fail with sanitizers
    ptr[0] = 0;
    ptr[9] = 0;

    ASSERT_NO_THROW(alloc.deallocate(ptr, 10));
}


TEST(ExecutorAllocator, WorksWithStdlib)
{
    auto exec = gko::ReferenceExecutor::create();
    auto alloc = gko::ExecutorAllocator<int>(exec);
    auto vec = std::vector<int, gko::ExecutorAllocator<int>>(10, 0, exec);

    // This test can only fail with sanitizers
    vec[0] = 0;
    vec[9] = 0;
}


TEST(ExecutorAllocator, ComparesEqual)
{
    auto exec = gko::ReferenceExecutor::create();
    auto alloc1 = gko::ExecutorAllocator<int>(exec);
    auto alloc2 = gko::ExecutorAllocator<float>(exec);

    ASSERT_TRUE(alloc1 == alloc2);
}


TEST(ExecutorAllocator, ComparesNotEqual)
{
    auto exec1 = gko::ReferenceExecutor::create();
    auto exec2 = gko::OmpExecutor::create();
    auto alloc1 = gko::ExecutorAllocator<int>(exec1);
    auto alloc2 = gko::ExecutorAllocator<float>(exec2);

    ASSERT_TRUE(alloc1 != alloc2);
}


TEST(ExecutorAllocator, MovedFromComparesEqual)
{
    auto exec = gko::ReferenceExecutor::create();
    auto alloc1 = gko::ExecutorAllocator<int>(exec);

    auto alloc2 = gko::ExecutorAllocator<int>(std::move(alloc1));

    ASSERT_TRUE(alloc1 == alloc2);
    ASSERT_EQ(alloc1.get_executor(), exec);
}


TEST(ExecutorAllocator, MovedFromMixedComparesEqual)
{
    auto exec = gko::ReferenceExecutor::create();
    auto alloc1 = gko::ExecutorAllocator<int>(exec);

    auto alloc2 = gko::ExecutorAllocator<float>(std::move(alloc1));

    ASSERT_TRUE(alloc1 == alloc2);
    ASSERT_EQ(alloc1.get_executor(), exec);
}


TEST(ExecutorAllocator, CopiedFromComparesEqual)
{
    auto exec = gko::ReferenceExecutor::create();
    auto alloc1 = gko::ExecutorAllocator<int>(exec);

    auto alloc2 = gko::ExecutorAllocator<int>(alloc1);

    ASSERT_TRUE(alloc1 == alloc2);
    ASSERT_EQ(alloc1.get_executor(), exec);
}


TEST(ExecutorAllocator, CopiedFromMixedComparesEqual)
{
    auto exec = gko::ReferenceExecutor::create();
    auto alloc1 = gko::ExecutorAllocator<int>(exec);

    auto alloc2 = gko::ExecutorAllocator<float>(alloc1);

    ASSERT_TRUE(alloc1 == alloc2);
    ASSERT_EQ(alloc1.get_executor(), exec);
}


TEST(ExecutorAllocator, MoveAssignedComparesEqual)
{
    auto exec1 = gko::ReferenceExecutor::create();
    auto exec2 = gko::ReferenceExecutor::create();
    auto alloc1 = gko::ExecutorAllocator<int>(exec1);
    auto alloc2 = gko::ExecutorAllocator<int>(exec2);

    alloc2 = std::move(alloc1);

    ASSERT_TRUE(alloc1 == alloc2);
    ASSERT_EQ(alloc1.get_executor(), exec1);
}


TEST(ExecutorAllocator, MoveAssignedMixedComparesEqual)
{
    auto exec1 = gko::ReferenceExecutor::create();
    auto exec2 = gko::ReferenceExecutor::create();
    auto alloc1 = gko::ExecutorAllocator<int>(exec1);
    auto alloc2 = gko::ExecutorAllocator<float>(exec2);

    alloc2 = std::move(alloc1);

    ASSERT_TRUE(alloc1 == alloc2);
    ASSERT_EQ(alloc1.get_executor(), exec1);
}


TEST(ExecutorAllocator, CopyAssignedComparesEqual)
{
    auto exec1 = gko::ReferenceExecutor::create();
    auto exec2 = gko::ReferenceExecutor::create();
    auto alloc1 = gko::ExecutorAllocator<int>(exec1);
    auto alloc2 = gko::ExecutorAllocator<int>(exec2);

    alloc2 = alloc1;

    ASSERT_TRUE(alloc1 == alloc2);
    ASSERT_EQ(alloc1.get_executor(), exec1);
}


TEST(ExecutorAllocator, CopyAssignedMixedComparesEqual)
{
    auto exec1 = gko::ReferenceExecutor::create();
    auto exec2 = gko::ReferenceExecutor::create();
    auto alloc1 = gko::ExecutorAllocator<int>(exec1);
    auto alloc2 = gko::ExecutorAllocator<float>(exec2);

    alloc2 = alloc1;

    ASSERT_TRUE(alloc1 == alloc2);
    ASSERT_EQ(alloc1.get_executor(), exec1);
}


}  // namespace
