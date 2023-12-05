// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include <ginkgo/extensions/kokkos/spaces.hpp>


#include "core/test/gtest/environments.hpp"
#include "core/test/utils.hpp"


TEST(Executor, CanCreateDefaultExecutor)
{
    auto exec = gko::ext::kokkos::create_default_executor();

#if defined(KOKKOS_ENABLE_CUDA)
    ASSERT_TRUE(std::dynamic_pointer_cast<gko::CudaExecutor>(exec));
#elif defined(KOKKOS_ENABLE_HIP)
    ASSERT_TRUE(std::dynamic_pointer_cast<gko::HipExecutor>(exec));
#elif defined(KOKKOS_ENABLE_SYCL)
    ASSERT_TRUE(std::dynamic_pointer_cast<gko::DpcppExecutor>(exec));
#elif defined(KOKKOS_ENABLE_OPENMP)
    // necessary because of our executor hierarchy...
    ASSERT_FALSE(std::dynamic_pointer_cast<gko::ReferenceExecutor>(exec));
    ASSERT_TRUE(std::dynamic_pointer_cast<gko::OmpExecutor>(exec));
#elif defined(KOKKOS_ENABLE_SERIAL)
    ASSERT_TRUE(std::dynamic_pointer_cast<gko::ReferenceExecutor>(exec));
#else
    ASSERT_TRUE(false);
#endif
}


TEST(Executor, CanCreateExecutorWithExecutorSpace)
{
#ifdef KOKKOS_ENABLE_SERIAL
    auto exec = gko::ext::kokkos::create_executor(Kokkos::Serial{});

    ASSERT_TRUE(std::dynamic_pointer_cast<gko::ReferenceExecutor>(exec));
#endif
}


// void shared_memory_access()
// {
//     auto exec = gko::ext::kokkos::create_executor(
//         Kokkos::DefaultExecutionSpace{}, Kokkos::SharedSpace{});
//     auto data = exec->alloc<int>(1);
//
//     *data = 10;
//
//     exec->free(data);
// }
//
//
// TEST(Executor, CanCreateExecutorWithMemorySpace)
// {
// #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
//     ASSERT_EXIT(shared_memory_access(), testing::ExitedWithCode(0),
//     "Success");
// #endif
// }
