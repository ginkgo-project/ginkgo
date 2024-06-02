// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <Kokkos_Core.hpp>


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


TEST(Executor, ThrowsIfIncompatibleSpaces)
{
#ifdef KOKKOS_ENABLE_SERIAL
    auto exec = gko::ext::kokkos::create_executor(Kokkos::Serial{});
    auto obj = gko::array<int>(exec);

#ifdef KOKKOS_ENABLE_CUDA
    ASSERT_THROW(
        gko::ext::kokkos::detail::assert_compatibility<Kokkos::CudaSpace>(obj),
        gko::InvalidStateError);
#elif KOKKOS_ENABLE_HIP
    ASSERT_THROW(
        gko::ext::kokkos::detail::assert_compatibility<Kokkos::HIPSpace>(obj),
        gko::InvalidStateError);
#elif KOKKOS_ENABLE_SYCL
    ASSERT_THROW(gko::ext::kokkos::detail::assert_compatibility<
                     Kokkos::Experimental::SYCL>(obj),
                 gko::InvalidStateError);
#endif
#endif
}


TEST(Executor, DoesntThrowIfCompatibleSpaces)
{
#ifdef KOKKOS_ENABLE_SERIAL
    auto exec = gko::ext::kokkos::create_executor(Kokkos::Serial{});
    auto obj = gko::array<int>(exec);

    ASSERT_NO_THROW(
        gko::ext::kokkos::detail::assert_compatibility<Kokkos::HostSpace>(obj));
#endif
}


void shared_memory_access()
{
    auto exec = gko::ext::kokkos::create_executor(
        Kokkos::DefaultExecutionSpace{}, Kokkos::SharedSpace{});
    auto data = exec->alloc<int>(1);

    // If `create_executor` would not use the UVM allocators, than this
    // would crash the program. Otherwise, the device memory is accessible
    // on the CPU without issues.
    *data = 10;

    exec->free(data);
    std::exit(EXIT_SUCCESS);
}


TEST(Executor, CanCreateExecutorWithMemorySpace)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    EXPECT_EXIT(shared_memory_access(), testing::ExitedWithCode(EXIT_SUCCESS),
                "");
#endif
}
