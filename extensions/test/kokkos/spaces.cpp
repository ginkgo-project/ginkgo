/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/extensions/kokkos/spaces.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/native_type.hpp>


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


void shared_memory_access()
{
    auto exec = gko::ext::kokkos::create_executor(
        Kokkos::DefaultExecutionSpace{}, Kokkos::SharedSpace{});
    auto data = exec->alloc<int>(1);

    *data = 10;
}


TEST(Executor, CanCreateExecutorWithMemorySpace)
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    ASSERT_EXIT(shared_memory_access(); testing::ExitedWithCode(0), "Success");
#endif
}
