/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/base/allocator.hpp"


#include <vector>


#include <gtest/gtest.h>


namespace {


TEST(ExecutorAllocator, Works)
{
    auto exec = gko::ReferenceExecutor::create();
    auto alloc = gko::ExecutorAllocator<int>(exec);

    int *ptr{};
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


}  // namespace
