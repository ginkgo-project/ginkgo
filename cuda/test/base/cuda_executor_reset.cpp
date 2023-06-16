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

#include <ginkgo/core/base/executor.hpp>


#include <thread>


#include <gtest/gtest.h>


namespace {


#define GTEST_ASSERT_NO_EXIT(statement) \
    ASSERT_EXIT({ {statement} exit(0); }, ::testing::ExitedWithCode(0), "")


TEST(DeviceReset, HipCuda)
{
    GTEST_ASSERT_NO_EXIT({
        auto ref = gko::ReferenceExecutor::create();
        auto hip = gko::HipExecutor::create(0, ref, true);
        auto cuda = gko::CudaExecutor::create(0, ref, true);
    });
}


TEST(DeviceReset, CudaHip)
{
    GTEST_ASSERT_NO_EXIT({
        auto ref = gko::ReferenceExecutor::create();
        auto cuda = gko::CudaExecutor::create(0, ref, true);
        auto hip = gko::HipExecutor::create(0, ref, true);
    });
}


void func()
{
    auto ref = gko::ReferenceExecutor::create();
    auto exec = gko::CudaExecutor::create(0, ref, true);
}


TEST(DeviceReset, CudaCuda)
{
    GTEST_ASSERT_NO_EXIT({
        std::thread t1(func);
        std::thread t2(func);
        t1.join();
        t2.join();
    });
}


}  // namespace
