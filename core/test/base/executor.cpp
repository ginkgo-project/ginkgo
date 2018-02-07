/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/base/executor.hpp>


#include <type_traits>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>


namespace {


using exec_ptr = std::shared_ptr<gko::Executor>;


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int &val) : value(val) {}
    void run(std::shared_ptr<const gko::CpuExecutor>) const override
    {
        value = 1;
    }
    void run(std::shared_ptr<const gko::GpuExecutor>) const override
    {
        value = 2;
    }
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = 3;
    }

    int &value;
};


TEST(CpuExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr cpu = gko::CpuExecutor::create();

    cpu->run(ExampleOperation(value));
    ASSERT_EQ(1, value);
}


TEST(CpuExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto cpu_lambda = [&value]() { value = 1; };
    auto gpu_lambda = [&value]() { value = 2; };
    exec_ptr cpu = gko::CpuExecutor::create();

    cpu->run(cpu_lambda, gpu_lambda);
    ASSERT_EQ(1, value);
}


TEST(CpuExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    exec_ptr cpu = gko::CpuExecutor::create();
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = cpu->alloc<int>(num_elems));
    ASSERT_NO_THROW(cpu->free(ptr));
}


TEST(CpuExecutor, FreeAcceptsNullptr)
{
    exec_ptr cpu = gko::CpuExecutor::create();
    ASSERT_NO_THROW(cpu->free(nullptr));
}


TEST(CpuExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    exec_ptr cpu = gko::CpuExecutor::create();
    int *ptr = nullptr;

    ASSERT_THROW(ptr = cpu->alloc<int>(num_elems), gko::AllocationError);

    cpu->free(ptr);
}


TEST(CpuExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr cpu = gko::CpuExecutor::create();
    int *copy = cpu->alloc<int>(num_elems);

    // user code is run on the CPU, so local variables are in CPU memory
    cpu->copy_from(cpu.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    cpu->free(copy);
}


TEST(CpuExecutor, IsItsOwnMaster)
{
    exec_ptr cpu = gko::CpuExecutor::create();

    ASSERT_EQ(cpu, cpu->get_master());
}


TEST(ReferenceExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr ref = gko::ReferenceExecutor::create();

    ref->run(ExampleOperation(value));
    ASSERT_EQ(3, value);
}


TEST(ReferenceExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto cpu_lambda = [&value]() { value = 1; };
    auto gpu_lambda = [&value]() { value = 2; };
    exec_ptr ref = gko::ReferenceExecutor::create();

    ref->run(cpu_lambda, gpu_lambda);
    ASSERT_EQ(1, value);
}


TEST(ReferenceExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = ref->alloc<int>(num_elems));
    ASSERT_NO_THROW(ref->free(ptr));
}


TEST(ReferenceExecutor, FreeAcceptsNullptr)
{
    exec_ptr cpu = gko::ReferenceExecutor::create();
    ASSERT_NO_THROW(cpu->free(nullptr));
}


TEST(ReferenceExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *ptr = nullptr;

    ASSERT_THROW(ptr = ref->alloc<int>(num_elems), gko::AllocationError);

    ref->free(ptr);
}


TEST(ReferenceExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *copy = ref->alloc<int>(num_elems);

    // ReferenceExecutor is a type of CPU executor, so this is O.K.
    ref->copy_from(ref.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, CopiesDataFromCpu)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr cpu = gko::CpuExecutor::create();
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *copy = ref->alloc<int>(num_elems);

    // ReferenceExecutor is a type of CPU executor, so this is O.K.
    ref->copy_from(cpu.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, CopiesDataToCpu)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr cpu = gko::CpuExecutor::create();
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *copy = cpu->alloc<int>(num_elems);

    // ReferenceExecutor is a type of CPU executor, so this is O.K.
    cpu->copy_from(ref.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, IsItsOwnMaster)
{
    exec_ptr ref = gko::ReferenceExecutor::create();

    ASSERT_EQ(ref, ref->get_master());
}


TEST(GpuExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr gpu = gko::GpuExecutor::create(0, gko::CpuExecutor::create());

    gpu->run(ExampleOperation(value));
    ASSERT_EQ(2, value);
}


TEST(GpuExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto cpu_lambda = [&value]() { value = 1; };
    auto gpu_lambda = [&value]() { value = 2; };
    exec_ptr gpu = gko::GpuExecutor::create(0, gko::CpuExecutor::create());

    gpu->run(cpu_lambda, gpu_lambda);
    ASSERT_EQ(2, value);
}


TEST(GpuExecutor, KnowsItsMaster)
{
    auto cpu = gko::CpuExecutor::create();
    exec_ptr gpu = gko::GpuExecutor::create(0, cpu);

    ASSERT_EQ(cpu, gpu->get_master());
}


TEST(GpuExecutor, KnowsItsDeviceId)
{
    auto cpu = gko::CpuExecutor::create();
    auto gpu = gko::GpuExecutor::create(5, cpu);

    ASSERT_EQ(5, gpu->get_device_id());
}


}  // namespace
