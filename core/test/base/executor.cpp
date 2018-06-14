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
    void run(std::shared_ptr<const gko::OmpExecutor>) const override
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


TEST(OmpExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr omp = gko::OmpExecutor::create();

    omp->run(ExampleOperation(value));
    ASSERT_EQ(1, value);
}


TEST(OmpExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto gpu_lambda = [&value]() { value = 2; };
    exec_ptr omp = gko::OmpExecutor::create();

    omp->run(omp_lambda, gpu_lambda);
    ASSERT_EQ(1, value);
}


TEST(OmpExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    exec_ptr omp = gko::OmpExecutor::create();
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = omp->alloc<int>(num_elems));
    ASSERT_NO_THROW(omp->free(ptr));
}


TEST(OmpExecutor, FreeAcceptsNullptr)
{
    exec_ptr omp = gko::OmpExecutor::create();
    ASSERT_NO_THROW(omp->free(nullptr));
}


TEST(OmpExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    exec_ptr omp = gko::OmpExecutor::create();
    int *ptr = nullptr;

    ASSERT_THROW(ptr = omp->alloc<int>(num_elems), gko::AllocationError);

    omp->free(ptr);
}


TEST(OmpExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr omp = gko::OmpExecutor::create();
    int *copy = omp->alloc<int>(num_elems);

    // user code is run on the OMP, so local variables are in OMP memory
    omp->copy_from(omp.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    omp->free(copy);
}


TEST(OmpExecutor, IsItsOwnMaster)
{
    exec_ptr omp = gko::OmpExecutor::create();

    ASSERT_EQ(omp, omp->get_master());
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
    auto omp_lambda = [&value]() { value = 1; };
    auto gpu_lambda = [&value]() { value = 2; };
    exec_ptr ref = gko::ReferenceExecutor::create();

    ref->run(omp_lambda, gpu_lambda);
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
    exec_ptr omp = gko::ReferenceExecutor::create();
    ASSERT_NO_THROW(omp->free(nullptr));
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

    // ReferenceExecutor is a type of OMP executor, so this is O.K.
    ref->copy_from(ref.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, CopiesDataFromOmp)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr omp = gko::OmpExecutor::create();
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *copy = ref->alloc<int>(num_elems);

    // ReferenceExecutor is a type of OMP executor, so this is O.K.
    ref->copy_from(omp.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, CopiesDataToOmp)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr omp = gko::OmpExecutor::create();
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *copy = omp->alloc<int>(num_elems);

    // ReferenceExecutor is a type of OMP executor, so this is O.K.
    omp->copy_from(ref.get(), num_elems, orig, copy);
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
    exec_ptr gpu = gko::GpuExecutor::create(0, gko::OmpExecutor::create());

    gpu->run(ExampleOperation(value));
    ASSERT_EQ(2, value);
}


TEST(GpuExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto gpu_lambda = [&value]() { value = 2; };
    exec_ptr gpu = gko::GpuExecutor::create(0, gko::OmpExecutor::create());

    gpu->run(omp_lambda, gpu_lambda);
    ASSERT_EQ(2, value);
}


TEST(GpuExecutor, KnowsItsMaster)
{
    auto omp = gko::OmpExecutor::create();
    exec_ptr gpu = gko::GpuExecutor::create(0, omp);

    ASSERT_EQ(omp, gpu->get_master());
}


TEST(GpuExecutor, KnowsItsDeviceId)
{
    auto omp = gko::OmpExecutor::create();
    auto gpu = gko::GpuExecutor::create(5, omp);

    ASSERT_EQ(5, gpu->get_device_id());
}


}  // namespace
