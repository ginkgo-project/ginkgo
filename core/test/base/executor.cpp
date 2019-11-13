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

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/memory_space.hpp>


#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>


namespace {


using exec_ptr = std::shared_ptr<gko::Executor>;


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int &val) : value(val) {}
    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        value = 1;
    }
    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        value = 2;
    }
    void run(std::shared_ptr<const gko::HipExecutor>) const override
    {
        value = 3;
    }
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = 4;
    }

    int &value;
};


TEST(OmpExecutor, CanBeCreatedWithAssociatedMemorySpace)
{
    auto mem_space = gko::HostMemorySpace::create();
    exec_ptr omp = gko::OmpExecutor::create(mem_space);

    ASSERT_EQ(omp->get_mem_space(), mem_space);
}


TEST(OmpExecutor, FailsWithInvalidMemorySpace)
{
    auto mem_space = gko::CudaMemorySpace::create(0);

    ASSERT_THROW(gko::OmpExecutor::create(mem_space), gko::MemSpaceMismatch);
}


TEST(OmpExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr omp = gko::OmpExecutor::create();

    omp->run(ExampleOperation(value));
    ASSERT_EQ(1, value);
}


#if GKO_HAVE_HWLOC


TEST(OmpExecutor, GetsExecInfo)
{
    int num_pus = 0;
    int num_cores = 0;
    int num_numas = 0;
    auto omp = gko::OmpExecutor::create();
    auto omp_info = omp->get_exec_info();
    num_pus = omp_info->get_num_pus();
    num_cores = omp_info->get_num_cores();
    num_numas = omp_info->get_num_numas();
    ASSERT_NE(0, num_numas);
    ASSERT_NE(0, num_pus);
    ASSERT_NE(0, num_cores);
}


#endif


TEST(OmpExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    exec_ptr omp = gko::OmpExecutor::create();

    omp->run(omp_lambda, cuda_lambda, hip_lambda);
    ASSERT_EQ(1, value);
}


TEST(OmpExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr omp = gko::OmpExecutor::create();
    int *copy = omp->get_mem_space()->alloc<int>(num_elems);

    // user code is run on the OMP, so local variables are in OMP memory
    omp->copy(num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    omp->get_mem_space()->free(copy);
}


TEST(OmpExecutor, IsItsOwnMaster)
{
    exec_ptr omp = gko::OmpExecutor::create();

    ASSERT_EQ(omp, omp->get_master());
}


TEST(ReferenceExecutor, CanBeCreatedWithAssociatedMemorySpace)
{
    auto mem_space = gko::HostMemorySpace::create();
    exec_ptr ref = gko::ReferenceExecutor::create(mem_space);

    ASSERT_EQ(ref->get_mem_space(), mem_space);
}


TEST(ReferenceExecutor, FailsWithInvalidMemorySpace)
{
    auto mem_space = gko::CudaMemorySpace::create(0);

    ASSERT_THROW(gko::ReferenceExecutor::create(mem_space),
                 gko::MemSpaceMismatch);
}


TEST(ReferenceExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr ref = gko::ReferenceExecutor::create();

    ref->run(ExampleOperation(value));
    ASSERT_EQ(4, value);
}


TEST(ReferenceExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    exec_ptr ref = gko::ReferenceExecutor::create();

    ref->run(omp_lambda, cuda_lambda, hip_lambda);
    ASSERT_EQ(1, value);
}


TEST(ReferenceExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *copy = ref->get_mem_space()->alloc<int>(num_elems);

    // ReferenceExecutor is a type of OMP executor, so this is O.K.
    ref->copy(num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->get_mem_space()->free(copy);
}


TEST(ReferenceExecutor, CopiesSingleValue)
{
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *el = ref->get_mem_space()->alloc<int>(1);
    el[0] = 83683;

    EXPECT_EQ(83683, ref->copy_val_to_host(el));

    ref->get_mem_space()->free(el);
}


TEST(ReferenceExecutor, IsItsOwnMaster)
{
    exec_ptr ref = gko::ReferenceExecutor::create();

    ASSERT_EQ(ref, ref->get_master());
}


TEST(CudaExecutor, CanBeCreatedWithAssociatedMemorySpace)
{
    auto mem_space = gko::CudaMemorySpace::create(0);
    auto uvm_space = gko::CudaUVMSpace::create(0);
    exec_ptr cuda =
        gko::CudaExecutor::create(0, mem_space, gko::OmpExecutor::create());
    exec_ptr cuda2 =
        gko::CudaExecutor::create(0, uvm_space, gko::OmpExecutor::create());

    ASSERT_EQ(cuda->get_mem_space(), mem_space);
    ASSERT_EQ(cuda2->get_mem_space(), uvm_space);
}


TEST(CudaExecutor, FailsWithInvalidMemorySpace)
{
    auto mem_space = gko::HostMemorySpace::create();

    ASSERT_THROW(
        gko::CudaExecutor::create(0, mem_space, gko::OmpExecutor::create()),
        gko::MemSpaceMismatch);
}


TEST(CudaExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr cuda =
        gko::CudaExecutor::create(0, gko::OmpExecutor::create(), true);

    cuda->run(ExampleOperation(value));
    ASSERT_EQ(2, value);
}


TEST(CudaExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    exec_ptr cuda =
        gko::CudaExecutor::create(0, gko::OmpExecutor::create(), true);

    cuda->run(omp_lambda, cuda_lambda, hip_lambda);
    ASSERT_EQ(2, value);
}


TEST(CudaExecutor, KnowsItsMaster)
{
    auto omp = gko::OmpExecutor::create();
    exec_ptr cuda = gko::CudaExecutor::create(0, omp);

    ASSERT_EQ(omp, cuda->get_master());
}


TEST(CudaExecutor, KnowsItsDeviceId)
{
    auto omp = gko::OmpExecutor::create();
    auto cuda = gko::CudaExecutor::create(0, omp);

    ASSERT_EQ(0, cuda->get_device_id());
}


TEST(CudaExecutor, CanGetDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto cuda = gko::CudaExecutor::create(0, omp);

    ASSERT_EQ(false, cuda->get_device_reset());
}


TEST(CudaExecutor, CanSetDefaultDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto cuda = gko::CudaExecutor::create(0, omp, true);

    ASSERT_EQ(true, cuda->get_device_reset());
}


TEST(CudaExecutor, CanSetDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto cuda = gko::CudaExecutor::create(0, omp);

    cuda->set_device_reset(true);

    ASSERT_EQ(true, cuda->get_device_reset());
}


TEST(HipExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr hip = gko::HipExecutor::create(0, gko::OmpExecutor::create());

    hip->run(ExampleOperation(value));
    ASSERT_EQ(3, value);
}


TEST(HipExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    exec_ptr hip = gko::HipExecutor::create(0, gko::OmpExecutor::create());

    hip->run(omp_lambda, cuda_lambda, hip_lambda);
    ASSERT_EQ(3, value);
}


TEST(HipExecutor, CanBeCreatedWithAssociatedMemorySpace)
{
    auto mem_space = gko::HipMemorySpace::create(0);
    exec_ptr hip =
        gko::HipExecutor::create(0, mem_space, gko::OmpExecutor::create());

    ASSERT_EQ(hip->get_mem_space(), mem_space);
}


TEST(HipExecutor, FailsWithInvalidMemorySpace)
{
    auto mem_space = gko::HostMemorySpace::create();

    ASSERT_THROW(
        gko::HipExecutor::create(0, mem_space, gko::OmpExecutor::create()),
        gko::MemSpaceMismatch);
}


TEST(HipExecutor, KnowsItsMaster)
{
    auto omp = gko::OmpExecutor::create();
    exec_ptr hip = gko::HipExecutor::create(0, omp);

    ASSERT_EQ(omp, hip->get_master());
}


TEST(HipExecutor, KnowsItsDeviceId)
{
    auto omp = gko::OmpExecutor::create();
    auto hip = gko::HipExecutor::create(0, omp);

    ASSERT_EQ(0, hip->get_device_id());
}


TEST(HipExecutor, CanGetDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto hip = gko::HipExecutor::create(0, omp);

    ASSERT_EQ(false, hip->get_device_reset());
}


TEST(HipExecutor, CanSetDefaultDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto hip = gko::HipExecutor::create(0, omp, true);

    ASSERT_EQ(true, hip->get_device_reset());
}


TEST(HipExecutor, CanSetDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto hip = gko::HipExecutor::create(0, omp);

    hip->set_device_reset(true);

    ASSERT_EQ(true, hip->get_device_reset());
}


}  // namespace
