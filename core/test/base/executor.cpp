/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include <type_traits>


#if defined(__unix__) || defined(__APPLE__)
#include <utmpx.h>
#endif


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>


namespace {


using exec_ptr = std::shared_ptr<gko::Executor>;


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int& val) : value(val) {}
    std::shared_ptr<gko::AsyncHandle> run(
        std::shared_ptr<const gko::OmpExecutor>) const override
    {
        auto l = [=]() { value = 1; };
        return gko::HostAsyncHandle<void>::create(
            std::async(std::launch::async, l));
    }
    std::shared_ptr<gko::AsyncHandle> run(
        std::shared_ptr<const gko::CudaExecutor>) const override
    {
        auto l = [=]() { value = 2; };
        return gko::HostAsyncHandle<void>::create(
            std::async(std::launch::async, l));
    }
    std::shared_ptr<gko::AsyncHandle> run(
        std::shared_ptr<const gko::HipExecutor>) const override
    {
        auto l = [=]() { value = 3; };
        return gko::HostAsyncHandle<void>::create(
            std::async(std::launch::async, l));
    }
    std::shared_ptr<gko::AsyncHandle> run(
        std::shared_ptr<const gko::DpcppExecutor>) const override
    {
        auto l = [=]() { value = 4; };
        return gko::HostAsyncHandle<void>::create(
            std::async(std::launch::async, l));
    }
    std::shared_ptr<gko::AsyncHandle> run(
        std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        auto l = [=]() { value = 5; };
        return gko::HostAsyncHandle<void>::create(
            std::async(std::launch::async, l));
    }

    int& value;
};


TEST(OmpExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr omp = gko::OmpExecutor::create();

    auto hand = omp->run(ExampleOperation(value));
    hand->wait();

    ASSERT_EQ(1, value);
}


TEST(OmpExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    auto dpcpp_lambda = [&value]() { value = 4; };
    exec_ptr omp = gko::OmpExecutor::create();

    omp->run(omp_lambda, cuda_lambda, hip_lambda, dpcpp_lambda);

    ASSERT_EQ(1, value);
}


TEST(OmpExecutor, IsItsOwnMaster)
{
    exec_ptr omp = gko::OmpExecutor::create();

    ASSERT_EQ(omp, omp->get_master());
}


#if GKO_HAVE_HWLOC


TEST(OmpExecutor, CanGetNumCpusFromExecInfo)
{
    auto omp = gko::OmpExecutor::create();

    auto num_cpus = omp->get_num_cores() * omp->get_num_threads_per_core();

    ASSERT_EQ(std::thread::hardware_concurrency(), num_cpus);
}


inline int get_os_id(int log_id)
{
    return gko::MachineTopology::get_instance()->get_core(log_id)->os_id;
}


TEST(MachineTopology, CanBindToASpecificCore)
{
    auto cpu_sys = sched_getcpu();

    const int bind_core = 3;
    gko::MachineTopology::get_instance()->bind_to_cores(
        std::vector<int>{bind_core});

    cpu_sys = sched_getcpu();
    ASSERT_EQ(cpu_sys, get_os_id(bind_core));
}


TEST(MachineTopology, CanBindToARangeofCores)
{
    auto cpu_sys = sched_getcpu();

    const std::vector<int> bind_core = {1, 3};
    gko::MachineTopology::get_instance()->bind_to_cores(bind_core);

    cpu_sys = sched_getcpu();
    ASSERT_TRUE(cpu_sys == get_os_id(3) || cpu_sys == get_os_id(1));
}


#endif


TEST(ReferenceExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr ref = gko::ReferenceExecutor::create();

    ref->run(ExampleOperation(value));

    ASSERT_EQ(5, value);
}


TEST(ReferenceExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    auto dpcpp_lambda = [&value]() { value = 4; };
    exec_ptr ref = gko::ReferenceExecutor::create();

    ref->run(omp_lambda, cuda_lambda, hip_lambda, dpcpp_lambda);

    ASSERT_EQ(1, value);
}


TEST(ReferenceExecutor, CopiesSingleValue)
{
    exec_ptr ref = gko::ReferenceExecutor::create();
    int* el = ref->get_mem_space()->alloc<int>(1);
    el[0] = 83683;

    EXPECT_EQ(83683, ref->copy_val_to_host(el));

    ref->get_mem_space()->free(el);
}


TEST(ReferenceExecutor, IsItsOwnMaster)
{
    exec_ptr ref = gko::ReferenceExecutor::create();

    ASSERT_EQ(ref, ref->get_master());
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
    auto dpcpp_lambda = [&value]() { value = 4; };
    exec_ptr cuda =
        gko::CudaExecutor::create(0, gko::OmpExecutor::create(), true);

    cuda->run(omp_lambda, cuda_lambda, hip_lambda, dpcpp_lambda);

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
    auto dpcpp_lambda = [&value]() { value = 4; };
    exec_ptr hip = gko::HipExecutor::create(0, gko::OmpExecutor::create());

    hip->run(omp_lambda, cuda_lambda, hip_lambda, dpcpp_lambda);

    ASSERT_EQ(3, value);
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


TEST(DpcppExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr dpcpp = gko::DpcppExecutor::create(0, gko::OmpExecutor::create());

    dpcpp->run(ExampleOperation(value));

    ASSERT_EQ(4, value);
}


TEST(DpcppExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    auto dpcpp_lambda = [&value]() { value = 4; };
    exec_ptr dpcpp = gko::DpcppExecutor::create(0, gko::OmpExecutor::create());

    dpcpp->run(omp_lambda, cuda_lambda, hip_lambda, dpcpp_lambda);

    ASSERT_EQ(4, value);
}


TEST(DpcppExecutor, KnowsItsMaster)
{
    auto omp = gko::OmpExecutor::create();
    exec_ptr dpcpp = gko::DpcppExecutor::create(0, omp);

    ASSERT_EQ(omp, dpcpp->get_master());
}


TEST(DpcppExecutor, KnowsItsDeviceId)
{
    auto omp = gko::OmpExecutor::create();
    auto dpcpp = gko::DpcppExecutor::create(0, omp);

    ASSERT_EQ(0, dpcpp->get_device_id());
}


}  // namespace
