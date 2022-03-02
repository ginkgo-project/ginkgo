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


#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/executor.hpp.inc"
#include "cuda/test/utils.hpp"


namespace {


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int& val) : value(val) {}
    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        value = -1;
    }
    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        cudaGetDevice(&value);
    }
    void run(std::shared_ptr<const gko::HipExecutor>) const override
    {
        value = -2;
    }
    void run(std::shared_ptr<const gko::DpcppExecutor>) const override
    {
        value = -3;
    }
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = -4;
    }

    int& value;
};


class ExampleAsyncOperation : public gko::AsyncOperation {
public:
    explicit ExampleAsyncOperation(int& val) : value(val) {}

    std::shared_ptr<gko::AsyncHandle> run(
        std::shared_ptr<const gko::OmpExecutor>,
        std::shared_ptr<gko::AsyncHandle> handle) const override
    {
        auto l = [=]() { value = -1; };
        return gko::HostAsyncHandle<void>::create(
            std::async(std::launch::async, l));
    }
    std::shared_ptr<gko::AsyncHandle> run(
        std::shared_ptr<const gko::CudaExecutor> exec,
        std::shared_ptr<gko::AsyncHandle> handle) const override
    {
        auto l = [=]() {
            gko::cuda::device_guard g(exec->get_device_id());
            cudaGetDevice(&value);
        };
        return gko::HostAsyncHandle<void>::create(
            std::async(std::launch::async, l));
    }
    std::shared_ptr<gko::AsyncHandle> run(
        std::shared_ptr<const gko::HipExecutor>,
        std::shared_ptr<gko::AsyncHandle> handle) const override
    {
        auto l = [=]() { value = -2; };
        return gko::HostAsyncHandle<void>::create(
            std::async(std::launch::async, l));
    }
    std::shared_ptr<gko::AsyncHandle> run(
        std::shared_ptr<const gko::DpcppExecutor>,
        std::shared_ptr<gko::AsyncHandle> handle) const override
    {
        auto l = [=]() { value = -3; };
        return gko::HostAsyncHandle<void>::create(
            std::async(std::launch::async, l));
    }
    std::shared_ptr<gko::AsyncHandle> run(
        std::shared_ptr<const gko::ReferenceExecutor>,
        std::shared_ptr<gko::AsyncHandle> handle) const override
    {
        auto l = [=]() { value = -4; };
        return gko::HostAsyncHandle<void>::create(
            std::async(std::launch::async, l));
    }

    int& value;
};


class CudaExecutor : public ::testing::Test {
protected:
    CudaExecutor()
        : omp(gko::OmpExecutor::create()),
          cuda(nullptr),
          cuda2(nullptr),
          cuda3(nullptr)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        cuda = gko::CudaExecutor::create(0, omp);
        cuda2 = gko::CudaExecutor::create(
            gko::CudaExecutor::get_num_devices() - 1, omp);
        cuda3 = gko::CudaExecutor::create(0, omp, false,
                                          gko::allocation_mode::unified_global);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    std::shared_ptr<gko::Executor> omp;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::shared_ptr<gko::CudaExecutor> cuda2;
    std::shared_ptr<gko::CudaExecutor> cuda3;
};


TEST_F(CudaExecutor, CanInstantiateTwoExecutorsOnOneDevice)
{
    auto cuda = gko::CudaExecutor::create(0, omp);
    auto cuda2 = gko::CudaExecutor::create(0, omp);

    // We want automatic deinitialization to not create any error
}


TEST_F(CudaExecutor, MasterKnowsNumberOfDevices)
{
    int count = 0;
    cudaGetDeviceCount(&count);

    auto num_devices = gko::CudaExecutor::get_num_devices();

    ASSERT_EQ(count, num_devices);
}


/* Properly checks if it works only when multiple GPUs exist */
TEST_F(CudaExecutor, PreservesDeviceSettings)
{
    auto previous_device = gko::CudaExecutor::get_num_devices() - 1;
    GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(previous_device));
    auto orig = cuda->get_mem_space()->alloc<int>(2);
    int current_device;
    GKO_ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&current_device));
    ASSERT_EQ(current_device, previous_device);

    cuda->get_mem_space()->free(orig);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&current_device));
    ASSERT_EQ(current_device, previous_device);
}


TEST_F(CudaExecutor, RunsOnProperDevice)
{
    int value = -1;

    GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(0));
    cuda2->run(ExampleOperation(value));

    ASSERT_EQ(value, cuda2->get_device_id());
}


TEST_F(CudaExecutor, Synchronizes)
{
    // Todo design a proper unit test once we support streams
    ASSERT_NO_THROW(cuda->synchronize());
}


TEST_F(CudaExecutor, ExecInfoSetsCorrectProperties)
{
    auto dev_id = cuda->get_device_id();
    auto num_sm = 0;
    auto major = 0;
    auto minor = 0;
    auto max_threads_per_block = 0;
    auto warp_size = 0;
    GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
        &num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, dev_id));
    GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
        &minor, cudaDevAttrComputeCapabilityMinor, dev_id));
    GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
        &max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, dev_id));
    GKO_ASSERT_NO_CUDA_ERRORS(
        cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, dev_id));
    auto num_cores = convert_sm_ver_to_cores(major, minor);

    ASSERT_EQ(cuda->get_major_version(), major);
    ASSERT_EQ(cuda->get_minor_version(), minor);
    ASSERT_EQ(cuda->get_num_multiprocessor(), num_sm);
    ASSERT_EQ(cuda->get_warp_size(), warp_size);
    ASSERT_EQ(cuda->get_num_warps(), num_sm * (num_cores / warp_size));
    ASSERT_EQ(cuda->get_num_warps_per_sm(), num_cores / warp_size);
}


}  // namespace
