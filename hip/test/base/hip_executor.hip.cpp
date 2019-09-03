#include "hip/hip_runtime.h"
/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include <hip/hip_runtime.h>
#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace {


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int &val) : value(val) {}
    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        value = -1;
    }
    void run(std::shared_ptr<const gko::HipExecutor>) const override
    {
        hipGetDevice(&value);
    }
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = -2;
    }

    int &value;
};


class HipExecutor : public ::testing::Test {
protected:
    HipExecutor() : omp(gko::OmpExecutor::create()), hip(nullptr), hip2(nullptr)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        hip = gko::HipExecutor::create(0, omp);
        hip2 = gko::HipExecutor::create(gko::HipExecutor::get_num_devices() - 1,
                                        omp);
    }

    void TearDown()
    {
        if (hip != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

    std::shared_ptr<gko::Executor> omp;
    std::shared_ptr<gko::HipExecutor> hip;
    std::shared_ptr<gko::HipExecutor> hip2;
};


TEST_F(HipExecutor, CanInstantiateTwoExecutorsOnOneDevice)
{
    auto hip = gko::HipExecutor::create(0, omp);
    auto hip2 = gko::HipExecutor::create(0, omp);

    // We want automatic deinitialization to not create any error
}


TEST_F(HipExecutor, MasterKnowsNumberOfDevices)
{
    int count = 0;
    hipGetDeviceCount(&count);
    ASSERT_EQ(count, gko::HipExecutor::get_num_devices());
}


TEST_F(HipExecutor, AllocatesAndFreesMemory)
{
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = hip->alloc<int>(2));
    ASSERT_NO_THROW(hip->free(ptr));
}


TEST_F(HipExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    int *ptr = nullptr;

    ASSERT_THROW(
        {
            ptr = hip->alloc<int>(num_elems);
            hip->synchronize();
        },
        gko::AllocationError);

    hip->free(ptr);
}


__global__ void check_data(int *data)
{
    if (data[0] != 3 || data[1] != 8) {
        asm("trap;");
    }
}

TEST_F(HipExecutor, CopiesDataToHip)
{
    int orig[] = {3, 8};
    auto *copy = hip->alloc<int>(2);

    hip->copy_from(omp.get(), 2, orig, copy);

    hipLaunchKernelGGL((check_data), dim3(1), dim3(1), 0, 0, copy);
    ASSERT_NO_THROW(hip->synchronize());
    hip->free(copy);
}


__global__ void init_data(int *data)
{
    data[0] = 3;
    data[1] = 8;
}

TEST_F(HipExecutor, CopiesDataFromHip)
{
    int copy[2];
    auto orig = hip->alloc<int>(2);
    hipLaunchKernelGGL((init_data), dim3(1), dim3(1), 0, 0, orig);

    omp->copy_from(hip.get(), 2, orig, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    hip->free(orig);
}

/* Properly checks if it works only when multiple GPUs exist */
TEST_F(HipExecutor, PreservesDeviceSettings)
{
    auto previous_device = gko::HipExecutor::get_num_devices() - 1;
    GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(previous_device));
    auto orig = hip->alloc<int>(2);
    int current_device;
    GKO_ASSERT_NO_HIP_ERRORS(hipGetDevice(&current_device));
    ASSERT_EQ(current_device, previous_device);

    hip->free(orig);
    GKO_ASSERT_NO_HIP_ERRORS(hipGetDevice(&current_device));
    ASSERT_EQ(current_device, previous_device);
}

TEST_F(HipExecutor, RunsOnProperDevice)
{
    int value = -1;
    GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(0));
    hip2->run(ExampleOperation(value));
    ASSERT_EQ(value, hip2->get_device_id());
}

TEST_F(HipExecutor, CopiesDataFromHipToHip)
{
    int copy[2];
    auto orig = hip->alloc<int>(2);
    GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(0));
    hipLaunchKernelGGL((init_data), dim3(1), dim3(1), 0, 0, orig);

    auto copy_hip2 = hip2->alloc<int>(2);
    hip2->copy_from(hip.get(), 2, orig, copy_hip2);

    // Check that the data is really on GPU2 and ensure we did not cheat
    int value = -1;
    GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(hip2->get_device_id()));
    hipLaunchKernelGGL((check_data), dim3(1), dim3(1), 0, 0, copy_hip2);
    GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(0));
    hip2->run(ExampleOperation(value));
    ASSERT_EQ(value, hip2->get_device_id());

    omp->copy_from(hip2.get(), 2, copy_hip2, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    hip->free(copy_hip2);
    hip->free(orig);
}

TEST_F(HipExecutor, Synchronizes)
{
    // Todo design a proper unit test once we support streams
    ASSERT_NO_THROW(hip->synchronize());
}


}  // namespace
