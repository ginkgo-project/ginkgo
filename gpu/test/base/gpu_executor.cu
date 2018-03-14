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
#include <core/base/exception_helpers.hpp>


namespace {


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int &val) : value(val) {}
    void run(std::shared_ptr<const gko::CpuExecutor>) const override
    {
        value = -1;
    }
    void run(std::shared_ptr<const gko::GpuExecutor> gpu) const override
    {
        cudaGetDevice(&value);
    }
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = -2;
    }

    int &value;
};


class GpuExecutor : public ::testing::Test {
protected:
    GpuExecutor() : cpu(gko::CpuExecutor::create()), gpu(nullptr), gpu2(nullptr)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::GpuExecutor::get_num_devices(), 0);
        gpu = gko::GpuExecutor::create(0, cpu);
        gpu2 = gko::GpuExecutor::create(gko::GpuExecutor::get_num_devices() - 1,
                                        cpu);
    }

    void TearDown()
    {
        if (gpu != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(gpu->synchronize());
        }
    }

    std::shared_ptr<gko::CpuExecutor> cpu;
    std::shared_ptr<gko::GpuExecutor> gpu;
    std::shared_ptr<gko::GpuExecutor> gpu2;
};


TEST_F(GpuExecutor, MasterKnowsNumberOfDevices)
{
    int count = 0;
    cudaGetDeviceCount(&count);
    ASSERT_EQ(count, gko::GpuExecutor::get_num_devices());
}


TEST_F(GpuExecutor, AllocatesAndFreesMemory)
{
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = gpu->alloc<int>(2));
    ASSERT_NO_THROW(gpu->free(ptr));
}


TEST_F(GpuExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    int *ptr = nullptr;

    ASSERT_THROW(
        {
            ptr = gpu->alloc<int>(num_elems);
            gpu->synchronize();
        },
        gko::AllocationError);

    gpu->free(ptr);
}


__global__ void check_data(int *data)
{
    if (data[0] != 3 || data[1] != 8) {
        asm("trap;");
    }
}

TEST_F(GpuExecutor, CopiesDataToGpu)
{
    int orig[] = {3, 8};
    auto *copy = gpu->alloc<int>(2);

    gpu->copy_from(cpu.get(), 2, orig, copy);

    check_data<<<1, 1>>>(copy);
    ASSERT_NO_THROW(gpu->synchronize());
    gpu->free(copy);
}


__global__ void init_data(int *data)
{
    data[0] = 5;
    data[1] = 2;
}

TEST_F(GpuExecutor, CopiesDataFromGpu)
{
    int copy[2];
    auto orig = gpu->alloc<int>(2);
    init_data<<<1, 1>>>(orig);

    cpu->copy_from(gpu.get(), 2, orig, copy);

    EXPECT_EQ(5, copy[0]);
    ASSERT_EQ(2, copy[1]);
    gpu->free(orig);
}

/* Properly checks if it works only when multiple GPUs exist */
TEST_F(GpuExecutor, PreservesDeviceSettings)
{
    auto previous_device = gko::GpuExecutor::get_num_devices() - 1;
    cudaSetDevice(previous_device);
    auto orig = gpu->alloc<int>(2);
    int current_device;
    cudaGetDevice(&current_device);
    ASSERT_EQ(current_device, previous_device);

    gpu->free(orig);
    cudaGetDevice(&current_device);
    ASSERT_EQ(current_device, previous_device);
}

TEST_F(GpuExecutor, RunsOnProperDevice)
{
    int value = -1;
    gpu2->run(ExampleOperation(value));
    ASSERT_EQ(value, gpu2->get_device_id());
}

TEST_F(GpuExecutor, CopiesDataFromGpuToGpu)
{
    int copy[2];
    auto orig = gpu->alloc<int>(2);
    cudaSetDevice(0);
    init_data<<<1, 1>>>(orig);

    auto copy_gpu2 = gpu2->alloc<int>(2);
    gpu2->copy_from(gpu.get(), 2, orig, copy_gpu2);
    cpu->copy_from(gpu2.get(), 2, copy_gpu2, copy);

    EXPECT_EQ(5, copy[0]);
    ASSERT_EQ(2, copy[1]);
    gpu->free(copy_gpu2);
    gpu->free(orig);
}

TEST_F(GpuExecutor, Synchronizes)
{
    // Todo design a proper unit test once we support streams
    ASSERT_NO_THROW(gpu->synchronize());
}


}  // namespace
