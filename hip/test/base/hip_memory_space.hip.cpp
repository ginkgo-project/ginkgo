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

#include <ginkgo/core/base/memory_space.hpp>


#include <memory>
#include <type_traits>


#include <gtest/gtest.h>
#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace {


class HipMemorySpace : public ::testing::Test {
protected:
    HipMemorySpace()
        : omp(gko::HostMemorySpace::create()), hip(nullptr), hip2(nullptr)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::HipMemorySpace::get_num_devices(), 0);
        hip = gko::HipMemorySpace::create(0);
        hip2 = gko::HipMemorySpace::create(
            gko::HipMemorySpace::get_num_devices() - 1);
    }

    void TearDown()
    {
        if (hip != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

    std::shared_ptr<gko::MemorySpace> omp;
    std::shared_ptr<gko::HipMemorySpace> hip;
    std::shared_ptr<gko::HipMemorySpace> hip2;
};


TEST_F(HipMemorySpace, AllocatesAndFreesMemory)
{
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = hip->alloc<int>(2));
    ASSERT_NO_THROW(hip->free(ptr));
}


TEST_F(HipMemorySpace, FailsWhenOverallocating)
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
#if GINKGO_HIP_PLATFORM_HCC
        asm("s_trap 0x02;");
#else  // GINKGO_HIP_PLATFORM_NVCC
        asm("trap;");
#endif
    }
}

TEST_F(HipMemorySpace, CopiesDataToHip)
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

TEST_F(HipMemorySpace, CopiesDataFromHip)
{
    int copy[2];
    auto orig = hip->alloc<int>(2);
    hipLaunchKernelGGL((init_data), dim3(1), dim3(1), 0, 0, orig);

    omp->copy_from(hip.get(), 2, orig, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    hip->free(orig);
}


TEST_F(HipMemorySpace, CopiesDataFromHipToHip)
{
    int copy[2];
    auto orig = hip->alloc<int>(2);
    GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(0));
    hipLaunchKernelGGL((init_data), dim3(1), dim3(1), 0, 0, orig);

    auto copy_hip2 = hip2->alloc<int>(2);
    hip2->copy_from(hip.get(), 2, orig, copy_hip2);

    // Check that the data is really on GPU2 and ensure we did not cheat
    GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(hip2->get_device_id()));
    hipLaunchKernelGGL((check_data), dim3(1), dim3(1), 0, 0, copy_hip2);
    GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(0));
    // Put the results on OpenMP and run CPU side assertions
    omp->copy_from(hip2.get(), 2, copy_hip2, copy);
    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    hip->free(copy_hip2);
    hip->free(orig);
}


TEST_F(HipMemorySpace, Synchronizes)
{
    // Todo design a proper unit test once we support streams
    ASSERT_NO_THROW(hip->synchronize());
}


}  // namespace
