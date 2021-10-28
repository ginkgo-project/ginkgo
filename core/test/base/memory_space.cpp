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

#include <ginkgo/core/base/memory_space.hpp>


#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>


namespace {


using mem_space_ptr = std::shared_ptr<gko::MemorySpace>;


TEST(HostMemorySpace, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    mem_space_ptr host = gko::HostMemorySpace::create();
    int* ptr = nullptr;

    ASSERT_NO_THROW(ptr = host->alloc<int>(num_elems));
    ASSERT_NO_THROW(host->free(ptr));
}


TEST(HostMemorySpace, FreeAcceptsNullptr)
{
    mem_space_ptr host = gko::HostMemorySpace::create();
    ASSERT_NO_THROW(host->free(nullptr));
}


TEST(HostMemorySpace, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    mem_space_ptr host = gko::HostMemorySpace::create();
    int* ptr = nullptr;

    ASSERT_THROW(ptr = host->alloc<int>(num_elems), gko::AllocationError);

    host->free(ptr);
}


TEST(HostMemorySpace, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    mem_space_ptr host = gko::HostMemorySpace::create();
    int* copy = host->alloc<int>(num_elems);

    // user code is run on the HOST, so local variables are in HOST memory
    host->copy_from(host.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    host->free(copy);
}


TEST(CudaMemorySpace, KnowsItsDeviceId)
{
    auto cuda = gko::CudaMemorySpace::create(0);

    ASSERT_EQ(0, cuda->get_device_id());
}


TEST(CudaUVMSpace, KnowsItsDeviceId)
{
    auto cuda_uvm = gko::CudaUVMSpace::create(0);

    ASSERT_EQ(0, cuda_uvm->get_device_id());
}


TEST(HipMemorySpace, KnowsItsDeviceId)
{
    auto hip = gko::HipMemorySpace::create(0);

    ASSERT_EQ(0, hip->get_device_id());
}


TEST(MemorySpace, CanVerifyMemory)
{
    auto host = gko::HostMemorySpace::create();
    auto hip = gko::HipMemorySpace::create(0);
    auto cuda = gko::CudaMemorySpace::create(0);
    auto host2 = gko::HostMemorySpace::create();
    auto hip2 = gko::HipMemorySpace::create(0);
    auto cuda2 = gko::CudaMemorySpace::create(0);
    auto hip_1 = gko::HipMemorySpace::create(1);
    auto cuda_1 = gko::CudaMemorySpace::create(1);
    std::shared_ptr<gko::DpcppMemorySpace> host_dpcpp;
    std::shared_ptr<gko::DpcppMemorySpace> cpu_dpcpp;
    std::shared_ptr<gko::DpcppMemorySpace> gpu_dpcpp;
    std::shared_ptr<gko::DpcppMemorySpace> host_dpcpp_dup;
    std::shared_ptr<gko::DpcppMemorySpace> cpu_dpcpp_dup;
    std::shared_ptr<gko::DpcppMemorySpace> gpu_dpcpp_dup;
    if (gko::DpcppMemorySpace::get_num_devices("host")) {
        host_dpcpp = gko::DpcppMemorySpace::create(0, "host");
        host_dpcpp_dup = gko::DpcppMemorySpace::create(0, "host");
    }
    if (gko::DpcppMemorySpace::get_num_devices("cpu")) {
        cpu_dpcpp = gko::DpcppMemorySpace::create(0, "cpu");
        cpu_dpcpp_dup = gko::DpcppMemorySpace::create(0, "cpu");
    }
    if (gko::DpcppMemorySpace::get_num_devices("gpu")) {
        gpu_dpcpp = gko::DpcppMemorySpace::create(0, "gpu");
        gpu_dpcpp_dup = gko::DpcppMemorySpace::create(0, "gpu");
    }

    ASSERT_EQ(true, host->memory_accessible(host));
    ASSERT_EQ(false, host->memory_accessible(hip));
    ASSERT_EQ(false, hip->memory_accessible(host));
    ASSERT_EQ(false, cuda->memory_accessible(host));
    ASSERT_EQ(false, host->memory_accessible(cuda));
    ASSERT_EQ(false, cuda->memory_accessible(host));
    if (gko::DpcppMemorySpace::get_num_devices("host")) {
        ASSERT_EQ(true, host_dpcpp->memory_accessible(host));
        ASSERT_EQ(true, host->memory_accessible(host_dpcpp));
        ASSERT_EQ(true, host_dpcpp->memory_accessible(host_dpcpp_dup));
        ASSERT_EQ(true, host_dpcpp_dup->memory_accessible(host_dpcpp));
    }
    if (gko::DpcppMemorySpace::get_num_devices("cpu")) {
        ASSERT_EQ(true, cpu_dpcpp->memory_accessible(host));
        ASSERT_EQ(true, host->memory_accessible(cpu_dpcpp));
        ASSERT_EQ(true, cpu_dpcpp->memory_accessible(cpu_dpcpp_dup));
        ASSERT_EQ(true, cpu_dpcpp_dup->memory_accessible(cpu_dpcpp));
    }
    if (gko::DpcppMemorySpace::get_num_devices("gpu")) {
        ASSERT_EQ(false, gpu_dpcpp->memory_accessible(host));
        ASSERT_EQ(false, host->memory_accessible(gpu_dpcpp));
        ASSERT_EQ(false, gpu_dpcpp->memory_accessible(gpu_dpcpp_dup));
        ASSERT_EQ(false, gpu_dpcpp_dup->memory_accessible(gpu_dpcpp));
    }
#if GINKGO_HIP_PLATFORM_NVCC
    ASSERT_EQ(true, hip->memory_accessible(cuda));
    ASSERT_EQ(true, cuda->memory_accessible(hip));
    ASSERT_EQ(true, hip_1->memory_accessible(cuda_1));
    ASSERT_EQ(true, cuda_1->memory_accessible(hip_1));
#else
    ASSERT_EQ(false, hip->memory_accessible(cuda));
    ASSERT_EQ(false, cuda->memory_accessible(hip));
    ASSERT_EQ(false, hip_1->memory_accessible(cuda_1));
    ASSERT_EQ(false, cuda_1->memory_accessible(hip_1));
#endif
    ASSERT_EQ(true, host->memory_accessible(host2));
    ASSERT_EQ(true, hip->memory_accessible(hip2));
    ASSERT_EQ(true, cuda->memory_accessible(cuda2));
    ASSERT_EQ(false, hip->memory_accessible(hip_1));
    ASSERT_EQ(false, cuda->memory_accessible(hip_1));
    ASSERT_EQ(false, cuda->memory_accessible(cuda_1));
    ASSERT_EQ(false, hip->memory_accessible(cuda_1));
}


template <typename T>
struct mock_free : T {
    /**
     * @internal Due to a bug with gcc 5.3, the constructor needs to be called
     * with `()` operator instead of `{}`.
     */
    template <typename... Params>
    mock_free(Params&&... params) : T(std::forward<Params>(params)...)
    {}

    void raw_free(void* ptr) const noexcept override
    {
        called_free = true;
        T::raw_free(ptr);
    }

    mutable bool called_free{false};
};


TEST(MemorySpaceDeleter, DeletesObject)
{
    auto host = std::make_shared<mock_free<gko::HostMemorySpace>>();
    auto x = host->alloc<int>(5);

    gko::memory_space_deleter<int>{host}(x);

    ASSERT_TRUE(host->called_free);
}


TEST(MemorySpaceDeleter, AvoidsDeletionForNullMemorySpace)
{
    int x[5];
    ASSERT_NO_THROW(gko::memory_space_deleter<int>{nullptr}(x));
}


}  // namespace
