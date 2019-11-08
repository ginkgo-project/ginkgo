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
#include <ginkgo/core/base/memory_space.hpp>


#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace {


class CudaMemorySpace : public ::testing::Test {
protected:
    CudaMemorySpace() : cuda(nullptr), cuda2(nullptr) {}

    void SetUp()
    {
        omp = gko::HostMemorySpace::create();
        cuda = gko::CudaMemorySpace::create(0);
        cuda2 = gko::CudaMemorySpace::create(
            gko::CudaMemorySpace::get_num_devices() - 1);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    std::shared_ptr<gko::HostMemorySpace> omp;
    std::shared_ptr<gko::CudaMemorySpace> cuda;
    std::shared_ptr<gko::CudaMemorySpace> cuda2;
};


TEST_F(CudaMemorySpace, AllocatesAndFreesMemory)
{
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = cuda->alloc<int>(2));
    ASSERT_NO_THROW(cuda->free(ptr));
}


TEST_F(CudaMemorySpace, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    int *ptr = nullptr;

    ASSERT_THROW(
        {
            ptr = cuda->alloc<int>(num_elems);
            cuda->synchronize();
        },
        gko::AllocationError);

    cuda->free(ptr);
}


__global__ void check_data(int *data)
{
    if (data[0] != 3 || data[1] != 8) {
        asm("trap;");
    }
}


TEST_F(CudaMemorySpace, CopiesDataToCuda)
{
    int orig[] = {3, 8};
    auto *copy = cuda->alloc<int>(2);

    cuda->copy_from(omp.get(), 2, orig, copy);

    check_data<<<1, 1>>>(copy);
    ASSERT_NO_THROW(cuda->synchronize());
    cuda->free(copy);
}


__global__ void init_data(int *data)
{
    data[0] = 3;
    data[1] = 8;
}

TEST_F(CudaMemorySpace, CopiesDataFromCuda)
{
    int copy[2];
    auto orig = cuda->alloc<int>(2);
    init_data<<<1, 1>>>(orig);

    omp->copy_from(cuda.get(), 2, orig, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    cuda->free(orig);
}


TEST_F(CudaMemorySpace, CopiesDataFromCudaToCuda)
{
    int copy[2];
    auto orig = cuda->alloc<int>(2);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(0));
    init_data<<<1, 1>>>(orig);

    auto copy_cuda2 = cuda2->alloc<int>(2);
    cuda2->copy_from(cuda.get(), 2, orig, copy_cuda2);

    // Check that the data is really on GPU2 and ensure we did not cheat
    GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(cuda2->get_device_id()));
    check_data<<<1, 1>>>(copy_cuda2);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(0));

    omp->copy_from(cuda2.get(), 2, copy_cuda2, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    cuda->free(copy_cuda2);
    cuda->free(orig);
}


}  // namespace
