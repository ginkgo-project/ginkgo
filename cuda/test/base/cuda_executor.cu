// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/executor.hpp>


#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/stream.hpp>

#include "common/cuda_hip/base/executor.hpp.inc"
#include "cuda/base/scoped_device_id.hpp"
#include "cuda/test/utils.hpp"


namespace {


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int& val) : value(val) {}

    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        value = -1;
    }

    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = -2;
    }

    void run(std::shared_ptr<const gko::HipExecutor>) const override
    {
        value = -3;
    }

    void run(std::shared_ptr<const gko::DpcppExecutor>) const override
    {
        value = -4;
    }

    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        cudaGetDevice(&value);
    }

    int& value;
};


class CudaExecutor : public ::testing::Test {
protected:
    CudaExecutor()
        :
#ifdef GKO_TEST_NONDEFAULT_STREAM
          stream(0),
          other_stream(gko::CudaExecutor::get_num_devices() - 1),
#endif
          ref(gko::ReferenceExecutor::create()),
          cuda(nullptr),
          cuda2(nullptr),
          cuda3(nullptr)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
#ifdef GKO_TEST_NONDEFAULT_STREAM
        cuda = gko::CudaExecutor::create(
            0, ref, std::make_shared<gko::CudaAllocator>(), stream.get());
        cuda2 = gko::CudaExecutor::create(
            gko::CudaExecutor::get_num_devices() - 1, ref,
            std::make_shared<gko::CudaAllocator>(), other_stream.get());
        cuda3 = gko::CudaExecutor::create(
            0, ref, std::make_shared<gko::CudaUnifiedAllocator>(0),
            stream.get());
#else
        cuda = gko::CudaExecutor::create(0, ref);
        cuda2 = gko::CudaExecutor::create(
            gko::CudaExecutor::get_num_devices() - 1, ref);
        cuda3 = gko::CudaExecutor::create(
            0, ref, std::make_shared<gko::CudaUnifiedAllocator>(0));
#endif
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

#ifdef GKO_TEST_NONDEFAULT_STREAM
    gko::cuda_stream stream;
    gko::cuda_stream other_stream;
#endif
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::shared_ptr<gko::CudaExecutor> cuda2;
    std::shared_ptr<gko::CudaExecutor> cuda3;
};


TEST_F(CudaExecutor, CanInstantiateTwoExecutorsOnOneDevice)
{
    auto cuda = gko::CudaExecutor::create(0, ref);
    auto cuda2 = gko::CudaExecutor::create(0, ref);

    // We want automatic deinitialization to not create any error
}


TEST_F(CudaExecutor, MasterKnowsNumberOfDevices)
{
    int count = 0;
    cudaGetDeviceCount(&count);

    auto num_devices = gko::CudaExecutor::get_num_devices();

    ASSERT_EQ(count, num_devices);
}


TEST_F(CudaExecutor, AllocatesAndFreesMemory)
{
    int* ptr = nullptr;

    ASSERT_NO_THROW(ptr = cuda->alloc<int>(2));
    ASSERT_NO_THROW(cuda->free(ptr));
}


TEST_F(CudaExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    int* ptr = nullptr;

    ASSERT_THROW(
        {
            ptr = cuda->alloc<int>(num_elems);
            cuda->synchronize();
        },
        gko::AllocationError);

    cuda->free(ptr);
}


__global__ void check_data(int* data)
{
    if (data[0] != 3 || data[1] != 8) {
        asm("trap;");
    }
}


TEST_F(CudaExecutor, CopiesDataToCuda)
{
    int orig[] = {3, 8};
    auto* copy = cuda->alloc<int>(2);

    cuda->copy_from(ref, 2, orig, copy);

    check_data<<<1, 1, 0, cuda->get_stream()>>>(copy);
    ASSERT_NO_THROW(cuda->synchronize());
    cuda->free(copy);
}


__global__ void check_data2(int* data)
{
    if (data[0] != 4 || data[1] != 8) {
        asm("trap;");
    }
}


TEST_F(CudaExecutor, CanAllocateOnUnifiedMemory)
{
    int orig[] = {3, 8};
    auto* copy = cuda3->alloc<int>(2);

    cuda3->copy_from(ref, 2, orig, copy);

    check_data<<<1, 1, 0, cuda3->get_stream()>>>(copy);
    ASSERT_NO_THROW(cuda3->synchronize());
    copy[0] = 4;
    check_data2<<<1, 1, 0, cuda3->get_stream()>>>(copy);
    cuda3->free(copy);
}


__global__ void init_data(int* data)
{
    data[0] = 3;
    data[1] = 8;
}

TEST_F(CudaExecutor, CopiesDataFromCuda)
{
    int copy[2];
    auto orig = cuda->alloc<int>(2);
    init_data<<<1, 1, 0, cuda->get_stream()>>>(orig);

    ref->copy_from(cuda, 2, orig, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    cuda->free(orig);
}


/* Properly checks if it works only when multiple GPUs exist */
TEST_F(CudaExecutor, PreservesDeviceSettings)
{
    auto previous_device = gko::CudaExecutor::get_num_devices() - 1;
    GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(previous_device));
    auto orig = cuda->alloc<int>(2);
    int current_device;
    GKO_ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&current_device));
    ASSERT_EQ(current_device, previous_device);

    cuda->free(orig);
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


TEST_F(CudaExecutor, CopiesDataFromCudaToCuda)
{
    int copy[2];
    auto orig = cuda->alloc<int>(2);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(0));
    init_data<<<1, 1, 0, cuda->get_stream()>>>(orig);

    auto copy_cuda2 = cuda2->alloc<int>(2);
    cuda2->copy_from(cuda, 2, orig, copy_cuda2);

    // Check that the data is really on GPU2 and ensure we did not cheat
    int value = -1;
    GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(cuda2->get_device_id()));
    check_data<<<1, 1, 0, cuda2->get_stream()>>>(copy_cuda2);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(0));
    cuda2->run(ExampleOperation(value));
    ASSERT_EQ(value, cuda2->get_device_id());
    // Put the results on OpenMP and run CPU side assertions
    ref->copy_from(cuda2, 2, copy_cuda2, copy);
    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    cuda2->free(copy_cuda2);
    cuda->free(orig);
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
