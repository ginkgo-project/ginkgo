// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// force-top: on
// prevent compilation failure related to disappearing assert(...) statements
#include <hip/hip_runtime.h>
// force-top: off


#include <ginkgo/core/base/executor.hpp>


#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/executor.hpp.inc"
#include "hip/base/scoped_device_id.hip.hpp"
#include "hip/test/utils.hip.hpp"


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

    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        value = -3;
    }

    void run(std::shared_ptr<const gko::DpcppExecutor>) const override
    {
        value = -4;
    }

    void run(std::shared_ptr<const gko::HipExecutor>) const override
    {
        hipGetDevice(&value);
    }

    int& value;
};


class HipExecutor : public ::testing::Test {
protected:
    HipExecutor()
        :
#ifdef GKO_TEST_NONDEFAULT_STREAM
          stream(0),
          other_stream(gko::HipExecutor::get_num_devices() - 1),
#endif
          ref(gko::ReferenceExecutor::create()),
          hip(nullptr),
          hip2(nullptr),
          hip3(nullptr)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
#ifdef GKO_TEST_NONDEFAULT_STREAM
        hip = gko::HipExecutor::create(
            0, ref, std::make_shared<gko::HipAllocator>(), stream.get());
        hip2 = gko::HipExecutor::create(
            gko::HipExecutor::get_num_devices() - 1, ref,
            std::make_shared<gko::HipAllocator>(), other_stream.get());
        hip3 = gko::HipExecutor::create(
            0, ref, std::make_shared<gko::HipUnifiedAllocator>(0),
            stream.get());
#else
        hip = gko::HipExecutor::create(0, ref);
        hip2 = gko::HipExecutor::create(gko::HipExecutor::get_num_devices() - 1,
                                        ref);
        hip3 = gko::HipExecutor::create(
            0, ref, std::make_shared<gko::HipUnifiedAllocator>(0));
#endif
    }

    void TearDown()
    {
        if (hip != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

#ifdef GKO_TEST_NONDEFAULT_STREAM
    gko::hip_stream stream;
    gko::hip_stream other_stream;
#endif
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> hip;
    std::shared_ptr<gko::HipExecutor> hip2;
    std::shared_ptr<gko::HipExecutor> hip3;
};


TEST_F(HipExecutor, CanInstantiateTwoExecutorsOnOneDevice)
{
    auto hip = gko::HipExecutor::create(0, ref);
    auto hip2 = gko::HipExecutor::create(0, ref);

    // We want automatic deinitialization to not create any error
}


TEST_F(HipExecutor, MasterKnowsNumberOfDevices)
{
    int count = 0;
    hipGetDeviceCount(&count);

    auto num_devices = gko::HipExecutor::get_num_devices();

    ASSERT_EQ(count, num_devices);
}


TEST_F(HipExecutor, AllocatesAndFreesMemory)
{
    int* ptr = nullptr;

    ASSERT_NO_THROW(ptr = hip->alloc<int>(2));
    ASSERT_NO_THROW(hip->free(ptr));
}


TEST_F(HipExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    int* ptr = nullptr;

    ASSERT_THROW(
        {
            ptr = hip->alloc<int>(num_elems);
            hip->synchronize();
        },
        gko::AllocationError);

    hip->free(ptr);
}


__global__ void check_data(int* data)
{
    if (data[0] != 3 || data[1] != 8) {
#if GINKGO_HIP_PLATFORM_HCC
        asm("s_trap 0x02;");
#else  // GINKGO_HIP_PLATFORM_NVCC
        asm("trap;");
#endif
    }
}

TEST_F(HipExecutor, CopiesDataToHip)
{
    int orig[] = {3, 8};
    auto* copy = hip->alloc<int>(2);

    hip->copy_from(ref, 2, orig, copy);

    check_data<<<1, 1, 0, hip->get_stream()>>>(copy);
    ASSERT_NO_THROW(hip->synchronize());
    hip->free(copy);
}


__global__ void check_data2(int* data)
{
    if (data[0] != 4 || data[1] != 8) {
#if GINKGO_HIP_PLATFORM_HCC
        asm("s_trap 0x02;");
#else  // GINKGO_HIP_PLATFORM_NVCC
        asm("trap;");
#endif
    }
}


#if GINKGO_HIP_PLATFORM_NVCC


TEST_F(HipExecutor, CanAllocateOnUnifiedMemory)
{
    int orig[] = {3, 8};
    auto* copy = hip3->alloc<int>(2);

    hip3->copy_from(ref, 2, orig, copy);

    check_data<<<1, 1, 0, hip3->get_stream()>>>(copy);
    ASSERT_NO_THROW(hip3->synchronize());
    copy[0] = 4;
    check_data2<<<1, 1, 0, hip3->get_stream()>>>(copy);
    hip3->free(copy);
}


#endif


__global__ void init_data(int* data)
{
    data[0] = 3;
    data[1] = 8;
}

TEST_F(HipExecutor, CopiesDataFromHip)
{
    int copy[2];
    auto orig = hip->alloc<int>(2);
    init_data<<<1, 1, 0, hip->get_stream()>>>(orig);

    ref->copy_from(hip, 2, orig, copy);

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
    init_data<<<1, 1, 0, hip->get_stream()>>>(orig);

    auto copy_hip2 = hip2->alloc<int>(2);
    hip2->copy_from(hip, 2, orig, copy_hip2);

    // Check that the data is really on GPU2 and ensure we did not cheat
    int value = -1;
    GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(hip2->get_device_id()));
    check_data<<<1, 1, 0, hip2->get_stream()>>>(copy_hip2);
    GKO_ASSERT_NO_HIP_ERRORS(hipSetDevice(0));
    hip2->run(ExampleOperation(value));
    ASSERT_EQ(value, hip2->get_device_id());
    // Put the results on OpenMP and run CPU side assertions
    ref->copy_from(hip2, 2, copy_hip2, copy);
    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    hip2->free(copy_hip2);
    hip->free(orig);
}


TEST_F(HipExecutor, Synchronizes)
{
    // Todo design a proper unit test once we support streams
    ASSERT_NO_THROW(hip->synchronize());
}


TEST_F(HipExecutor, ExecInfoSetsCorrectProperties)
{
    auto dev_id = hip->get_device_id();
    auto num_sm = 0;
    auto major = 0;
    auto minor = 0;
    auto max_threads_per_block = 0;
    auto warp_size = 0;
    GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
        &num_sm, hipDeviceAttributeMultiprocessorCount, dev_id));
    GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
        &major, hipDeviceAttributeComputeCapabilityMajor, dev_id));
    GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
        &minor, hipDeviceAttributeComputeCapabilityMinor, dev_id));
    GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
        &max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock, dev_id));
    GKO_ASSERT_NO_HIP_ERRORS(
        hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, dev_id));
#if GINKGO_HIP_PLATFORM_NVCC
    auto num_cores = convert_sm_ver_to_cores(major, minor);
#else
    auto num_cores = warp_size * 4;
#endif

    ASSERT_EQ(hip->get_major_version(), major);
    ASSERT_EQ(hip->get_minor_version(), minor);
    ASSERT_EQ(hip->get_num_multiprocessor(), num_sm);
    ASSERT_EQ(hip->get_warp_size(), warp_size);
    ASSERT_EQ(hip->get_num_warps(), num_sm * (num_cores / warp_size));
    ASSERT_EQ(hip->get_num_warps_per_sm(), num_cores / warp_size);
}


}  // namespace
