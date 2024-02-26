// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/executor.hpp>


#include <exception>
#include <memory>
#include <type_traits>


#include <CL/sycl.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace {


class DpcppExecutor : public ::testing::Test {
protected:
    DpcppExecutor()
        : ref(gko::ReferenceExecutor::create()), dpcpp(nullptr), dpcpp2(nullptr)
    {}

    void SetUp()
    {
        if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
            dpcpp = gko::DpcppExecutor::create(0, ref, "gpu");
            if (gko::DpcppExecutor::get_num_devices("gpu") > 1) {
                dpcpp2 = gko::DpcppExecutor::create(1, ref, "gpu");
            }
        } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
            dpcpp = gko::DpcppExecutor::create(0, ref, "cpu");
            if (gko::DpcppExecutor::get_num_devices("cpu") > 1) {
                dpcpp2 = gko::DpcppExecutor::create(1, ref, "cpu");
            }
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    void TearDown()
    {
        // ensure that previous calls finished and didn't throw an error
        ASSERT_NO_THROW(dpcpp->synchronize());
        if (dpcpp2 != nullptr) {
            ASSERT_NO_THROW(dpcpp2->synchronize());
        }
    }

    std::shared_ptr<gko::Executor> ref{};
    std::shared_ptr<const gko::DpcppExecutor> dpcpp{};
    std::shared_ptr<const gko::DpcppExecutor> dpcpp2{};
};


TEST_F(DpcppExecutor, CanInstantiateTwoExecutorsOnOneDevice)
{
    auto dpcpp = gko::DpcppExecutor::create(0, ref);
    if (dpcpp2 != nullptr) {
        auto dpcpp2 = gko::DpcppExecutor::create(0, ref);
    }

    // We want automatic deinitialization to not create any error
}


TEST_F(DpcppExecutor, CanGetExecInfo)
{
    dpcpp = gko::DpcppExecutor::create(0, ref);

    ASSERT_TRUE(dpcpp->get_num_computing_units() > 0);
    ASSERT_TRUE(dpcpp->get_subgroup_sizes().size() > 0);
    ASSERT_TRUE(dpcpp->get_max_workitem_sizes().size() > 0);
    ASSERT_TRUE(dpcpp->get_max_workgroup_size() > 0);
    ASSERT_TRUE(dpcpp->get_max_subgroup_size() > 0);
}


TEST_F(DpcppExecutor, KnowsNumberOfDevicesOfTypeAll)
{
    auto count = sycl::device::get_devices(sycl::info::device_type::all).size();

    auto num_devices = gko::DpcppExecutor::get_num_devices("all");

    ASSERT_EQ(count, num_devices);
}


TEST_F(DpcppExecutor, KnowsNumberOfDevicesOfTypeCPU)
{
    auto count = sycl::device::get_devices(sycl::info::device_type::cpu).size();

    auto num_devices = gko::DpcppExecutor::get_num_devices("cpu");

    ASSERT_EQ(count, num_devices);
}


TEST_F(DpcppExecutor, KnowsNumberOfDevicesOfTypeGPU)
{
    auto count = sycl::device::get_devices(sycl::info::device_type::gpu).size();

    auto num_devices = gko::DpcppExecutor::get_num_devices("gpu");

    ASSERT_EQ(count, num_devices);
}


TEST_F(DpcppExecutor, KnowsNumberOfDevicesOfTypeAccelerator)
{
    auto count =
        sycl::device::get_devices(sycl::info::device_type::accelerator).size();

    auto num_devices = gko::DpcppExecutor::get_num_devices("accelerator");

    ASSERT_EQ(count, num_devices);
}


TEST_F(DpcppExecutor, AllocatesAndFreesMemory)
{
    int* ptr = nullptr;

    ASSERT_NO_THROW(ptr = dpcpp->alloc<int>(2));
    ASSERT_NO_THROW(dpcpp->free(ptr));
}


TEST_F(DpcppExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    int* ptr = nullptr;

    ASSERT_THROW(
        {
            ptr = dpcpp->alloc<int>(num_elems);
            dpcpp->synchronize();
        },
        gko::AllocationError);

    dpcpp->free(ptr);
}


void check_data(int* data, bool* result)
{
    *result = false;
    if (data[0] == 3 && data[1] == 8) {
        *result = true;
    }
}

TEST_F(DpcppExecutor, CopiesDataToCPU)
{
    int orig[] = {3, 8};
    auto* copy = dpcpp->alloc<int>(2);
    gko::array<bool> is_set(ref, 1);

    dpcpp->copy_from(ref, 2, orig, copy);

    is_set.set_executor(dpcpp);
    ASSERT_NO_THROW(dpcpp->synchronize());
    ASSERT_NO_THROW(dpcpp->get_queue()->submit([&](sycl::handler& cgh) {
        auto* is_set_ptr = is_set.get_data();
        cgh.single_task([=]() { check_data(copy, is_set_ptr); });
    }));
    is_set.set_executor(ref);
    ASSERT_EQ(*is_set.get_data(), true);
    ASSERT_NO_THROW(dpcpp->synchronize());
    dpcpp->free(copy);
}

void init_data(int* data)
{
    data[0] = 3;
    data[1] = 8;
}

TEST_F(DpcppExecutor, CopiesDataFromCPU)
{
    int copy[2];
    auto orig = dpcpp->alloc<int>(2);
    dpcpp->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.single_task([=]() { init_data(orig); });
    });

    ref->copy_from(dpcpp, 2, orig, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    dpcpp->free(orig);
}


TEST_F(DpcppExecutor, CopiesDataFromDpcppToDpcpp)
{
    if (dpcpp2 == nullptr) {
        GTEST_SKIP();
    }

    int copy[2];
    gko::array<bool> is_set(ref, 1);
    auto orig = dpcpp->alloc<int>(2);
    dpcpp->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.single_task([=]() { init_data(orig); });
    });

    auto copy_dpcpp2 = dpcpp2->alloc<int>(2);
    dpcpp2->copy_from(dpcpp, 2, orig, copy_dpcpp2);
    // Check that the data is really on GPU
    is_set.set_executor(dpcpp2);
    ASSERT_NO_THROW(dpcpp2->get_queue()->submit([&](sycl::handler& cgh) {
        auto* is_set_ptr = is_set.get_data();
        cgh.single_task([=]() { check_data(copy_dpcpp2, is_set_ptr); });
    }));
    is_set.set_executor(ref);
    ASSERT_EQ(*is_set.get_data(), true);

    // Put the results on OpenMP and run CPU side assertions
    ref->copy_from(dpcpp2, 2, copy_dpcpp2, copy);
    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    dpcpp2->free(copy_dpcpp2);
    dpcpp->free(orig);
}


TEST_F(DpcppExecutor, Synchronizes)
{
    // Todo design a proper unit test once we support streams
    ASSERT_NO_THROW(dpcpp->synchronize());
}


TEST_F(DpcppExecutor, FreeAfterKernel)
{
    size_t length = 10000;
    auto dpcpp =
        gko::DpcppExecutor::create(0, gko::ReferenceExecutor::create());
    {
        gko::array<float> x(dpcpp, length);
        gko::array<float> y(dpcpp, length);
        auto x_val = x.get_data();
        auto y_val = y.get_data();
        dpcpp->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>{length},
                             [=](sycl::id<1> i) { y_val[i] += x_val[i]; });
        });
    }
    // to ensure everything on queue is finished.
    dpcpp->synchronize();
}


}  // namespace
