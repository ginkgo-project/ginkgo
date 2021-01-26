/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
        : omp(gko::OmpExecutor::create()), dpcpp(nullptr), dpcpp2(nullptr)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::DpcppExecutor::get_num_devices("cpu"), 0);
        dpcpp = gko::DpcppExecutor::create(0, omp, "cpu");
        if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
            dpcpp2 = gko::DpcppExecutor::create(0, omp, "gpu");
        }
    }

    void TearDown()
    {
        if (dpcpp != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(dpcpp->synchronize());
        }
    }

    std::shared_ptr<gko::Executor> omp;
    std::shared_ptr<const gko::DpcppExecutor> dpcpp;
    std::shared_ptr<const gko::DpcppExecutor> dpcpp2;
};


TEST_F(DpcppExecutor, CanInstantiateTwoExecutorsOnOneDevice)
{
    auto dpcpp = gko::DpcppExecutor::create(0, omp);
    auto dpcpp2 = gko::DpcppExecutor::create(0, omp);

    // We want automatic deinitialization to not create any error
}


TEST_F(DpcppExecutor, CanGetExecInfo)
{
    dpcpp = gko::DpcppExecutor::create(0, omp);

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


TEST_F(DpcppExecutor, AllocatesAndFreesMemoryOnCPU)
{
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = dpcpp->alloc<int>(2));
    ASSERT_NO_THROW(dpcpp->free(ptr));
}


TEST_F(DpcppExecutor, AllocatesAndFreesMemoryOnGPU)
{
    if (!dpcpp2) {
        GTEST_SKIP() << "No DPC++ compatible GPU.";
    }
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = dpcpp2->alloc<int>(2));
    ASSERT_NO_THROW(dpcpp2->free(ptr));
}


TEST_F(DpcppExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    int *ptr = nullptr;

    ASSERT_THROW(
        {
            ptr = dpcpp->alloc<int>(num_elems);
            dpcpp->synchronize();
        },
        gko::AllocationError);

    dpcpp->free(ptr);
}


void check_data(int *data, bool *result)
{
    *result = false;
    if (data[0] == 3 && data[1] == 8) {
        *result = true;
    }
}

TEST_F(DpcppExecutor, CopiesDataToCPU)
{
    int orig[] = {3, 8};
    auto *copy = dpcpp->alloc<int>(2);
    gko::Array<bool> is_set(omp, 1);

    dpcpp->copy_from(omp.get(), 2, orig, copy);

    is_set.set_executor(dpcpp);
    ASSERT_NO_THROW(dpcpp->synchronize());
    ASSERT_NO_THROW(dpcpp->get_queue()->submit([&](sycl::handler &cgh) {
        auto *is_set_ptr = is_set.get_data();
        cgh.single_task([=]() { check_data(copy, is_set_ptr); });
    }));
    is_set.set_executor(omp);
    ASSERT_EQ(*is_set.get_data(), true);
    ASSERT_NO_THROW(dpcpp->synchronize());
    dpcpp->free(copy);
}


TEST_F(DpcppExecutor, CopiesDataToGPU)
{
    if (!dpcpp2) {
        GTEST_SKIP() << "No DPC++ compatible GPU.";
    }
    int orig[] = {3, 8};
    auto *copy = dpcpp2->alloc<int>(2);
    gko::Array<bool> is_set(omp, 1);

    dpcpp2->copy_from(omp.get(), 2, orig, copy);

    is_set.set_executor(dpcpp2);
    ASSERT_NO_THROW(dpcpp2->get_queue()->submit([&](sycl::handler &cgh) {
        auto *is_set_ptr = is_set.get_data();
        cgh.single_task([=]() { check_data(copy, is_set_ptr); });
    }));
    is_set.set_executor(omp);
    ASSERT_EQ(*is_set.get_data(), true);
    ASSERT_NO_THROW(dpcpp2->synchronize());
    dpcpp2->free(copy);
}


void init_data(int *data)
{
    data[0] = 3;
    data[1] = 8;
}

TEST_F(DpcppExecutor, CopiesDataFromCPU)
{
    int copy[2];
    auto orig = dpcpp->alloc<int>(2);
    dpcpp->get_queue()->submit([&](sycl::handler &cgh) {
        cgh.single_task([=]() { init_data(orig); });
    });

    omp->copy_from(dpcpp.get(), 2, orig, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    dpcpp->free(orig);
}


TEST_F(DpcppExecutor, CopiesDataFromGPU)
{
    if (!dpcpp2) {
        GTEST_SKIP() << "No DPC++ compatible GPU.";
    }
    int copy[2];
    auto orig = dpcpp2->alloc<int>(2);
    dpcpp2->get_queue()->submit([&](sycl::handler &cgh) {
        cgh.single_task([=]() { init_data(orig); });
    });

    omp->copy_from(dpcpp2.get(), 2, orig, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    dpcpp2->free(orig);
}


TEST_F(DpcppExecutor, CopiesDataFromDpcppToDpcpp)
{
    if (!dpcpp2) {
        GTEST_SKIP() << "No DPC++ compatible GPU.";
    }
    int copy[2];
    gko::Array<bool> is_set(omp, 1);
    auto orig = dpcpp->alloc<int>(2);
    dpcpp->get_queue()->submit([&](sycl::handler &cgh) {
        cgh.single_task([=]() { init_data(orig); });
    });

    auto copy_dpcpp2 = dpcpp2->alloc<int>(2);
    dpcpp2->copy_from(dpcpp.get(), 2, orig, copy_dpcpp2);
    // Check that the data is really on GPU
    is_set.set_executor(dpcpp2);
    ASSERT_NO_THROW(dpcpp2->get_queue()->submit([&](sycl::handler &cgh) {
        auto *is_set_ptr = is_set.get_data();
        cgh.single_task([=]() { check_data(copy_dpcpp2, is_set_ptr); });
    }));
    is_set.set_executor(omp);
    ASSERT_EQ(*is_set.get_data(), true);

    // Put the results on OpenMP and run CPU side assertions
    omp->copy_from(dpcpp2.get(), 2, copy_dpcpp2, copy);
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


}  // namespace
