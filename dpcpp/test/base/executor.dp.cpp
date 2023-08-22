/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


class SyclExecutor : public ::testing::Test {
protected:
    SyclExecutor()
        : ref(gko::ReferenceExecutor::create()), sycl(nullptr), sycl2(nullptr)
    {}

    void SetUp()
    {
        if (gko::SyclExecutor::get_num_devices("gpu") > 0) {
            sycl = gko::SyclExecutor::create(0, ref, "gpu");
            if (gko::SyclExecutor::get_num_devices("gpu") > 1) {
                sycl2 = gko::SyclExecutor::create(1, ref, "gpu");
            }
        } else if (gko::SyclExecutor::get_num_devices("cpu") > 0) {
            sycl = gko::SyclExecutor::create(0, ref, "cpu");
            if (gko::SyclExecutor::get_num_devices("cpu") > 1) {
                sycl2 = gko::SyclExecutor::create(1, ref, "cpu");
            }
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    void TearDown()
    {
        // ensure that previous calls finished and didn't throw an error
        ASSERT_NO_THROW(sycl->synchronize());
        if (sycl2 != nullptr) {
            ASSERT_NO_THROW(sycl2->synchronize());
        }
    }

    std::shared_ptr<gko::Executor> ref{};
    std::shared_ptr<const gko::SyclExecutor> sycl{};
    std::shared_ptr<const gko::SyclExecutor> sycl2{};
};


TEST_F(SyclExecutor, CanInstantiateTwoExecutorsOnOneDevice)
{
    auto sycl = gko::SyclExecutor::create(0, ref);
    if (sycl2 != nullptr) {
        auto sycl2 = gko::SyclExecutor::create(0, ref);
    }

    // We want automatic deinitialization to not create any error
}


TEST_F(SyclExecutor, CanGetExecInfo)
{
    sycl = gko::SyclExecutor::create(0, ref);

    ASSERT_TRUE(sycl->get_num_computing_units() > 0);
    ASSERT_TRUE(sycl->get_subgroup_sizes().size() > 0);
    ASSERT_TRUE(sycl->get_max_workitem_sizes().size() > 0);
    ASSERT_TRUE(sycl->get_max_workgroup_size() > 0);
    ASSERT_TRUE(sycl->get_max_subgroup_size() > 0);
}


TEST_F(SyclExecutor, KnowsNumberOfDevicesOfTypeAll)
{
    auto count = sycl::device::get_devices(sycl::info::device_type::all).size();

    auto num_devices = gko::SyclExecutor::get_num_devices("all");

    ASSERT_EQ(count, num_devices);
}


TEST_F(SyclExecutor, KnowsNumberOfDevicesOfTypeCPU)
{
    auto count = sycl::device::get_devices(sycl::info::device_type::cpu).size();

    auto num_devices = gko::SyclExecutor::get_num_devices("cpu");

    ASSERT_EQ(count, num_devices);
}


TEST_F(SyclExecutor, KnowsNumberOfDevicesOfTypeGPU)
{
    auto count = sycl::device::get_devices(sycl::info::device_type::gpu).size();

    auto num_devices = gko::SyclExecutor::get_num_devices("gpu");

    ASSERT_EQ(count, num_devices);
}


TEST_F(SyclExecutor, KnowsNumberOfDevicesOfTypeAccelerator)
{
    auto count =
        sycl::device::get_devices(sycl::info::device_type::accelerator).size();

    auto num_devices = gko::SyclExecutor::get_num_devices("accelerator");

    ASSERT_EQ(count, num_devices);
}


TEST_F(SyclExecutor, AllocatesAndFreesMemory)
{
    int* ptr = nullptr;

    ASSERT_NO_THROW(ptr = sycl->alloc<int>(2));
    ASSERT_NO_THROW(sycl->free(ptr));
}


TEST_F(SyclExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    int* ptr = nullptr;

    ASSERT_THROW(
        {
            ptr = sycl->alloc<int>(num_elems);
            sycl->synchronize();
        },
        gko::AllocationError);

    sycl->free(ptr);
}


void check_data(int* data, bool* result)
{
    *result = false;
    if (data[0] == 3 && data[1] == 8) {
        *result = true;
    }
}

TEST_F(SyclExecutor, CopiesDataToCPU)
{
    int orig[] = {3, 8};
    auto* copy = sycl->alloc<int>(2);
    gko::array<bool> is_set(ref, 1);

    sycl->copy_from(ref, 2, orig, copy);

    is_set.set_executor(sycl);
    ASSERT_NO_THROW(sycl->synchronize());
    ASSERT_NO_THROW(sycl->get_queue()->submit([&](sycl::handler& cgh) {
        auto* is_set_ptr = is_set.get_data();
        cgh.single_task([=]() { check_data(copy, is_set_ptr); });
    }));
    is_set.set_executor(ref);
    ASSERT_EQ(*is_set.get_data(), true);
    ASSERT_NO_THROW(sycl->synchronize());
    sycl->free(copy);
}

void init_data(int* data)
{
    data[0] = 3;
    data[1] = 8;
}

TEST_F(SyclExecutor, CopiesDataFromCPU)
{
    int copy[2];
    auto orig = sycl->alloc<int>(2);
    sycl->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.single_task([=]() { init_data(orig); });
    });

    ref->copy_from(sycl, 2, orig, copy);

    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    sycl->free(orig);
}


TEST_F(SyclExecutor, CopiesDataFromSyclToSycl)
{
    if (sycl2 == nullptr) {
        GTEST_SKIP();
    }

    int copy[2];
    gko::array<bool> is_set(ref, 1);
    auto orig = sycl->alloc<int>(2);
    sycl->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.single_task([=]() { init_data(orig); });
    });

    auto copy_sycl2 = sycl2->alloc<int>(2);
    sycl2->copy_from(sycl, 2, orig, copy_sycl2);
    // Check that the data is really on GPU
    is_set.set_executor(sycl2);
    ASSERT_NO_THROW(sycl2->get_queue()->submit([&](sycl::handler& cgh) {
        auto* is_set_ptr = is_set.get_data();
        cgh.single_task([=]() { check_data(copy_sycl2, is_set_ptr); });
    }));
    is_set.set_executor(ref);
    ASSERT_EQ(*is_set.get_data(), true);

    // Put the results on OpenMP and run CPU side assertions
    ref->copy_from(sycl2, 2, copy_sycl2, copy);
    EXPECT_EQ(3, copy[0]);
    ASSERT_EQ(8, copy[1]);
    sycl2->free(copy_sycl2);
    sycl->free(orig);
}


TEST_F(SyclExecutor, Synchronizes)
{
    // Todo design a proper unit test once we support streams
    ASSERT_NO_THROW(sycl->synchronize());
}


TEST_F(SyclExecutor, FreeAfterKernel)
{
    size_t length = 10000;
    auto sycl = gko::SyclExecutor::create(0, gko::ReferenceExecutor::create());
    {
        gko::array<float> x(sycl, length);
        gko::array<float> y(sycl, length);
        auto x_val = x.get_data();
        auto y_val = y.get_data();
        sycl->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>{length},
                             [=](sycl::id<1> i) { y_val[i] += x_val[i]; });
        });
    }
    // to ensure everything on queue is finished.
    sycl->synchronize();
}


}  // namespace
