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

#include <ginkgo/core/base/executor.hpp>


#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>


namespace {


using exec_ptr = std::shared_ptr<gko::Executor>;


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int &val) : value(val) {}
    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        value = 1;
    }
    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        value = 2;
    }
    void run(std::shared_ptr<const gko::HipExecutor>) const override
    {
        value = 3;
    }
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = 4;
    }

    int &value;
};


TEST(OmpExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr omp = gko::OmpExecutor::create();

    omp->run(ExampleOperation(value));
    ASSERT_EQ(1, value);
}


TEST(OmpExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    exec_ptr omp = gko::OmpExecutor::create();

    omp->run(omp_lambda, cuda_lambda, hip_lambda);
    ASSERT_EQ(1, value);
}


TEST(OmpExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    exec_ptr omp = gko::OmpExecutor::create();
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = omp->alloc<int>(num_elems));
    ASSERT_NO_THROW(omp->free(ptr));
}


TEST(OmpExecutor, FreeAcceptsNullptr)
{
    exec_ptr omp = gko::OmpExecutor::create();
    ASSERT_NO_THROW(omp->free(nullptr));
}


TEST(OmpExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    exec_ptr omp = gko::OmpExecutor::create();
    int *ptr = nullptr;

    ASSERT_THROW(ptr = omp->alloc<int>(num_elems), gko::AllocationError);

    omp->free(ptr);
}


TEST(OmpExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr omp = gko::OmpExecutor::create();
    int *copy = omp->alloc<int>(num_elems);

    // user code is run on the OMP, so local variables are in OMP memory
    omp->copy(num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    omp->free(copy);
}


TEST(OmpExecutor, IsItsOwnMaster)
{
    exec_ptr omp = gko::OmpExecutor::create();

    ASSERT_EQ(omp, omp->get_master());
}


TEST(ReferenceExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr ref = gko::ReferenceExecutor::create();

    ref->run(ExampleOperation(value));
    ASSERT_EQ(4, value);
}


TEST(ReferenceExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    exec_ptr ref = gko::ReferenceExecutor::create();

    ref->run(omp_lambda, cuda_lambda, hip_lambda);
    ASSERT_EQ(1, value);
}


TEST(ReferenceExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = ref->alloc<int>(num_elems));
    ASSERT_NO_THROW(ref->free(ptr));
}


TEST(ReferenceExecutor, FreeAcceptsNullptr)
{
    exec_ptr omp = gko::ReferenceExecutor::create();
    ASSERT_NO_THROW(omp->free(nullptr));
}


TEST(ReferenceExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *ptr = nullptr;

    ASSERT_THROW(ptr = ref->alloc<int>(num_elems), gko::AllocationError);

    ref->free(ptr);
}


TEST(ReferenceExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *copy = ref->alloc<int>(num_elems);

    // ReferenceExecutor is a type of OMP executor, so this is O.K.
    ref->copy(num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, CopiesSingleValue)
{
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *el = ref->alloc<int>(1);
    el[0] = 83683;

    EXPECT_EQ(83683, ref->copy_val_to_host(el));

    ref->free(el);
}


TEST(ReferenceExecutor, CopiesDataFromOmp)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr omp = gko::OmpExecutor::create();
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *copy = ref->alloc<int>(num_elems);

    // ReferenceExecutor is a type of OMP executor, so this is O.K.
    ref->copy_from(omp.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, CopiesDataToOmp)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr omp = gko::OmpExecutor::create();
    exec_ptr ref = gko::ReferenceExecutor::create();
    int *copy = omp->alloc<int>(num_elems);

    // ReferenceExecutor is a type of OMP executor, so this is O.K.
    omp->copy_from(ref.get(), num_elems, orig, copy);
    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, IsItsOwnMaster)
{
    exec_ptr ref = gko::ReferenceExecutor::create();

    ASSERT_EQ(ref, ref->get_master());
}


TEST(CudaExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr cuda =
        gko::CudaExecutor::create(0, gko::OmpExecutor::create(), true);

    cuda->run(ExampleOperation(value));
    ASSERT_EQ(2, value);
}


TEST(CudaExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    exec_ptr cuda =
        gko::CudaExecutor::create(0, gko::OmpExecutor::create(), true);

    cuda->run(omp_lambda, cuda_lambda, hip_lambda);
    ASSERT_EQ(2, value);
}


TEST(CudaExecutor, KnowsItsMaster)
{
    auto omp = gko::OmpExecutor::create();
    exec_ptr cuda = gko::CudaExecutor::create(0, omp);

    ASSERT_EQ(omp, cuda->get_master());
}


TEST(CudaExecutor, KnowsItsDeviceId)
{
    auto omp = gko::OmpExecutor::create();
    auto cuda = gko::CudaExecutor::create(0, omp);

    ASSERT_EQ(0, cuda->get_device_id());
}


TEST(CudaExecutor, CanGetDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto cuda = gko::CudaExecutor::create(0, omp);

    ASSERT_EQ(false, cuda->get_device_reset());
}


TEST(CudaExecutor, CanSetDefaultDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto cuda = gko::CudaExecutor::create(0, omp, true);

    ASSERT_EQ(true, cuda->get_device_reset());
}


TEST(CudaExecutor, CanSetDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto cuda = gko::CudaExecutor::create(0, omp);

    cuda->set_device_reset(true);

    ASSERT_EQ(true, cuda->get_device_reset());
}


TEST(HipExecutor, RunsCorrectOperation)
{
    int value = 0;
    exec_ptr hip = gko::HipExecutor::create(0, gko::OmpExecutor::create());

    hip->run(ExampleOperation(value));
    ASSERT_EQ(3, value);
}


TEST(HipExecutor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto cuda_lambda = [&value]() { value = 2; };
    auto hip_lambda = [&value]() { value = 3; };
    exec_ptr hip = gko::HipExecutor::create(0, gko::OmpExecutor::create());

    hip->run(omp_lambda, cuda_lambda, hip_lambda);
    ASSERT_EQ(3, value);
}


TEST(HipExecutor, KnowsItsMaster)
{
    auto omp = gko::OmpExecutor::create();
    exec_ptr hip = gko::HipExecutor::create(0, omp);

    ASSERT_EQ(omp, hip->get_master());
}


TEST(HipExecutor, KnowsItsDeviceId)
{
    auto omp = gko::OmpExecutor::create();
    auto hip = gko::HipExecutor::create(0, omp);

    ASSERT_EQ(0, hip->get_device_id());
}


TEST(HipExecutor, CanGetDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto hip = gko::HipExecutor::create(0, omp);

    ASSERT_EQ(false, hip->get_device_reset());
}


TEST(HipExecutor, CanSetDefaultDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto hip = gko::HipExecutor::create(0, omp, true);

    ASSERT_EQ(true, hip->get_device_reset());
}


TEST(HipExecutor, CanSetDeviceResetBoolean)
{
    auto omp = gko::OmpExecutor::create();
    auto hip = gko::HipExecutor::create(0, omp);

    hip->set_device_reset(true);

    ASSERT_EQ(true, hip->get_device_reset());
}


template <typename T>
struct mock_free : T {
    /**
     * @internal Due to a bug with gcc 5.3, the constructor needs to be called
     * with `()` operator instead of `{}`.
     */
    template <typename... Params>
    mock_free(Params &&... params) : T(std::forward<Params>(params)...)
    {}

    void raw_free(void *ptr) const noexcept override
    {
        called_free = true;
        T::raw_free(ptr);
    }

    mutable bool called_free{false};
};


TEST(ExecutorDeleter, DeletesObject)
{
    auto ref = std::make_shared<mock_free<gko::ReferenceExecutor>>();
    auto x = ref->alloc<int>(5);

    gko::executor_deleter<int>{ref}(x);

    ASSERT_TRUE(ref->called_free);
}


TEST(ExecutorDeleter, AvoidsDeletionForNullExecutor)
{
    int x[5];
    ASSERT_NO_THROW(gko::executor_deleter<int>{nullptr}(x));
}


}  // namespace
