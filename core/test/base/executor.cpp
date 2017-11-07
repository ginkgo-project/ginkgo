#include <core/base/executor.hpp>


#include <type_traits>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>


namespace {


using exec_ptr = std::shared_ptr<msparse::Executor>;


class ExampleOperation : public msparse::Operation {
public:
    explicit ExampleOperation(int &val) : value(val) {}
    void run(const msparse::CpuExecutor *) const override { value = 1; }
    void run(const msparse::GpuExecutor *) const override { value = 2; }
    void run(const msparse::ReferenceExecutor *) const override { value = 3; }

    int &value;
};


TEST(Executor, RunsCorrectOperation)
{
    int value = 0;
    ExampleOperation op(value);

    exec_ptr cpu = msparse::CpuExecutor::create();
    cpu->run(op);
    EXPECT_EQ(1, value);

    exec_ptr gpu = msparse::GpuExecutor::create(
        0, std::static_pointer_cast<msparse::CpuExecutor>(cpu));
    gpu->run(op);
    EXPECT_EQ(2, value);

    exec_ptr ref = msparse::ReferenceExecutor::create();
    ref->run(op);
    EXPECT_EQ(3, value);
}


TEST(Executor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto cpu_lambda = [&value]() { value = 1; };
    auto gpu_lambda = [&value]() { value = 2; };

    exec_ptr cpu = msparse::CpuExecutor::create();
    cpu->run(cpu_lambda, gpu_lambda);
    EXPECT_EQ(1, value);

    exec_ptr gpu = msparse::GpuExecutor::create(
        0, std::static_pointer_cast<msparse::CpuExecutor>(cpu));
    gpu->run(cpu_lambda, gpu_lambda);
    EXPECT_EQ(2, value);

    exec_ptr ref = msparse::ReferenceExecutor::create();
    ref->run(cpu_lambda, gpu_lambda);
    EXPECT_EQ(1, value);
}


TEST(CpuExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    exec_ptr cpu = msparse::CpuExecutor::create();
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = cpu->alloc<int>(num_elems));
    ASSERT_NO_THROW(cpu->free(ptr));
}


TEST(CpuExecutor, FailsWhenOverallocating)
{
    const msparse::size_type num_elems = 1ll << 50;  // 4PB of integers
    exec_ptr cpu = msparse::CpuExecutor::create();
    int *ptr = nullptr;

    ASSERT_THROW(ptr = cpu->alloc<int>(num_elems), msparse::AllocationError);
    cpu->free(ptr);
}


TEST(CpuExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr cpu = msparse::CpuExecutor::create();
    int *copy = cpu->alloc<int>(num_elems);
    // user code is run on the CPU, so local variables are in CPU memory
    cpu->copy_from(cpu.get(), num_elems, orig, copy);

    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    cpu->free(copy);
}


TEST(CpuExecutor, IsItsOwnMaster)
{
    exec_ptr cpu = msparse::CpuExecutor::create();
    ASSERT_EQ(cpu, cpu->get_master());
}


TEST(ReferenceExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    exec_ptr ref = msparse::ReferenceExecutor::create();
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = ref->alloc<int>(num_elems));
    ASSERT_NO_THROW(ref->free(ptr));
}


TEST(ReferenceExecutor, FailsWhenOverallocating)
{
    const msparse::size_type num_elems = 1ll << 50;  // 4PB of integers
    exec_ptr ref = msparse::ReferenceExecutor::create();
    int *ptr = nullptr;

    ASSERT_THROW(ptr = ref->alloc<int>(num_elems), msparse::AllocationError);
    ref->free(ptr);
}


TEST(ReferenceExecutor, CopiesData)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr ref = msparse::ReferenceExecutor::create();
    int *copy = ref->alloc<int>(num_elems);
    // ReferenceExecutor is a type of CPU executor, so this is O.K.
    ref->copy_from(ref.get(), num_elems, orig, copy);

    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, CopiesDataFromCpu)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr cpu = msparse::CpuExecutor::create();
    exec_ptr ref = msparse::ReferenceExecutor::create();
    int *copy = ref->alloc<int>(num_elems);
    // ReferenceExecutor is a type of CPU executor, so this is O.K.
    ref->copy_from(cpu.get(), num_elems, orig, copy);

    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, CopiesDataToCpu)
{
    int orig[] = {3, 8};
    const int num_elems = std::extent<decltype(orig)>::value;
    exec_ptr cpu = msparse::CpuExecutor::create();
    exec_ptr ref = msparse::ReferenceExecutor::create();
    int *copy = cpu->alloc<int>(num_elems);
    // ReferenceExecutor is a type of CPU executor, so this is O.K.
    cpu->copy_from(ref.get(), num_elems, orig, copy);

    EXPECT_EQ(3, copy[0]);
    EXPECT_EQ(8, copy[1]);

    ref->free(copy);
}


TEST(ReferenceExecutor, IsItsOwnMaster)
{
    exec_ptr ref = msparse::ReferenceExecutor::create();
    ASSERT_EQ(ref, ref->get_master());
}


TEST(GpuExecutor, KnowsItsMaster)
{
    auto cpu = msparse::CpuExecutor::create();
    exec_ptr gpu = msparse::GpuExecutor::create(0, cpu);
    ASSERT_EQ(cpu, gpu->get_master());
}


TEST(GpuExecutor, KnowsItsDeviceId)
{
    auto cpu = msparse::CpuExecutor::create();
    auto gpu = msparse::GpuExecutor::create(5, cpu);
    ASSERT_EQ(5, gpu->get_device_id());
}


}  // namespace
