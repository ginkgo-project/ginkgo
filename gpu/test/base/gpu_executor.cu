#include <core/base/executor.hpp>


#include <type_traits>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/exception_helpers.hpp>


namespace {


class GpuExecutor : public ::testing::Test {
protected:
    GpuExecutor() : cpu(gko::CpuExecutor::create()), gpu(nullptr) {}

    void SetUp()
    {
        ASSERT_GT(gko::GpuExecutor::get_num_devices(), 0);
        gpu = gko::GpuExecutor::create(0, cpu);
    }

    void TearDown()
    {
        if (gpu != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(gpu->synchronize());
        }
    }

    std::shared_ptr<gko::CpuExecutor> cpu;
    std::shared_ptr<gko::GpuExecutor> gpu;
};


TEST_F(GpuExecutor, AllocatesAndFreesMemory)
{
    int *ptr = nullptr;

    ASSERT_NO_THROW(ptr = gpu->alloc<int>(2));
    ASSERT_NO_THROW(gpu->free(ptr));
}


TEST_F(GpuExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    int *ptr = nullptr;

    ASSERT_THROW(
        {
            ptr = gpu->alloc<int>(num_elems);
            gpu->synchronize();
        },
        gko::AllocationError);

    gpu->free(ptr);
}


__global__ void check_data(int *data)
{
    if (data[0] != 3 || data[1] != 8) {
        asm("trap;");
    }
}

TEST_F(GpuExecutor, CopiesDataToGpu)
{
    int orig[] = {3, 8};
    auto *copy = gpu->alloc<int>(2);

    gpu->copy_from(cpu.get(), 2, orig, copy);

    check_data<<<1, 1>>>(copy);
    ASSERT_NO_THROW(gpu->synchronize());
    gpu->free(copy);
}


__global__ void init_data(int *data)
{
    data[0] = 5;
    data[1] = 2;
}

TEST_F(GpuExecutor, CopiesDataFromGpu)
{
    int copy[2];
    auto orig = gpu->alloc<int>(2);
    init_data<<<1, 1>>>(orig);

    cpu->copy_from(gpu.get(), 2, orig, copy);

    EXPECT_EQ(5, copy[0]);
    ASSERT_EQ(2, copy[1]);
    gpu->free(orig);
}

TEST_F(GpuExecutor, CudaErrorThrowWorks)

{
    cudaError_t err_code = cudaSuccess;
    ASSERT_NO_THROW(ASSERT_NO_CUDA_ERRORS(err_code));
}

TEST_F(GpuExecutor, SynchronizesWithMasterDoesNotThrowError)

{
    int orig[] = {3, 8};
    auto *copy = gpu->alloc<int>(2);

    gpu->copy_from(cpu.get(), 2, orig, copy);

    check_data<<<1, 1>>>(copy);
    ASSERT_NO_THROW(gpu->synchronize());
}


TEST_F(GpuExecutor, MasterKnowsNumberOfDevices)

{
    int count = 0;
    cudaGetDeviceCount(&count);
    ASSERT_EQ(count, gpu->get_num_devices());
}

}  // namespace
