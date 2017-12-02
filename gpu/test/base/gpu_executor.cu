#include <core/base/executor.hpp>


#include <type_traits>


#include <gtest/gtest.h>

#include <core/base/exception_helpers.hpp>

#include <gpu/base/exception.hpp>


#include <gpu/test/base/gpu_kernel.cu>


namespace {


using exec_ptr = std::shared_ptr<gko::Executor>;


TEST(GpuExecutor, AllocatesAndFreesMemory)
{
    const int num_elems = 10;
    auto cpu = gko::CpuExecutor::create();
    exec_ptr gpu = gko::GpuExecutor::create(0, cpu);
    int *ptr ;
    
    cudaError_t errcode;
    ASSERT_NO_THROW(ptr = gpu->alloc<int>(num_elems));
    ASSERT_NO_THROW(gpu->free(ptr));

    errcode = cudaDeviceSynchronize();
    //throw CUDA_ERROR(errcode);
    if (errcode != cudaSuccess){throw ::get_error(errcode);} 
}


TEST(GpuExecutor, FailsWhenOverallocating)
{
    const gko::size_type num_elems = 1ll << 50;  // 4PB of integers
    auto cpu = gko::CpuExecutor::create();
    exec_ptr gpu = gko::GpuExecutor::create(0, cpu);
    int *ptr ;

    ASSERT_THROW(ptr = gpu->alloc<int>(num_elems), gko::AllocationError);

    gpu->free(ptr);
}


TEST(GpuExecutor, CopiesDataFromCpu)
{
    
    double orig[] = {3,8};
    const int num_elems = std::extent<decltype(orig)>::value;
    auto cpu = gko::CpuExecutor::create();
    exec_ptr gpu = gko::GpuExecutor::create(0, cpu);
    double *d_copy = gpu->alloc<double>(num_elems);
   
    double *copy = cpu->alloc<double>(num_elems);
    //copy = orig;
    //copy = {3, 8};
    gpu->copy_from(cpu.get(), num_elems, orig, d_copy);
    //copy = NULL ;
    run_on_gpu(num_elems, d_copy);
    cpu->copy_from(gpu.get(), num_elems, d_copy, copy);
    EXPECT_EQ(2.5, copy[0]);
    EXPECT_EQ(5, copy[1]);

    //gpu->free(d_copy);
    //cpu->free(orig);
    
}


}  // namespace
