#include "core/base/executor.hpp"


#include <iostream>


#include <cuda_runtime.h>


#include "core/base/exception_helpers.hpp"


namespace gko {


void CpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    ASSERT_NO_CUDA_ERRORS(
        cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice));
}


void GpuExecutor::free(void *ptr) const noexcept
{
    auto errcode = cudaFree(ptr);
    if (errcode != cudaSuccess) {
        // Unfortunately, if memory free fails, there's not much we can do
        std::cerr << "Unrecoverable CUDA error in " << __func__ << ": "
                  << cudaGetErrorName(errcode) << ": "
                  << cudaGetErrorString(errcode) << std::endl
                  << "Exiting program" << std::endl;
        std::exit(errcode);
    }
}


void *GpuExecutor::raw_alloc(size_type num_bytes) const
{
    void *dev_ptr = nullptr;
    auto errcode = cudaMalloc(&dev_ptr, num_bytes);
    if (errcode != cudaErrorMemoryAllocation) {
        ASSERT_NO_CUDA_ERRORS(errcode);
    }
    ENSURE_ALLOCATED(dev_ptr, "gpu", num_bytes);
    return dev_ptr;
}


void GpuExecutor::raw_copy_to(const CpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    ASSERT_NO_CUDA_ERRORS(
        cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost));
}


void GpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    ASSERT_NO_CUDA_ERRORS(
        cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToDevice));
}


void GpuExecutor::synchronize() const
{
    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
}


int GpuExecutor::get_num_devices()
{
    int deviceCount = 0;
    auto errcode = cudaGetDeviceCount(&deviceCount);
    if (errcode == cudaErrorNoDevice) {
        return 0;
    }
    ASSERT_NO_CUDA_ERRORS(errcode);
    return deviceCount;
}


}  // namespace gko
