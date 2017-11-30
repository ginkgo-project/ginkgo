#include "core/base/executor.hpp"


#include "core/base/exception_helpers.hpp"


#include <cuda_runtime.h>


namespace gko {


void CpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    cudaError_t errcode;
    errcode = cudaMemcpy(void *dest_ptr, const void *src_ptr,
                         size_type num_bytes, cudaMemcpyHostToDevice);
    if (errcode != 0) {
        //    THROW_CUDA_ERROR(errcode);
    }
}


void GpuExecutor::free(void *ptr) const noexcept { cudaFree(void *ptr); }


void *GpuExecutor::raw_alloc(size_type num_bytes) const

{
    void *dev_ptr;
    cudaMalloc(&dev_ptr, num_bytes);
    ENSURE_ALLOCATED(dev_ptr, gpu, num_bytes);
    return dev_ptr;
}


void GpuExecutor::raw_copy_to(const CpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    cudaError_t errcode;
    errcode = cudaMemcpy(void *dest_ptr, const void *src_ptr,
                         size_type num_bytes, cudaMemcpyDeviceToHost);
    if (errcode != 0) {
        //  THROW_CUDA_ERROR(errcode);
    }
}

void GpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    cudaError_t errcode;
    errcode = cudaMemcpy(void *dest_ptr, const void *src_ptr,
                         size_type num_bytes, cudaMemcpyDeviceToDevice);
    if (errcode != 0) {
        // THROW_CUDA_ERROR(errcode);
    }
}

}  // namespace gko
