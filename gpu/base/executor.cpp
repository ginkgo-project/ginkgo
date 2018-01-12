/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
