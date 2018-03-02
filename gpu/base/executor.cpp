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


#define SET_DEVICE_AND_CALL(__device_id__, __function__) \
    int prev_device;                                     \
    ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&prev_device));  \
    ASSERT_NO_CUDA_ERRORS(cudaSetDevice(__device_id__)); \
    auto errorcode = __function__;                       \
    ASSERT_NO_CUDA_ERRORS(cudaSetDevice(prev_device))


void CpuExecutor::raw_copy_to(const GpuExecutor *dest, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    SET_DEVICE_AND_CALL(
        dest->get_device_id(),
        cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERRORS(errorcode);
}


void GpuExecutor::free(void *ptr) const noexcept
{
    SET_DEVICE_AND_CALL(this->device_id_, cudaFree(ptr));
    if (errorcode != cudaSuccess) {
        // Unfortunately, if memory free fails, there's not much we can do
        std::cerr << "Unrecoverable CUDA error on device " << this->device_id_
                  << " in " << __func__ << ": " << cudaGetErrorName(errorcode)
                  << ": " << cudaGetErrorString(errorcode) << std::endl
                  << "Exiting program" << std::endl;
        std::exit(errorcode);
    }
}


void *GpuExecutor::raw_alloc(size_type num_bytes) const
{
    void *dev_ptr = nullptr;
    SET_DEVICE_AND_CALL(this->device_id_, cudaMalloc(&dev_ptr, num_bytes));
    if (errorcode != cudaErrorMemoryAllocation) {
        ASSERT_NO_CUDA_ERRORS(errorcode);
    }
    ENSURE_ALLOCATED(dev_ptr, "gpu", num_bytes);
    return dev_ptr;
}


void GpuExecutor::raw_copy_to(const CpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    SET_DEVICE_AND_CALL(
        this->device_id_,
        cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost));
    ASSERT_NO_CUDA_ERRORS(errorcode);
}


void GpuExecutor::raw_copy_to(const GpuExecutor *src, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    ASSERT_NO_CUDA_ERRORS(cudaMemcpyPeer(dest_ptr, this->device_id_, src_ptr,
                                         src->get_device_id(), num_bytes));
}


void GpuExecutor::synchronize() const
{
    SET_DEVICE_AND_CALL(this->device_id_, cudaDeviceSynchronize());
    ASSERT_NO_CUDA_ERRORS(errorcode);
}


void GpuExecutor::run(const Operation &op) const
{
    int prev_device;
    ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&prev_device));
    ASSERT_NO_CUDA_ERRORS(cudaSetDevice(this->device_id_));
    op.run(
        std::static_pointer_cast<const GpuExecutor>(this->shared_from_this()));
    ASSERT_NO_CUDA_ERRORS(cudaSetDevice(prev_device));
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


#undef SET_DEVICE_AND_CALL


}  // namespace gko
