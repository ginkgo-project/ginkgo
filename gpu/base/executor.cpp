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


class device_guard {
public:
    device_guard(int device_id)
    {
        ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&original_device_id));
        ASSERT_NO_CUDA_ERRORS(cudaSetDevice(device_id));
    }

    ~device_guard()
    {
        ASSERT_NO_CUDA_ERRORS(cudaSetDevice(original_device_id));
    }

private:
    int original_device_id{};
};


void CpuExecutor::raw_copy_to(const GpuExecutor *dest, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    device_guard g(dest->get_device_id());
    ASSERT_NO_CUDA_ERRORS(
        cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice));
}


void GpuExecutor::free(void *ptr) const noexcept
{
    device_guard g(this->get_device_id());
    auto error_code = cudaFree(ptr);
    if (error_code != cudaSuccess) {
        // Unfortunately, if memory free fails, there's not much we can do
        std::cerr << "Unrecoverable CUDA error on device " << this->device_id_
                  << " in " << __func__ << ": " << cudaGetErrorName(error_code)
                  << ": " << cudaGetErrorString(error_code) << std::endl
                  << "Exiting program" << std::endl;
        std::exit(error_code);
    }
}


void *GpuExecutor::raw_alloc(size_type num_bytes) const
{
    void *dev_ptr = nullptr;
    device_guard g(this->get_device_id());
    auto error_code = cudaMalloc(&dev_ptr, num_bytes);
    if (error_code != cudaErrorMemoryAllocation) {
        ASSERT_NO_CUDA_ERRORS(error_code);
    }
    ENSURE_ALLOCATED(dev_ptr, "gpu", num_bytes);
    return dev_ptr;
}


void GpuExecutor::raw_copy_to(const CpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    device_guard g(this->get_device_id());
    ASSERT_NO_CUDA_ERRORS(
        cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost));
}


void GpuExecutor::raw_copy_to(const GpuExecutor *src, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    ASSERT_NO_CUDA_ERRORS(cudaMemcpyPeer(dest_ptr, this->device_id_, src_ptr,
                                         src->get_device_id(), num_bytes));
}


void GpuExecutor::synchronize() const
{
    device_guard g(this->get_device_id());
    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
}


void GpuExecutor::run(const Operation &op) const
{
    device_guard g(this->get_device_id());
    op.run(
        std::static_pointer_cast<const GpuExecutor>(this->shared_from_this()));
}


int GpuExecutor::get_num_devices()
{
    int deviceCount = 0;
    auto error_code = cudaGetDeviceCount(&deviceCount);
    if (error_code == cudaErrorNoDevice) {
        return 0;
    }
    ASSERT_NO_CUDA_ERRORS(error_code);
    return deviceCount;
}


#undef SET_DEVICE_AND_CALL


}  // namespace gko
