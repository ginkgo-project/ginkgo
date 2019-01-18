/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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

#include <ginkgo/core/base/executor.hpp>


#include <iostream>


#include <cuda_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/cusparse_bindings.hpp"


namespace gko {
namespace {


class device_guard {
public:
    device_guard(int device_id)
    {
        ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&original_device_id));
        ASSERT_NO_CUDA_ERRORS(cudaSetDevice(device_id));
    }

    ~device_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (std::uncaught_exception()) {
            cudaSetDevice(original_device_id);
        } else {
            ASSERT_NO_CUDA_ERRORS(cudaSetDevice(original_device_id));
        }
    }

private:
    int original_device_id{};
};


// The function is copied from _ConvertSMVer2Cores of
// cuda-9.2/samples/common/inc/helper_cuda.h
inline int convert_sm_ver_to_cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},  // Kepler Generation (SM 3.0) GK10x class
        {0x32, 192},  // Kepler Generation (SM 3.2) GK10x class
        {0x35, 192},  // Kepler Generation (SM 3.5) GK11x class
        {0x37, 192},  // Kepler Generation (SM 3.7) GK21x class
        {0x50, 128},  // Maxwell Generation (SM 5.0) GM10x class
        {0x52, 128},  // Maxwell Generation (SM 5.2) GM20x class
        {0x53, 128},  // Maxwell Generation (SM 5.3) GM20x class
        {0x60, 64},   // Pascal Generation (SM 6.0) GP100 class
        {0x61, 128},  // Pascal Generation (SM 6.1) GP10x class
        {0x62, 128},  // Pascal Generation (SM 6.2) GP10x class
        {0x70, 64},   // Volta Generation (SM 7.0) GV100 class
        {0x72, 64},   // Volta Generation (SM 7.2) GV11b class
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    // If we don't find the values, we use the last valid value by default to
    // allow proper execution
    std::cerr << "MapSMtoCores for SM " << major << "." << minor
              << "is undefined. The default value of "
              << nGpuArchCoresPerSM[index - 1].Cores << " Cores/SM is used."
              << std::endl;
    return nGpuArchCoresPerSM[index - 1].Cores;
}


}  // namespace


void OmpExecutor::raw_copy_to(const CudaExecutor *dest, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    device_guard g(dest->get_device_id());
    ASSERT_NO_CUDA_ERRORS(
        cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice));
}


void CudaExecutor::raw_free(void *ptr) const noexcept
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


void *CudaExecutor::raw_alloc(size_type num_bytes) const
{
    void *dev_ptr = nullptr;
    device_guard g(this->get_device_id());
    auto error_code = cudaMalloc(&dev_ptr, num_bytes);
    if (error_code != cudaErrorMemoryAllocation) {
        ASSERT_NO_CUDA_ERRORS(error_code);
    }
    ENSURE_ALLOCATED(dev_ptr, "cuda", num_bytes);
    return dev_ptr;
}


void CudaExecutor::raw_copy_to(const OmpExecutor *, size_type num_bytes,
                               const void *src_ptr, void *dest_ptr) const
{
    device_guard g(this->get_device_id());
    ASSERT_NO_CUDA_ERRORS(
        cudaMemcpy(dest_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost));
}


void CudaExecutor::raw_copy_to(const CudaExecutor *src, size_type num_bytes,
                               const void *src_ptr, void *dest_ptr) const
{
    ASSERT_NO_CUDA_ERRORS(cudaMemcpyPeer(dest_ptr, this->device_id_, src_ptr,
                                         src->get_device_id(), num_bytes));
}


void CudaExecutor::synchronize() const
{
    device_guard g(this->get_device_id());
    ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
}


void CudaExecutor::run(const Operation &op) const
{
    this->template log<log::Logger::operation_launched>(this, &op);
    device_guard g(this->get_device_id());
    op.run(
        std::static_pointer_cast<const CudaExecutor>(this->shared_from_this()));
    this->template log<log::Logger::operation_completed>(this, &op);
}


int CudaExecutor::get_num_devices()
{
    int deviceCount = 0;
    auto error_code = cudaGetDeviceCount(&deviceCount);
    if (error_code == cudaErrorNoDevice) {
        return 0;
    }
    ASSERT_NO_CUDA_ERRORS(error_code);
    return deviceCount;
}


void CudaExecutor::set_gpu_property()
{
    if (device_id_ < this->get_num_devices() && device_id_ >= 0) {
        device_guard g(this->get_device_id());
        ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &major_, cudaDevAttrComputeCapabilityMajor, device_id_));
        ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &minor_, cudaDevAttrComputeCapabilityMinor, device_id_));
        ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &num_multiprocessor_, cudaDevAttrMultiProcessorCount, device_id_));
        num_cores_per_sm_ = convert_sm_ver_to_cores(major_, minor_);
    }
}


void CudaExecutor::init_handles()
{
    this->cublas_handle_ = handle_manager<cublasContext>(
        kernels::cuda::cublas::init(), kernels::cuda::cublas::destroy);
    typedef void (*func_ptr)(cusparseHandle_t);
    this->cusparse_handle_ = handle_manager<cusparseContext>(
        kernels::cuda::cusparse::init(),
        static_cast<func_ptr>(kernels::cuda::cusparse::destroy));
}


}  // namespace gko
