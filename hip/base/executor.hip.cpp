/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include <iostream>


#include <hip/hip_runtime.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "hip/base/config.hip.hpp"
#include "hip/base/device_guard.hip.hpp"
#include "hip/base/hipblas_bindings.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"


namespace gko {


#include "common/base/executor.hpp.inc"


std::shared_ptr<HipExecutor> HipExecutor::create(
    int device_id, std::shared_ptr<Executor> master)
{
    return std::shared_ptr<HipExecutor>(
        new HipExecutor(device_id, std::move(master)),
        [device_id](HipExecutor *exec) {
            delete exec;
            if (!HipExecutor::get_num_execs(device_id)) {
                hip::device_guard g(device_id);
                hipDeviceReset();
            }
        });
}


void OmpExecutor::raw_copy_to(const HipExecutor *dest, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    hip::device_guard g(dest->get_device_id());
    GKO_ASSERT_NO_HIP_ERRORS(
        hipMemcpy(dest_ptr, src_ptr, num_bytes, hipMemcpyHostToDevice));
}


void HipExecutor::raw_free(void *ptr) const noexcept
{
    hip::device_guard g(this->get_device_id());
    auto error_code = hipFree(ptr);
    if (error_code != hipSuccess) {
#if GKO_VERBOSE_LEVEL >= 1
        // Unfortunately, if memory free fails, there's not much we can do
        std::cerr << "Unrecoverable HIP error on device " << this->device_id_
                  << " in " << __func__ << ": " << hipGetErrorName(error_code)
                  << ": " << hipGetErrorString(error_code) << std::endl
                  << "Exiting program" << std::endl;
#endif
        std::exit(error_code);
    }
}


void *HipExecutor::raw_alloc(size_type num_bytes) const
{
    void *dev_ptr = nullptr;
    hip::device_guard g(this->get_device_id());
    auto error_code = hipMalloc(&dev_ptr, num_bytes);
    if (error_code != hipErrorMemoryAllocation) {
        GKO_ASSERT_NO_HIP_ERRORS(error_code);
    }
    GKO_ENSURE_ALLOCATED(dev_ptr, "hip", num_bytes);
    return dev_ptr;
}


void HipExecutor::raw_copy_to(const OmpExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    hip::device_guard g(this->get_device_id());
    GKO_ASSERT_NO_HIP_ERRORS(
        hipMemcpy(dest_ptr, src_ptr, num_bytes, hipMemcpyDeviceToHost));
}


void HipExecutor::raw_copy_to(const CudaExecutor *src, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
#if GINKGO_HIP_PLATFORM_NVCC == 1
    hip::device_guard g(this->get_device_id());
    GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyPeer(dest_ptr, this->device_id_, src_ptr,
                                           src->get_device_id(), num_bytes));
#else
    GKO_NOT_SUPPORTED(HipExecutor);
#endif
}


void HipExecutor::raw_copy_to(const HipExecutor *src, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    hip::device_guard g(this->get_device_id());
    GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyPeer(dest_ptr, this->device_id_, src_ptr,
                                           src->get_device_id(), num_bytes));
}


void HipExecutor::synchronize() const
{
    hip::device_guard g(this->get_device_id());
    GKO_ASSERT_NO_HIP_ERRORS(hipDeviceSynchronize());
}


void HipExecutor::run(const Operation &op) const
{
    this->template log<log::Logger::operation_launched>(this, &op);
    hip::device_guard g(this->get_device_id());
    op.run(
        std::static_pointer_cast<const HipExecutor>(this->shared_from_this()));
    this->template log<log::Logger::operation_completed>(this, &op);
}


int HipExecutor::get_num_devices()
{
    int deviceCount = 0;
    auto error_code = hipGetDeviceCount(&deviceCount);
    if (error_code == hipErrorNoDevice) {
        return 0;
    }
    GKO_ASSERT_NO_HIP_ERRORS(error_code);
    return deviceCount;
}


void HipExecutor::set_gpu_property()
{
    if (device_id_ < this->get_num_devices() && device_id_ >= 0) {
        hip::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &num_multiprocessor_, hipDeviceAttributeMultiprocessorCount,
            device_id_));
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &major_, hipDeviceAttributeComputeCapabilityMajor, device_id_));
        GKO_ASSERT_NO_HIP_ERRORS(hipDeviceGetAttribute(
            &minor_, hipDeviceAttributeComputeCapabilityMinor, device_id_));
#if GINKGO_HIP_PLATFORM_NVCC
        num_warps_per_sm_ = convert_sm_ver_to_cores(major_, minor_) /
                            kernels::hip::hip_config::warp_size;
#else
        // In GCN (Graphics Core Next), each multiprocessor has 4 SIMD
        // Reference: https://en.wikipedia.org/wiki/Graphics_Core_Next
        num_warps_per_sm_ = 4;
#endif  // GINKGO_HIP_PLATFORM_NVCC
    }
}


void HipExecutor::init_handles()
{
    if (device_id_ < this->get_num_devices() && device_id_ >= 0) {
        const auto id = this->get_device_id();
        hip::device_guard g(id);
        this->hipblas_handle_ = handle_manager<hipblasContext>(
            kernels::hip::hipblas::init(), [id](hipblasContext *handle) {
                hip::device_guard g(id);
                kernels::hip::hipblas::destroy_hipblas_handle(handle);
            });
        this->hipsparse_handle_ = handle_manager<hipsparseContext>(
            kernels::hip::hipsparse::init(), [id](hipsparseContext *handle) {
                hip::device_guard g(id);
                kernels::hip::hipsparse::destroy_hipsparse_handle(handle);
            });
    }
}


}  // namespace gko
