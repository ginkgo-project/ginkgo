/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <cuda_runtime.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/device.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/config.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/cusparse_handle.hpp"
#include "cuda/base/device_guard.hpp"


namespace gko {


#include "common/cuda_hip/base/executor.hpp.inc"


std::shared_ptr<CudaExecutor> CudaExecutor::create(
    int device_id, std::shared_ptr<Executor> master, bool device_reset,
    allocation_mode alloc_mode)
{
    return std::shared_ptr<CudaExecutor>(
        new CudaExecutor(device_id, std::move(master), device_reset,
                         alloc_mode),
        [device_id](CudaExecutor* exec) {
            auto device_reset = exec->get_device_reset();
            std::lock_guard<std::mutex> guard(
                nvidia_device::get_mutex(device_id));
            delete exec;
            auto& num_execs = nvidia_device::get_num_execs(device_id);
            num_execs--;
            if (!num_execs && device_reset) {
                cuda::device_guard g(device_id);
                cudaDeviceReset();
            }
        });
}


void CudaExecutor::populate_exec_info(const MachineTopology* mach_topo)
{
    if (this->get_device_id() < this->get_num_devices() &&
        this->get_device_id() >= 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(
            cudaDeviceGetPCIBusId(&(this->get_exec_info().pci_bus_id.front()),
                                  13, this->get_device_id()));

        auto cuda_hwloc_obj =
            mach_topo->get_pci_device(this->get_exec_info().pci_bus_id);
        if (cuda_hwloc_obj) {
            this->get_exec_info().numa_node = cuda_hwloc_obj->closest_numa;
            this->get_exec_info().closest_pu_ids =
                cuda_hwloc_obj->closest_pu_ids;
        }
    }
}


void CudaExecutor::synchronize() const
{
    cuda::device_guard g(this->get_device_id());
    GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
}


void CudaExecutor::run(const Operation& op) const
{
    this->template log<log::Logger::operation_launched>(this, &op);
    cuda::device_guard g(this->get_device_id());
    op.run(
        std::static_pointer_cast<const CudaExecutor>(this->shared_from_this()));
    this->template log<log::Logger::operation_completed>(this, &op);
}


std::shared_ptr<AsyncHandle> CudaExecutor::run(
    const AsyncOperation& op, std::shared_ptr<AsyncHandle> handle) const
{
    cuda::device_guard g(this->get_device_id());
    return op.run(
        std::static_pointer_cast<const CudaExecutor>(this->shared_from_this()),
        handle);
    // FIXME
    // this->template log<log::Logger::operation_completed>(this, &op);
}


int CudaExecutor::get_num_devices()
{
    int deviceCount = 0;
    auto error_code = cudaGetDeviceCount(&deviceCount);
    if (error_code == cudaErrorNoDevice) {
        return 0;
    }
    GKO_ASSERT_NO_CUDA_ERRORS(error_code);
    return deviceCount;
}


void CudaExecutor::set_gpu_property()
{
    if (this->get_device_id() < this->get_num_devices() &&
        this->get_device_id() >= 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &this->get_exec_info().major, cudaDevAttrComputeCapabilityMajor,
            this->get_device_id()));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &this->get_exec_info().minor, cudaDevAttrComputeCapabilityMinor,
            this->get_device_id()));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &this->get_exec_info().num_computing_units,
            cudaDevAttrMultiProcessorCount, this->get_device_id()));
        auto max_threads_per_block = 0;
        GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &max_threads_per_block, cudaDevAttrMaxThreadsPerBlock,
            this->get_device_id()));
        std::vector<int> max_threads_per_block_dim(3, 0);
        GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &max_threads_per_block_dim[0], cudaDevAttrMaxBlockDimX,
            this->get_device_id()));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &max_threads_per_block_dim[1], cudaDevAttrMaxBlockDimY,
            this->get_device_id()));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &max_threads_per_block_dim[2], cudaDevAttrMaxBlockDimZ,
            this->get_device_id()));
        this->get_exec_info().max_workgroup_size = max_threads_per_block;
        this->get_exec_info().max_workitem_sizes = max_threads_per_block_dim;
        this->get_exec_info().num_pu_per_cu =
            convert_sm_ver_to_cores(this->get_exec_info().major,
                                    this->get_exec_info().minor) /
            kernels::cuda::config::warp_size;
        this->get_exec_info().max_subgroup_size =
            kernels::cuda::config::warp_size;
    }
}


void CudaExecutor::init_handles()
{
    if (this->get_device_id() < this->get_num_devices() &&
        this->get_device_id() >= 0) {
        const auto id = this->get_device_id();
        cuda::device_guard g(id);
        this->cublas_handle_ = handle_manager<cublasContext>(
            kernels::cuda::cublas::init(), [id](cublasHandle_t handle) {
                cuda::device_guard g(id);
                kernels::cuda::cublas::destroy(handle);
            });
        this->cusparse_handle_ = handle_manager<cusparseContext>(
            kernels::cuda::cusparse::init(), [id](cusparseHandle_t handle) {
                cuda::device_guard g(id);
                kernels::cuda::cusparse::destroy(handle);
            });
    }
}


}  // namespace gko
