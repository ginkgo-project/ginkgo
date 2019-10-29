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


#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <iostream>


#include <cuda.h>
#include <cuda_runtime.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/config.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/device_guard.hpp"

#if GKO_HAVE_HWLOC
#include "hwloc/cuda.h"
#endif

namespace gko {


#include "common/base/executor.hpp.inc"


namespace machine_config {


template <>
void topology<CudaExecutor>::load_gpus()
{
#if GKO_HAVE_HWLOC
    std::size_t num_in_numa = 0;
    int last_numa = 0;
    auto topology = this->topo_.get();
    int ngpus = 0;
    GKO_ASSERT_NO_CUDA_ERRORS(cudaGetDeviceCount(&ngpus));
    auto n_objs = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_OS_DEVICE);
    for (std::size_t i = 1; i < n_objs; i++, num_in_numa++) {
        hwloc_obj_t obj = NULL;
        while ((obj = hwloc_get_next_osdev(topology, obj)) != NULL) {
            if (HWLOC_OBJ_OSDEV_COPROC == obj->attr->osdev.type && obj->name &&
                !strncmp("cuda", obj->name, 4) &&
                atoi(obj->name + 4) == (int)i) {
                while (obj &&
                       (!obj->nodeset || hwloc_bitmap_iszero(obj->nodeset)))
                    obj = obj->parent;
                if (obj && obj->nodeset) {
                    auto this_numa = hwloc_bitmap_first(obj->nodeset);
                    if (this_numa != last_numa) {
                        num_in_numa = 0;
                    }
                    this->gpus_.push_back(
                        topology_obj_info{obj, this_numa, i, num_in_numa});
                    last_numa = this_numa;
                }
            }
        }
    }

#endif
}


}  // namespace machine_config


std::shared_ptr<CudaExecutor> CudaExecutor::create(
    int device_id, std::shared_ptr<Executor> master)
{
    return std::shared_ptr<CudaExecutor>(
        new CudaExecutor(device_id, std::move(master)),
        [device_id](CudaExecutor *exec) {
            delete exec;
            if (!CudaExecutor::get_num_execs(device_id)) {
                cuda::device_guard g(device_id);
                cudaDeviceReset();
            }
        });
}


std::shared_ptr<CudaExecutor> CudaExecutor::create(
    int device_id, std::shared_ptr<MemorySpace> mem_space,
    std::shared_ptr<Executor> master)
{
    return std::shared_ptr<CudaExecutor>(
        new CudaExecutor(device_id, mem_space, std::move(master)),
        [device_id](CudaExecutor *exec) {
            delete exec;
            if (!CudaExecutor::get_num_execs(device_id)) {
                device_guard g(device_id);
                cudaDeviceReset();
            }
        });
}


void CudaExecutor::synchronize() const
{
    cuda::device_guard g(this->get_device_id());
    GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
}


void CudaExecutor::run(const Operation &op) const
{
    this->template log<log::Logger::operation_launched>(this, &op);
    cuda::device_guard g(this->get_device_id());
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
    GKO_ASSERT_NO_CUDA_ERRORS(error_code);
    return deviceCount;
}


void CudaExecutor::set_gpu_property()
{
    if (device_id_ < this->get_num_devices() && device_id_ >= 0) {
        cuda::device_guard g(this->get_device_id());
        GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &major_, cudaDevAttrComputeCapabilityMajor, device_id_));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &minor_, cudaDevAttrComputeCapabilityMinor, device_id_));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaDeviceGetAttribute(
            &num_multiprocessor_, cudaDevAttrMultiProcessorCount, device_id_));
        num_warps_per_sm_ = convert_sm_ver_to_cores(major_, minor_) /
                            kernels::cuda::config::warp_size;
        warp_size_ = kernels::cuda::config::warp_size;
    }
}


void CudaExecutor::init_handles()
{
    if (device_id_ < this->get_num_devices() && device_id_ >= 0) {
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
