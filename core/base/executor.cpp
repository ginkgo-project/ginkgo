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


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/name_demangling.hpp>


namespace gko {


void Operation::run(std::shared_ptr<const OmpExecutor> executor) const
    GKO_NOT_IMPLEMENTED;


void Operation::run(std::shared_ptr<const CudaExecutor> executor) const
    GKO_NOT_IMPLEMENTED;


void Operation::run(std::shared_ptr<const HipExecutor> executor) const
    GKO_NOT_IMPLEMENTED;


void Operation::run(std::shared_ptr<const DpcppExecutor> executor) const
    GKO_NOT_IMPLEMENTED;


void Operation::run(std::shared_ptr<const ReferenceExecutor> executor) const
{
    this->run(static_cast<std::shared_ptr<const OmpExecutor>>(executor));
}


const char* Operation::get_name() const noexcept
{
    static auto name = name_demangling::get_dynamic_type(*this);
    return name.c_str();
}


bool Executor::memory_accessible(
    const std::shared_ptr<const Executor>& other) const
{
    return this->verify_memory_from(other.get());
}


const Executor::exec_info& Executor::get_exec_info() const
{
    return this->exec_info_;
}


std::shared_ptr<OmpExecutor> OmpExecutor::create()
{
    return std::shared_ptr<OmpExecutor>(new OmpExecutor());
}


int OmpExecutor::get_num_cores() const
{
    return this->get_exec_info().num_computing_units;
}


int OmpExecutor::get_num_threads_per_core() const
{
    return this->get_exec_info().num_pu_per_cu;
}


OmpExecutor::OmpExecutor()
{
    this->OmpExecutor::populate_exec_info(machine_topology::get_instance());
}


bool OmpExecutor::verify_memory_to(const OmpExecutor* other) const
{
    return true;
}


bool OmpExecutor::verify_memory_to(const ReferenceExecutor* other) const
{
    return false;
}


bool OmpExecutor::verify_memory_to(const HipExecutor* other) const
{
    return false;
}


bool OmpExecutor::verify_memory_to(const CudaExecutor* other) const
{
    return false;
}


std::shared_ptr<ReferenceExecutor> ReferenceExecutor::create()
{
    return std::shared_ptr<ReferenceExecutor>(new ReferenceExecutor());
}


void ReferenceExecutor::run(const Operation& op) const
{
    this->template log<log::Logger::operation_launched>(this, &op);
    op.run(std::static_pointer_cast<const ReferenceExecutor>(
        this->shared_from_this()));
    this->template log<log::Logger::operation_completed>(this, &op);
}


ReferenceExecutor::ReferenceExecutor()
{
    this->ReferenceExecutor::populate_exec_info(
        machine_topology::get_instance());
}


void ReferenceExecutor::populate_exec_info(const machine_topology*)
{
    this->get_exec_info().device_id = -1;
    this->get_exec_info().num_computing_units = 1;
    this->get_exec_info().num_pu_per_cu = 1;
}


bool ReferenceExecutor::verify_memory_from(const Executor* src_exec) const
{
    return src_exec->verify_memory_to(this);
}


bool ReferenceExecutor::verify_memory_to(const ReferenceExecutor* other) const
{
    return true;
}


bool ReferenceExecutor::verify_memory_to(const OmpExecutor* other) const
{
    return false;
}


bool ReferenceExecutor::verify_memory_to(const DpcppExecutor* other) const
{
    return false;
}


bool ReferenceExecutor::verify_memory_to(const CudaExecutor* other) const
{
    return false;
}


bool ReferenceExecutor::verify_memory_to(const HipExecutor* other) const
{
    return false;
}


int CudaExecutor::get_device_id() const noexcept
{
    return this->get_exec_info().device_id;
}


int CudaExecutor::get_num_warps_per_sm() const noexcept
{
    return this->get_exec_info().num_pu_per_cu;
}


int CudaExecutor::get_num_multiprocessor() const noexcept
{
    return this->get_exec_info().num_computing_units;
}


int CudaExecutor::get_num_warps() const noexcept
{
    return this->get_exec_info().num_computing_units *
           this->get_exec_info().num_pu_per_cu;
}


int CudaExecutor::get_warp_size() const noexcept
{
    return this->get_exec_info().max_subgroup_size;
}


int CudaExecutor::get_major_version() const noexcept
{
    return this->get_exec_info().major;
}


int CudaExecutor::get_minor_version() const noexcept
{
    return this->get_exec_info().minor;
}


cublasContext* CudaExecutor::get_cublas_handle() const
{
    return cublas_handle_.get();
}


cusparseContext* CudaExecutor::get_cusparse_handle() const
{
    return cusparse_handle_.get();
}


std::vector<int> CudaExecutor::get_closest_pus() const
{
    return this->get_exec_info().closest_pu_ids;
}


int CudaExecutor::get_closest_numa() const
{
    return this->get_exec_info().numa_node;
}


CudaExecutor::CudaExecutor(int device_id, std::shared_ptr<Executor> master,
                           bool device_reset = false,
                           allocation_mode alloc_mode = default_cuda_alloc_mode)
    : EnableDeviceReset{device_reset}, master_(master), alloc_mode_{alloc_mode}
{
    this->get_exec_info().device_id = device_id;
    this->get_exec_info().num_computing_units = 0;
    this->get_exec_info().num_pu_per_cu = 0;
    this->CudaExecutor::populate_exec_info(machine_topology::get_instance());
    if (this->get_exec_info().closest_pu_ids.size()) {
        machine_topology::get_instance()->bind_to_pus(this->get_closest_pus());
    }
    // it only gets attribute from device, so it should not be affected by
    // DeviceReset.
    this->set_gpu_property();
    // increase the number of executor before any operations may be affected
    // by DeviceReset.
    increase_num_execs(this->get_exec_info().device_id);
    this->init_handles();
}


bool CudaExecutor::verify_memory_to(const ReferenceExecutor* other) const
{
    return false;
}


bool CudaExecutor::verify_memory_to(const OmpExecutor* other) const
{
    return false;
}


bool CudaExecutor::verify_memory_to(const DpcppExecutor* other) const
{
    return false;
}


int HipExecutor::get_device_id() const noexcept
{
    return this->get_exec_info().device_id;
}


int HipExecutor::get_num_warps_per_sm() const noexcept
{
    return this->get_exec_info().num_pu_per_cu;
}


int HipExecutor::get_num_multiprocessor() const noexcept
{
    return this->get_exec_info().num_computing_units;
}


int HipExecutor::get_num_warps() const noexcept
{
    return this->get_exec_info().num_computing_units *
           this->get_exec_info().num_pu_per_cu;
}


int HipExecutor::get_warp_size() const noexcept
{
    return this->get_exec_info().max_subgroup_size;
}


int HipExecutor::get_major_version() const noexcept
{
    return this->get_exec_info().major;
}


int HipExecutor::get_minor_version() const noexcept
{
    return this->get_exec_info().minor;
}


hipblasContext* HipExecutor::get_hipblas_handle() const
{
    return hipblas_handle_.get();
}


hipsparseContext* HipExecutor::get_hipsparse_handle() const
{
    return hipsparse_handle_.get();
}


std::vector<int> HipExecutor::get_closest_pus() const
{
    return this->get_exec_info().closest_pu_ids;
}


int HipExecutor::get_closest_numa() const
{
    return this->get_exec_info().numa_node;
}


HipExecutor::HipExecutor(int device_id, std::shared_ptr<Executor> master,
                         bool device_reset = false,
                         allocation_mode alloc_mode = default_hip_alloc_mode)
    : EnableDeviceReset{device_reset}, master_(master), alloc_mode_(alloc_mode)
{
    this->get_exec_info().device_id = device_id;
    this->get_exec_info().num_computing_units = 0;
    this->get_exec_info().num_pu_per_cu = 0;
    this->HipExecutor::populate_exec_info(machine_topology::get_instance());
    if (this->get_exec_info().closest_pu_ids.size()) {
        machine_topology::get_instance()->bind_to_pus(this->get_closest_pus());
    }
    // it only gets attribute from device, so it should not be affected by
    // DeviceReset.
    this->set_gpu_property();
    // increase the number of executor before any operations may be affected
    // by DeviceReset.
    increase_num_execs(this->get_exec_info().device_id);
    this->init_handles();
}


bool HipExecutor::verify_memory_to(const ReferenceExecutor* other) const
{
    return false;
}


bool HipExecutor::verify_memory_to(const OmpExecutor* other) const
{
    return false;
}


bool HipExecutor::verify_memory_to(const DpcppExecutor* other) const
{
    return false;
}


int DpcppExecutor::get_device_id() const noexcept
{
    return this->get_exec_info().device_id;
}


::cl::sycl::queue* DpcppExecutor::get_queue() const noexcept
{
    return queue_.get();
}


const std::vector<int>& DpcppExecutor::get_subgroup_sizes() const noexcept
{
    return this->get_exec_info().subgroup_sizes;
}


int DpcppExecutor::get_num_computing_units() const noexcept
{
    return this->get_exec_info().num_computing_units;
}


const std::vector<int>& DpcppExecutor::get_max_workitem_sizes() const noexcept
{
    return this->get_exec_info().max_workitem_sizes;
}


int DpcppExecutor::get_max_workgroup_size() const noexcept
{
    return this->get_exec_info().max_workgroup_size;
}


int DpcppExecutor::get_max_subgroup_size() const noexcept
{
    return this->get_exec_info().max_subgroup_size;
}


const std::string& DpcppExecutor::get_device_type() const noexcept
{
    return this->get_exec_info().device_type;
}


DpcppExecutor::DpcppExecutor(int device_id, std::shared_ptr<Executor> master,
                             std::string device_type)
    : master_(master)
{
    std::for_each(device_type.begin(), device_type.end(),
                  [](char& c) { c = std::tolower(c); });
    this->get_exec_info().device_type = std::string(device_type);
    this->get_exec_info().device_id = device_id;
    this->set_device_property();
}


bool DpcppExecutor::verify_memory_to(const ReferenceExecutor* other) const
{
    return false;
}


bool DpcppExecutor::verify_memory_to(const CudaExecutor* other) const
{
    return false;
}


bool DpcppExecutor::verify_memory_to(const HipExecutor* other) const
{
    return false;
}


}  // namespace gko
