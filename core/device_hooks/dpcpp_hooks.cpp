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

#include <memory>
#include <string>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_dpcpp_version() noexcept
{
    // We just return the version with a special "not compiled" tag in
    // placeholder modules.
    return {GKO_VERSION_STR, "not compiled"};
}


std::shared_ptr<DpcppExecutor> DpcppExecutor::create(
    int device_id, std::shared_ptr<Executor> master, std::string device_type)
{
    return std::shared_ptr<DpcppExecutor>(
        new DpcppExecutor(device_id, std::move(master), device_type));
}


std::shared_ptr<DpcppExecutor> DpcppExecutor::create(
    int device_id, std::shared_ptr<MemorySpace> mem_space,
    std::shared_ptr<Executor> master, std::string device_type)
{
    return std::shared_ptr<DpcppExecutor>(new DpcppExecutor(
        device_id, std::move(mem_space), std::move(master), device_type));
}


DpcppMemorySpace::DpcppMemorySpace(int device_id, std::string device_type)
    : device_id_(device_id), device_type_(device_type)
{}


int DpcppMemorySpace::get_num_devices(std::string) { return 0; }


void DpcppExecutor::populate_exec_info(const MachineTopology* mach_topo)
{
    // This method is always called, so cannot throw when not compiled.
}


std::shared_ptr<AsyncHandle> HostMemorySpace::raw_copy_to(
    const DpcppMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const GKO_NOT_COMPILED(dpcpp);


std::shared_ptr<AsyncHandle> ReferenceMemorySpace::raw_copy_to(
    const DpcppMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const GKO_NOT_COMPILED(dpcpp);


void DpcppMemorySpace::raw_free(void* ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void* DpcppMemorySpace::raw_alloc(size_type num_bytes) const
    GKO_NOT_COMPILED(dpcpp);


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(
    const ReferenceMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const GKO_NOT_COMPILED(dpcpp);


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(
    const HostMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const GKO_NOT_COMPILED(dpcpp);


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(
    const CudaMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const GKO_NOT_COMPILED(dpcpp);


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(const CudaUVMSpace*,
                                                           size_type num_bytes,
                                                           const void* src_ptr,
                                                           void* dest_ptr) const
    GKO_NOT_COMPILED(dpcpp);


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(
    const HipMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const GKO_NOT_COMPILED(dpcpp);


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(
    const DpcppMemorySpace*, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const GKO_NOT_COMPILED(dpcpp);


void DpcppMemorySpace::synchronize() const GKO_NOT_COMPILED(dpcpp);


void DpcppExecutor::synchronize() const GKO_NOT_COMPILED(dpcpp);


void DpcppExecutor::run(const Operation& op) const
{
    op.run(std::static_pointer_cast<const DpcppExecutor>(
        this->shared_from_this()));
}


std::shared_ptr<AsyncHandle> DpcppExecutor::run(
    const AsyncOperation& op, std::shared_ptr<AsyncHandle> handle) const
{
    return op.run(
        std::static_pointer_cast<const DpcppExecutor>(this->shared_from_this()),
        handle);
}


int DpcppExecutor::get_num_devices(std::string) { return 0; }


void DpcppExecutor::set_device_property() {}


bool DpcppMemorySpace::verify_memory_to(
    const HostMemorySpace* dest_mem_space) const
{
    // Dummy check
    return this->get_device_type() == "cpu" ||
           this->get_device_type() == "host";
}

bool DpcppMemorySpace::verify_memory_to(
    const DpcppMemorySpace* dest_mem_space) const
{
    // Dummy check
    return dest_mem_space->get_device_type() == this->get_device_type() &&
           dest_mem_space->get_device_id() == this->get_device_id();
}


}  // namespace gko


#define GKO_HOOK_MODULE dpcpp
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
