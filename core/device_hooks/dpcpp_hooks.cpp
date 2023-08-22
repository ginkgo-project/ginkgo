/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
#include <ginkgo/core/base/memory.hpp>
#include <ginkgo/core/base/timer.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_sycl_version() noexcept
{
    // We just return the version with a special "not compiled" tag in
    // placeholder modules.
    return {GKO_VERSION_STR, "not compiled"};
}


std::shared_ptr<SyclExecutor> SyclExecutor::create(
    int device_id, std::shared_ptr<Executor> master, std::string device_type,
    sycl_queue_property property)
{
    return std::shared_ptr<SyclExecutor>(
        new SyclExecutor(device_id, std::move(master), device_type, property));
}


void SyclExecutor::populate_exec_info(const machine_topology* mach_topo)
{
    // This method is always called, so cannot throw when not compiled.
}


void OmpExecutor::raw_copy_to(const SyclExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(sycl);


bool OmpExecutor::verify_memory_to(const SyclExecutor* dest_exec) const
{
    // Dummy check
    auto dev_type = dest_exec->get_device_type();
    return dev_type == "cpu" || dev_type == "host";
}


void SyclExecutor::raw_free(void* ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void* SyclExecutor::raw_alloc(size_type num_bytes) const GKO_NOT_COMPILED(sycl);


void SyclExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(sycl);


void SyclExecutor::raw_copy_to(const CudaExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(sycl);


void SyclExecutor::raw_copy_to(const HipExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(sycl);


void SyclExecutor::raw_copy_to(const SyclExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(sycl);


void SyclExecutor::synchronize() const GKO_NOT_COMPILED(sycl);


scoped_device_id_guard SyclExecutor::get_scoped_device_id_guard() const
    GKO_NOT_COMPILED(sycl);


int SyclExecutor::get_num_devices(std::string) { return 0; }


void SyclExecutor::set_device_property(sycl_queue_property property) {}


bool SyclExecutor::verify_memory_to(const OmpExecutor* dest_exec) const
{
    // Dummy check
    return this->get_device_type() == "cpu" ||
           this->get_device_type() == "host";
}

bool SyclExecutor::verify_memory_to(const SyclExecutor* dest_exec) const
{
    // Dummy check
    return dest_exec->get_device_type() == this->get_device_type() &&
           dest_exec->get_device_id() == this->get_device_id();
}


scoped_device_id_guard::scoped_device_id_guard(const SyclExecutor* exec,
                                               int device_id)
    GKO_NOT_COMPILED(sycl);


namespace kernels {
namespace sycl {


void destroy_event(::sycl::event* event) GKO_NOT_COMPILED(sycl);


}  // namespace sycl
}  // namespace kernels


SyclTimer::SyclTimer(std::shared_ptr<const SyclExecutor> exec)
    GKO_NOT_COMPILED(sycl);


void SyclTimer::init_time_point(time_point&) GKO_NOT_COMPILED(sycl);


void SyclTimer::record(time_point&) GKO_NOT_COMPILED(sycl);


void SyclTimer::wait(time_point& time) GKO_NOT_COMPILED(sycl);


std::chrono::nanoseconds SyclTimer::difference_async(const time_point& start,
                                                     const time_point& stop)
    GKO_NOT_COMPILED(sycl);


}  // namespace gko


#define GKO_HOOK_MODULE sycl
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
