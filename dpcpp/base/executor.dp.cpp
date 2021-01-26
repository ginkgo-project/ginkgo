/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <string>


#include <CL/sycl.hpp>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace detail {


const std::vector<sycl::device> get_devices(std::string device_type)
{
    std::map<std::string, sycl::info::device_type> device_type_map{
        {"accelerator", sycl::info::device_type::accelerator},
        {"all", sycl::info::device_type::all},
        {"cpu", sycl::info::device_type::cpu},
        {"host", sycl::info::device_type::host},
        {"gpu", sycl::info::device_type::gpu}};
    std::for_each(device_type.begin(), device_type.end(),
                  [](char &c) { c = std::tolower(c); });
    return sycl::device::get_devices(device_type_map.at(device_type));
}


}  // namespace detail


void OmpExecutor::raw_copy_to(const DpcppExecutor *dest, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    if (num_bytes > 0) {
        dest->get_queue()->memcpy(dest_ptr, src_ptr, num_bytes).wait();
    }
}


bool OmpExecutor::verify_memory_to(const DpcppExecutor *dest_exec) const
{
    auto device = detail::get_devices(
        dest_exec->get_device_type())[dest_exec->get_device_id()];
    return device.is_host() || device.is_cpu();
}


std::shared_ptr<DpcppExecutor> DpcppExecutor::create(
    int device_id, std::shared_ptr<Executor> master, std::string device_type)
{
    return std::shared_ptr<DpcppExecutor>(
        new DpcppExecutor(device_id, std::move(master), device_type));
}


void DpcppExecutor::populate_exec_info(const MachineTopology *mach_topo)
{
    // Closest CPUs, NUMA node can be updated when there is a way to identify
    // the device itself, which is currently not available with DPC++.
}


void DpcppExecutor::raw_free(void *ptr) const noexcept
{
    sycl::free(ptr, queue_->get_context());
}


void *DpcppExecutor::raw_alloc(size_type num_bytes) const
{
    void *dev_ptr = sycl::malloc_device(num_bytes, *queue_.get());
    GKO_ENSURE_ALLOCATED(dev_ptr, "DPC++", num_bytes);
    return dev_ptr;
}


void DpcppExecutor::raw_copy_to(const OmpExecutor *, size_type num_bytes,
                                const void *src_ptr, void *dest_ptr) const
{
    if (num_bytes > 0) {
        queue_->memcpy(dest_ptr, src_ptr, num_bytes).wait();
    }
}


void DpcppExecutor::raw_copy_to(const CudaExecutor *dest, size_type num_bytes,
                                const void *src_ptr, void *dest_ptr) const
{
    // TODO: later when possible, if we have DPC++ with a CUDA backend
    // support/compiler, we could maybe support native copies?
    GKO_NOT_SUPPORTED(dest);
}


void DpcppExecutor::raw_copy_to(const HipExecutor *dest, size_type num_bytes,
                                const void *src_ptr, void *dest_ptr) const
{
    GKO_NOT_SUPPORTED(dest);
}


void DpcppExecutor::raw_copy_to(const DpcppExecutor *dest, size_type num_bytes,
                                const void *src_ptr, void *dest_ptr) const
{
    if (num_bytes > 0) {
        dest->get_queue()->memcpy(dest_ptr, src_ptr, num_bytes).wait();
    }
}


void DpcppExecutor::synchronize() const { queue_->wait_and_throw(); }


void DpcppExecutor::run(const Operation &op) const
{
    this->template log<log::Logger::operation_launched>(this, &op);
    op.run(std::static_pointer_cast<const DpcppExecutor>(
        this->shared_from_this()));
    this->template log<log::Logger::operation_completed>(this, &op);
}


int DpcppExecutor::get_num_devices(std::string device_type)
{
    return detail::get_devices(device_type).size();
}


bool DpcppExecutor::verify_memory_to(const OmpExecutor *dest_exec) const
{
    auto device = detail::get_devices(
        get_exec_info().device_type)[get_exec_info().device_id];
    return device.is_host() || device.is_cpu();
}

bool DpcppExecutor::verify_memory_to(const DpcppExecutor *dest_exec) const
{
    auto device = detail::get_devices(
        get_exec_info().device_type)[get_exec_info().device_id];
    auto other_device = detail::get_devices(
        dest_exec->get_device_type())[dest_exec->get_device_id()];
    return ((device.is_host() || device.is_cpu()) &&
            (other_device.is_host() || other_device.is_cpu())) ||
           (device.get_info<cl::sycl::info::device::device_type>() ==
                other_device.get_info<cl::sycl::info::device::device_type>() &&
            device.get() == other_device.get());
}


namespace detail {


void delete_queue(sycl::queue *queue)
{
    queue->wait();
    delete queue;
}


}  // namespace detail


void DpcppExecutor::set_device_property()
{
    assert(this->get_exec_info().device_id <
           DpcppExecutor::get_num_devices(this->get_exec_info().device_type));
    auto device = detail::get_devices(
        this->get_exec_info().device_type)[this->get_exec_info().device_id];
    if (!device.is_host()) {
        try {
            auto subgroup_sizes =
                device.get_info<cl::sycl::info::device::sub_group_sizes>();
            for (auto &i : subgroup_sizes) {
                this->get_exec_info().subgroup_sizes.push_back(i);
            }
        } catch (cl::sycl::runtime_error &err) {
            GKO_NOT_SUPPORTED(device);
        }
    }
    this->get_exec_info().num_computing_units = static_cast<int>(
        device.get_info<sycl::info::device::max_compute_units>());
    const auto subgroup_sizes = this->get_exec_info().subgroup_sizes;
    if (subgroup_sizes.size()) {
        this->get_exec_info().max_subgroup_size = static_cast<int>(
            *std::max_element(subgroup_sizes.begin(), subgroup_sizes.end()));
    }
    this->get_exec_info().max_workgroup_size = static_cast<int>(
        device.get_info<sycl::info::device::max_work_group_size>());
    auto max_workitem_sizes =
        device.get_info<sycl::info::device::max_work_item_sizes>();
    // Get the max dimension of a sycl::id object
    auto max_work_item_dimensions =
        device.get_info<sycl::info::device::max_work_item_dimensions>();
    for (uint32 i = 0; i < max_work_item_dimensions; i++) {
        this->get_exec_info().max_workitem_sizes.push_back(
            max_workitem_sizes[i]);
    }
    // Here we declare the queue with the property `in_order` which ensures the
    // kernels are executed in the submission order. Otherwise, calls to
    // `wait()` would be needed after every call to a DPC++ function or kernel.
    // For example, without `in_order`, doing a copy, a kernel, and a copy, will
    // not necessarily happen in that order by default, which we need to avoid.
    auto *queue = new sycl::queue{device, sycl::property::queue::in_order{}};
    queue_ = std::move(queue_manager<sycl::queue>{queue, detail::delete_queue});
}


}  // namespace gko
