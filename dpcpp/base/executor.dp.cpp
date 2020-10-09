/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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
        {"gpu", sycl::info::device_type::gpu}};
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


std::shared_ptr<DpcppExecutor> DpcppExecutor::create(
    int device_id, std::shared_ptr<Executor> master, std::string device_type)
{
    return std::shared_ptr<DpcppExecutor>(
        new DpcppExecutor(device_id, std::move(master), device_type));
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
        // TODO: does this work? Or is it needed to go through host?
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


void delete_queue(sycl::queue *queue)
{
    queue->wait();
    delete queue;
}


void DpcppExecutor::set_device_property()
{
    assert(device_id_ < DpcppExecutor::get_num_devices(device_type_));
    auto device = detail::get_devices(device_type_)[device_id_];
    try {
        subgroup_sizes_ =
            device.get_info<cl::sycl::info::device::sub_group_sizes>();
    } catch (cl::sycl::runtime_error &err) {
        GKO_NOT_SUPPORTED(device);
    }
    num_computing_units_ =
        device.get_info<sycl::info::device::max_compute_units>();
    auto max_workitem_sizes =
        device.get_info<sycl::info::device::max_work_item_sizes>();
    // There is no way to get the dimension of a sycl::id object
    for (std::size_t i = 0; i < 3; i++) {
        max_workitem_sizes_.push_back(max_workitem_sizes[i]);
    }
    max_workgroup_size_ =
        device.get_info<sycl::info::device::max_work_group_size>();
    auto *queue = new sycl::queue{device, sycl::property::queue::in_order{}};
    queue_ = std::move(queue_manager<sycl::queue>{queue, delete_queue});
}


}  // namespace gko
