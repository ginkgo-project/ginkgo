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

#include <ginkgo/core/base/memory_space.hpp>


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
                  [](char& c) { c = std::tolower(c); });
    return sycl::device::get_devices(device_type_map.at(device_type));
}


void delete_queue(sycl::queue* queue)
{
    queue->wait();
    delete queue;
}


}  // namespace detail


int DpcppMemorySpace::get_num_devices(std::string device_type)
{
    return detail::get_devices(device_type).size();
}


std::shared_ptr<AsyncHandle> HostMemorySpace::raw_copy_to(
    const DpcppMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const
{
    auto cpy_lambda = [=]() {
        if (num_bytes > 0) {
            dest->get_queue()->memcpy(dest_ptr, src_ptr, num_bytes).wait();
        }
    };
    return std::async(std::launch::async, cpy_lambda);
}


std::shared_ptr<AsyncHandle> ReferenceMemorySpace::raw_copy_to(
    const DpcppMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const
{
    auto cpy_lambda = [=]() {
        if (num_bytes > 0) {
            dest->get_queue()->memcpy(dest_ptr, src_ptr, num_bytes).wait();
        }
    };
    return std::async(std::launch::async, cpy_lambda);
}


void DpcppMemorySpace::raw_free(void* ptr) const noexcept
{
    // the free function may syncronize excution or not, which depends on
    // implementation or backend, so it is not guaranteed.
    // TODO: maybe a light wait implementation?
    try {
        queue_->wait_and_throw();
        sycl::free(ptr, queue_->get_context());
    } catch (cl::sycl::exception& err) {
#if GKO_VERBOSE_LEVEL >= 1
        // Unfortunately, if memory free fails, there's not much we can do
        std::cerr << "Unrecoverable Dpcpp error on device "
                  << this->get_device_id() << " in " << __func__ << ": "
                  << err.what() << std::endl
                  << "Exiting program" << std::endl;
#endif  // GKO_VERBOSE_LEVEL >= 1
        // OpenCL error code use 0 for CL_SUCCESS and negative number for others
        // error. if the error is not from OpenCL, it will return CL_SUCCESS.
        int err_code = err.get_cl_code();
        // if return CL_SUCCESS, exit 1 as DPCPP error.
        if (err_code == 0) {
            err_code = 1;
        }
        std::exit(err_code);
    }
}


void* DpcppMemorySpace::raw_alloc(size_type num_bytes) const
{
    void* dev_ptr = sycl::malloc_device(num_bytes, *queue_.get());
    GKO_ENSURE_ALLOCATED(dev_ptr, "DPC++", num_bytes);
    return dev_ptr;
}


DpcppMemorySpace::DpcppMemorySpace(int device_id, std::string device_type)
    : device_id_(device_id), device_type_(device_type)
{
    assert(device_id < max_devices);
    // Here we declare the queue with the property `in_order` which ensures the
    // kernels are executed in the submission order. Otherwise, calls to
    // `wait()` would be needed after every call to a DPC++ function or kernel.
    // For example, without `in_order`, doing a copy, a kernel, and a copy, will
    // not necessarily happen in that order by default, which we need to avoid.
    auto* queue = new sycl::queue{device, sycl::property::queue::in_order{}};
    queue_ = std::move(queue_manager<sycl::queue>{queue, detail::delete_queue});
}


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(
    const DpcppMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const
{
    auto cpy_lambda = [=]() {
        if (num_bytes > 0) {
            // If the queue is different and is not cpu/host, the queue can not
            // transfer the data to another queue (on the same device)
            // Note. it could be changed when we ensure the behavior is
            // expected.
            auto queue = this->get_queue();
            auto dest_queue = dest->get_queue();
            auto device = queue->get_device();
            auto dest_device = dest_queue->get_device();
            if (((device.is_host() || device.is_cpu()) &&
                 (dest_device.is_host() || dest_device.is_cpu())) ||
                (queue == dest_queue)) {
                dest->get_queue()->memcpy(dest_ptr, src_ptr, num_bytes).wait();
            } else {
                // the memcpy only support host<->device or itself memcpy
                GKO_NOT_SUPPORTED(dest);
            }
        }
    };
    return std::async(std::launch::async, cpy_lambda);
}


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(
    const HostMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const
{
    auto cpy_lambda = [=]() {
        if (num_bytes > 0) {
            queue_->memcpy(dest_ptr, src_ptr, num_bytes).wait();
        }
    };
    return std::async(std::launch::async, cpy_lambda);
}


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(
    const ReferenceMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const
{
    auto cpy_lambda = [=]() {
        if (num_bytes > 0) {
            queue_->memcpy(dest_ptr, src_ptr, num_bytes).wait();
        }
    };
    return std::async(std::launch::async, cpy_lambda);
}


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(
    const HipMemorySpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const GKO_NOT_SUPPORTED(this);


std::shared_ptr<AsyncHandle> DpcppMemorySpace::raw_copy_to(
    const CudaUVMSpace* dest, size_type num_bytes, const void* src_ptr,
    void* dest_ptr) const GKO_NOT_SUPPORTED(this);


bool DpcppMemorySpace::verify_memory_to(
    const HostMemorySpace* dest_mem_space) const
{
    auto device = detail::get_devices(device_type_[device_id_]);
    return device.is_host() || device.is_cpu();
}

bool DpcppMemorySpace::verify_memory_to(
    const DpcppMemorySpace* dest_mem_space) const
{
    // If the queue is different and is not cpu/host, the queue can not access
    // the data from another queue (on the same device)
    // Note. it could be changed when we ensure the behavior is expected.
    auto queue = this->get_queue();
    auto dest_queue = dest_mem_space->get_queue();
    auto device = queue->get_device();
    auto dest_device = dest_queue->get_device();
    return ((device.is_host() || device.is_cpu()) &&
            (dest_device.is_host() || dest_device.is_cpu())) ||
           (queue == dest_queue);
}


}  // namespace gko
