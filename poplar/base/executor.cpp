// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/executor.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <string>

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


void OmpExecutor::raw_copy_to(const PoplarExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    // TODO
}


bool OmpExecutor::verify_memory_to(const PoplarExecutor* dest_exec) const
{
    // TODO
    return false;
}


std::shared_ptr<PoplarExecutor> PoplarExecutor::create(
    int device_id, std::shared_ptr<Executor> master)
{
    // TODO
    return std::shared_ptr<PoplarExecutor>(
        new PoplarExecutor(device_id, std::move(master)));
}


void PoplarExecutor::populate_exec_info(const machine_topology* mach_topo)
{
    // Closest CPUs, NUMA node can be updated when there is a way to identify
    // the device itself, which is currently not available with DPC++.
}


void PoplarExecutor::raw_free(void* ptr) const noexcept
{
    // TODO
    GKO_NOT_IMPLEMENTED;
}


void* PoplarExecutor::raw_alloc(size_type num_bytes) const
{
    // TODO
    void* dev_ptr;
    return dev_ptr;
}


void PoplarExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                                 const void* src_ptr, void* dest_ptr) const
{
    // TODO
    GKO_NOT_IMPLEMENTED;
}


void PoplarExecutor::raw_copy_to(const CudaExecutor* dest, size_type num_bytes,
                                 const void* src_ptr, void* dest_ptr) const
{
    // TODO: later when possible, if we have DPC++ with a CUDA backend
    // support/compiler, we could maybe support native copies?
    GKO_NOT_SUPPORTED(dest);
}


void PoplarExecutor::raw_copy_to(const HipExecutor* dest, size_type num_bytes,
                                 const void* src_ptr, void* dest_ptr) const
{
    GKO_NOT_SUPPORTED(dest);
}


void PoplarExecutor::raw_copy_to(const PoplarExecutor* dest,
                                 size_type num_bytes, const void* src_ptr,
                                 void* dest_ptr) const
{
    // TODO
    GKO_NOT_IMPLEMENTED;
}


void PoplarExecutor::synchronize() const
{
    // TODO
    GKO_NOT_IMPLEMENTED;
}

scoped_device_id_guard PoplarExecutor::get_scoped_device_id_guard() const
{
    return {this, this->get_device_id()};
}


std::string PoplarExecutor::get_description() const
{
    // TODO
    return "Not implemented";
}


int PoplarExecutor::get_num_devices(std::string device_type)
{
    // TODO
    return 0;
}


bool PoplarExecutor::verify_memory_to(const OmpExecutor* dest_exec) const
{
    // TODO
    return false;
}

bool PoplarExecutor::verify_memory_to(const PoplarExecutor* dest_exec) const
{
    // TODO
    return false;
}


namespace kernels {
namespace poplar {


void destroy_event(poplar::event* event) { delete event; }


std::string get_device_name(int device_id) { return "Not implemented"; }


}  // namespace poplar
}  // namespace kernels
}  // namespace gko
