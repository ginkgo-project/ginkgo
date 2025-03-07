// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/executor.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <string>

#include <csl/Engine.hpp>
#include <csl/IPUModel.hpp>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


void OmpExecutor::raw_copy_to(const CslExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    // TODO
}


bool OmpExecutor::verify_memory_to(const CslExecutor* dest_exec) const
{
    // TODO
    return false;
}


std::shared_ptr<CslExecutor> CslExecutor::create(
    int device_id, std::shared_ptr<Executor> master)
{
    // TODO
    return std::shared_ptr<CslExecutor>(
        new CslExecutor(device_id, std::move(master)));
}


void CslExecutor::populate_exec_info(const machine_topology* mach_topo)
{
    // Closest CPUs, NUMA node can be updated when there is a way to identify
    // the device itself, which is currently not available with DPC++.
}


void CslExecutor::raw_free(void* ptr) const noexcept
{
    // TODO
    GKO_NOT_IMPLEMENTED;
}


void* CslExecutor::raw_alloc(size_type num_bytes) const
{
    // TODO
    void* dev_ptr;
    return dev_ptr;
}


void CslExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    // TODO
    GKO_NOT_IMPLEMENTED;
}


void CslExecutor::raw_copy_to(const CudaExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    // TODO: later when possible, if we have DPC++ with a CUDA backend
    // support/compiler, we could maybe support native copies?
    GKO_NOT_SUPPORTED(dest);
}


void CslExecutor::raw_copy_to(const HipExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    GKO_NOT_SUPPORTED(dest);
}


void CslExecutor::raw_copy_to(const CslExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    // TODO
    GKO_NOT_IMPLEMENTED;
}


void CslExecutor::synchronize() const
{
    // TODO
    GKO_NOT_IMPLEMENTED;
}

scoped_device_id_guard CslExecutor::get_scoped_device_id_guard() const
{
    return {this, this->get_device_id()};
}


std::string CslExecutor::get_description() const
{
    // TODO
    return "Not implemented";
}


int CslExecutor::get_num_devices(std::string device_type)
{
    // TODO
    return 0;
}


bool CslExecutor::verify_memory_to(const OmpExecutor* dest_exec) const
{
    // TODO
    return false;
}

bool CslExecutor::verify_memory_to(const CslExecutor* dest_exec) const
{
    // TODO
    return false;
}


namespace kernels {
namespace csl {


void destroy_event(csl::event* event) { delete event; }


std::string get_device_name(int device_id) { return "Not implemented"; }


}  // namespace csl
}  // namespace kernels
}  // namespace gko
