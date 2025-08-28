// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/executor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>

#include <csl/cerebras_interface.hpp>
#include <csl/cerebras_layout.hpp>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {

// class CerebrasImpl
//{
// public:
//     CerebrasImpl()
//     {
//         cerebras_device_ = std::make_unique<CerebrasInterface>(true);
//     }
//
//     void copy_h2d(std::string target_var, float* vec, size_t vec_size, int
//     offset1,
//         int offset2, int size1, int size2, int elements_per_pe, bool
//         streaming, bool nonblocking)
//     {
//         cerebras_device_->copy_h2d(target_var, vec, vec_size, offset1,
//         offset2,
//             size1, size2, elements_per_pe, streaming, nonblocking);
//     }
//
//     void copy_d2h(std::string target_var, float* vec, size_t vec_size, int
//     offset1,
//         int offset2, int size1, int size2, int elements_per_pe, bool
//         streaming, bool nonblocking)
//     {
//         cerebras_device_->copy_d2h(target_var, vec, vec_size, offset1,
//         offset2,
//             size1, size2, elements_per_pe, streaming, nonblocking);
//     }
//
//     void call_func(std::string func_name, bool nonblocking = false)
//     {
//         cerebras_device_->call_func(func_name, nonblocking);
//     }
//
// private:
//     std::unique_ptr<CerebrasInterface> cerebras_device_;
// };


void CslExecutor::init_handle()
{
    this->cerebras_handle_ =
        handle_manager<CerebrasContext>(CerebrasInterface(true));
}


CerebrasContext* CslExecutor::get_handle() const { return cerebras_context_; }

void OmpExecutor::raw_copy_to(const CslExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    std::cout << "copying " << num_bytes << "bytes from Omp to Csl!"
              << std::endl;
    std::memcpy(dest_ptr, src_ptr, num_bytes);
}


void DpcppExecutor::raw_copy_to(const CslExecutor* dest, size_type num_bytes,
                                const void* src_ptr, void* dest_ptr) const
{
    GKO_NOT_SUPPORTED(dest);
}


void CudaExecutor::raw_copy_to(const CslExecutor* dest, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
{
    GKO_NOT_SUPPORTED(dest);
}


void HipExecutor::raw_copy_to(const CslExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    GKO_NOT_SUPPORTED(dest);
}


std::shared_ptr<CslExecutor> CslExecutor::create(
    int device_id, std::shared_ptr<Executor> master)
{
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
    std::cout << "freeing memory on csl!" << std::endl;
    std::free(ptr);
}


void* CslExecutor::raw_alloc(size_type num_bytes) const
{
    std::cout << "allocating " << num_bytes << "memory on csl!" << std::endl;
    return malloc(num_bytes);
}


void CslExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    std::cout << "copying " << num_bytes << "bytes from csl to omp!"
              << std::endl;
    std::memcpy(dest_ptr, src_ptr, num_bytes);
}


void CslExecutor::raw_copy_to(const CudaExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
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
    std::memcpy(dest_ptr, src_ptr, num_bytes);
}


void CslExecutor::raw_copy_to(const DpcppExecutor* dest, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    GKO_NOT_SUPPORTED(dest);
}


void CslExecutor::synchronize() const
{
    // TODO
    GKO_NOT_IMPLEMENTED;
}


std::string CslExecutor::get_description() const
{
    return "CslExecutor on device with host " +
           this->get_master()->get_description();
}


int CslExecutor::get_num_devices(std::string device_type)
{
    // TODO
    return 0;
}


scoped_device_id_guard CslExecutor::get_scoped_device_id_guard() const
{
    return {this, this->get_device_id()};
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


}  // namespace gko
