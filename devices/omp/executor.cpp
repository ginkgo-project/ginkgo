// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/executor.hpp>


#include <cstdlib>
#include <cstring>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


void OmpExecutor::populate_exec_info(const machine_topology* mach_topo)
{
    auto num_cores =
        (mach_topo->get_num_cores() == 0 ? 1 : mach_topo->get_num_cores());
    auto num_pus =
        (mach_topo->get_num_pus() == 0 ? 1 : mach_topo->get_num_pus());
    this->get_exec_info().num_computing_units = num_cores;
    this->get_exec_info().num_pu_per_cu = num_pus / num_cores;
}


void OmpExecutor::raw_free(void* ptr) const noexcept
{
    return alloc_->deallocate(ptr);
}


std::shared_ptr<Executor> OmpExecutor::get_master() noexcept
{
    return this->shared_from_this();
}


std::shared_ptr<const Executor> OmpExecutor::get_master() const noexcept
{
    return this->shared_from_this();
}


void* OmpExecutor::raw_alloc(size_type num_bytes) const
{
    return alloc_->allocate(num_bytes);
}


void OmpExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
{
    if (num_bytes > 0) {
        std::memcpy(dest_ptr, src_ptr, num_bytes);
    }
}


void OmpExecutor::synchronize() const
{
    // This is a no-op for single-threaded OMP
    // TODO: change when adding support for multi-threaded OMP execution
}


scoped_device_id_guard OmpExecutor::get_scoped_device_id_guard() const
{
    return {this, 0};
}


}  // namespace gko
