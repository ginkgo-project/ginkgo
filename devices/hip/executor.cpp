// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/executor.hpp>


namespace gko {


std::shared_ptr<Executor> HipExecutor::get_master() noexcept { return master_; }


std::shared_ptr<const Executor> HipExecutor::get_master() const noexcept
{
    return master_;
}


bool HipExecutor::verify_memory_to(const HipExecutor* dest_exec) const
{
    return this->get_device_id() == dest_exec->get_device_id();
}


bool HipExecutor::verify_memory_to(const CudaExecutor* dest_exec) const
{
#if GINKGO_HIP_PLATFORM_NVCC
    return this->get_device_id() == dest_exec->get_device_id();
#else
    return false;
#endif
}


}  // namespace gko
