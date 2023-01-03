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

#include <ginkgo/core/base/executor.hpp>


namespace gko {


std::shared_ptr<Executor> CudaExecutor::get_master() noexcept
{
    return master_;
}


std::shared_ptr<const Executor> CudaExecutor::get_master() const noexcept
{
    return master_;
}


bool CudaExecutor::verify_memory_to(const CudaExecutor* dest_exec) const
{
    return this->get_device_id() == dest_exec->get_device_id();
}


bool CudaExecutor::verify_memory_to(const HipExecutor* dest_exec) const
{
#if GINKGO_HIP_PLATFORM_NVCC
    return this->get_device_id() == dest_exec->get_device_id();
#else
    return false;
#endif
}


void CudaExecutor::increase_num_execs(unsigned device_id)
{
#ifdef GKO_COMPILING_CUDA_DEVICE
    // increase the Cuda Device count only when ginkgo build cuda
    std::lock_guard<std::mutex> guard(nvidia_device::get_mutex(device_id));
    nvidia_device::get_num_execs(device_id)++;
#endif  // GKO_COMPILING_CUDA_DEVICE
}


void CudaExecutor::decrease_num_execs(unsigned device_id)
{
#ifdef GKO_COMPILING_CUDA_DEVICE
    // increase the Cuda Device count only when ginkgo build cuda
    std::lock_guard<std::mutex> guard(nvidia_device::get_mutex(device_id));
    nvidia_device::get_num_execs(device_id)--;
#endif  // GKO_COMPILING_CUDA_DEVICE
}


unsigned CudaExecutor::get_num_execs(unsigned device_id)
{
    std::lock_guard<std::mutex> guard(nvidia_device::get_mutex(device_id));
    return nvidia_device::get_num_execs(device_id);
}


}  // namespace gko
