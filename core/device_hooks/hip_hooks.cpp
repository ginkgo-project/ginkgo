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

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_hip_version() noexcept
{
    // We just return 1.1.0 with a special "not compiled" tag in placeholder
    // modules.
    return {1, 1, 0, "not compiled"};
}


std::shared_ptr<HipExecutor> HipExecutor::create(
    int device_id, std::shared_ptr<Executor> master, bool device_reset)
{
    return std::shared_ptr<HipExecutor>(
        new HipExecutor(device_id, std::move(master), device_reset));
}


void OmpExecutor::raw_copy_to(const HipExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
    GKO_NOT_COMPILED(hip);


void HipExecutor::raw_free(void *ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void *HipExecutor::raw_alloc(size_type num_bytes) const GKO_NOT_COMPILED(hip);


void HipExecutor::raw_copy_to(const OmpExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
    GKO_NOT_COMPILED(hip);


void HipExecutor::raw_copy_to(const CudaExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
    GKO_NOT_COMPILED(hip);


void HipExecutor::raw_copy_to(const HipExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
    GKO_NOT_COMPILED(hip);


void HipExecutor::synchronize() const GKO_NOT_COMPILED(hip);


void HipExecutor::run(const Operation &op) const
{
    op.run(
        std::static_pointer_cast<const HipExecutor>(this->shared_from_this()));
}


std::string HipError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


std::string HipblasError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


std::string HipsparseError::get_error(int64)
{
    return "ginkgo HIP module is not compiled";
}


int HipExecutor::get_num_devices() { return 0; }


void HipExecutor::set_gpu_property() {}


void HipExecutor::init_handles() {}


}  // namespace gko


#define GKO_HOOK_MODULE hip
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
