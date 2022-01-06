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

#include <ginkgo/core/base/async_handle.hpp>


#include <iostream>


#include <cuda_runtime.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/device_guard.hpp"
#include "cuda/base/stream_bindings.hpp"


namespace gko {


CudaAsyncHandle::CudaAsyncHandle(create_type c_type)
{
    if (c_type == create_type::non_blocking) {
        this->handle_ = handle_manager<CUstream_st>(
            kernels::cuda::stream::create_non_blocking(),
            [](cudaStream_t stream) {
                kernels::cuda::stream::destroy(stream);
            });
    } else if (c_type == create_type::default_blocking) {
        this->handle_ = handle_manager<CUstream_st>(
            kernels::cuda::stream::create_default_blocking(),
            [](cudaStream_t stream) {
                kernels::cuda::stream::destroy(stream);
            });
    } else if (c_type == create_type::legacy_blocking) {
        this->handle_ = handle_manager<CUstream_st>(
            cudaStreamLegacy,
            [](cudaStream_t stream) { /* Don't delete the default stream*/ });
    }
}


void CudaAsyncHandle::get_result() GKO_NOT_IMPLEMENTED;


void CudaAsyncHandle::wait()
{
    GKO_ASSERT_NO_CUDA_ERRORS(cudaStreamSynchronize(this->get_handle()));
}


void CudaAsyncHandle::wait_for(const std::chrono::duration<int>& time)
    GKO_NOT_IMPLEMENTED;


void CudaAsyncHandle::wait_until(
    const std::chrono::time_point<std::chrono::steady_clock>& time)
    GKO_NOT_IMPLEMENTED;


}  // namespace gko
