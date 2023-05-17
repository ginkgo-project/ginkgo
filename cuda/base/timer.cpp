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

#include <ginkgo/core/base/timer.hpp>


#include <cuda.h>
#include <cuda_runtime_api.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/scoped_device_id.hpp"


namespace gko {


CudaTimer::CudaTimer(std::shared_ptr<const CudaExecutor> exec)
    : exec_{std::move(exec)}
{}


void CudaTimer::init_time_point(time_point& time)
{
    detail::cuda_scoped_device_id_guard guard{exec_->get_device_id()};
    time.type_ = time_point::type::cuda;
    GKO_ASSERT_NO_CUDA_ERRORS(cudaEventCreate(&time.data_.cuda_event));
}


void CudaTimer::record(time_point& time)
{
    detail::cuda_scoped_device_id_guard guard{exec_->get_device_id()};
    GKO_ASSERT(time.type_ == time_point::type::cuda);
    GKO_ASSERT_NO_CUDA_ERRORS(
        cudaEventRecord(time.data_.cuda_event, exec_->get_stream()));
}


void CudaTimer::wait(time_point& time)
{
    detail::cuda_scoped_device_id_guard guard{exec_->get_device_id()};
    GKO_ASSERT(time.type_ == time_point::type::cuda);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaEventSynchronize(time.data_.cuda_event));
}


std::chrono::nanoseconds CudaTimer::difference_async(const time_point& start,
                                                     const time_point& stop)
{
    detail::cuda_scoped_device_id_guard guard{exec_->get_device_id()};
    GKO_ASSERT(start.type_ == time_point::type::cuda);
    GKO_ASSERT(stop.type_ == time_point::type::cuda);
    GKO_ASSERT_NO_CUDA_ERRORS(cudaEventSynchronize(stop.data_.cuda_event));
    float ms{};
    GKO_ASSERT_NO_CUDA_ERRORS(cudaEventElapsedTime(&ms, start.data_.cuda_event,
                                                   stop.data_.cuda_event));
    return std::chrono::nanoseconds{static_cast<int64>(ms * double{1e6})};
}


}  // namespace gko
